# -*- coding: utf-8 -*-

import boto3
import botocore.exceptions
import json
import re
import time
import logging
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES (MODIFICADAS) ---

# O MODEL_ID foi trocado para o Claude 3.5 Sonnet
MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
REGION_NAME = 'us-east-1'
client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (Totalmente Modificada para Claude) ---

def translate_example_with_context(example_in):
    """
    Chama o Claude 3.5 Sonnet no Bedrock para traduzir um EXEMPLO INTEIRO.
    Instrui o modelo a retornar um JSON estruturado.
    """
    
    context_en = example_in['context']
    sentences_en = example_in['sentences']['sentence']

    # O prompt de sistema é separado no Claude 3
    system_prompt = (
        "Você é um tradutor especialista de Inglês para Português do Brasil. "
        "Sua tarefa é traduzir o contexto e a lista de frases-alvo, mantendo a "
        "relação semântica entre eles. "
        "Responda *apenas* com um objeto JSON válido, começando com '{' e terminando com '}'. "
        "Não inclua nenhum texto explicativo antes ou depois do JSON."
    )
    
    # O prompt do usuário contém os dados e a estrutura de saída
    user_prompt = f"""Traduza o seguinte conjunto de textos:

Contexto em Inglês:
"{context_en}"

Frases-Alvo em Inglês:
1. "{sentences_en[0]}"
2. "{sentences_en[1]}"
3. "{sentences_en[2]}"

Formato de Saída JSON Obrigatório:
{{
  "contexto_traduzido": "...",
  "alvos_traduzidos": [
    "tradução do alvo 1",
    "tradução do alvo 2",
    "tradução do alvo 3"
  ]
}}"""
    
    # --- Lógica de Pre-filling para forçar JSON ---
    # Começamos a resposta do assistente com '{'
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{"}]}
    ]
    
    # Corpo da requisição formatado para a API de Messages do Claude 3
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9
    })

    retries = 3
    delay = 5
    for i in range(retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Extrai o texto da resposta do Claude
            # A resposta do Claude já inclui o nosso pre-fill "{"
            json_response_str = "{" + response_body['content'][0]['text']
            translated_data = json.loads(json_response_str)
            
            # Valida se a estrutura está correta
            if 'contexto_traduzido' in translated_data and len(translated_data.get('alvos_traduzidos', [])) == 3:
                return translated_data
            else:
                raise Exception(f"JSON retornado com estrutura inválida: {translated_data}")

        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"Throttling... retentativa em {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return None
        except Exception as e:
            raw_response_text = "N/A"
            if 'response_body' in locals() and 'content' in response_body:
                raw_response_text = response_body['content'][0]['text']
            elif 'response_body' in locals():
                raw_response_text = str(response_body)

            logging.error(f"Erro ao processar exemplo (ID: {example_in['id']}): {e}")
            logging.error(f"Resposta JSON inválida: {raw_response_text}")
            return None # Retorna None em caso de falha
    
    logging.error(f"Excedeu retentativas para o exemplo (ID: {example_in['id']}).")
    return None


# --- 3. FUNÇÃO AUXILIAR (Sem alterações) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (Quase idêntica) ---

def traduzir_e_recriar_estrutura_com_llm():
    """
    Executa o pipeline completo de tradução usando o Bedrock
    e recria a estrutura original do Stereoset.
    """
    
    reconstructed_data = {}
    
    # Itera sobre 'intersentence' e 'intrasentence'
    for config in CONFIGS:
        print(f"Carregando e processando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        
        new_examples_list = []
        
        # --- ETAPA DE TRADUÇÃO E RECONSTRUÇÃO (TUDO EM UMA) ---
        for original_example in tqdm(dataset, desc=f"Traduzindo {config}"):
            
            # Chama a API para traduzir o exemplo completo
            translated_data = translate_example_with_context(original_example)
            
            # Pula o exemplo se a tradução falhar
            if translated_data is None:
                continue
                
            # Extrai os textos traduzidos
            context_pt = sanitize_text(translated_data['contexto_traduzido'])
            alvos_pt = [sanitize_text(t) for t in translated_data['alvos_traduzidos']]
            
            # Monta a nova estrutura do exemplo mantendo os dados originais
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": context_pt,
                "sentences": []
            }
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # Recria a estrutura interna de 'labels'
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({
                        "human_id": human_ids[j],
                        "label": INNER_LABEL_MAP[inner_int_labels[j]]
                    })

                # Cria o novo objeto de sentença com o texto traduzido
                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": alvos_pt[i], # Usa o texto da lista de alvos traduzidos
                    "labels": recreated_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO (Nome do arquivo alterado) ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    # Nome do arquivo de saída foi atualizado
    output_path = f"stereoset_{DATASET_SPLIT}_pt_claude3-5-sonnet.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido via Claude 3.5 Sonnet) é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
