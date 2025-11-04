# -*- coding: utf-8 -*-

import boto3
import botocore.exceptions
import json
import re
import time
import logging
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

MODEL_ID = 'meta.llama3-8b-instruct-v1:0'
REGION_NAME = 'us-east-1'
client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (Totalmente Nova) ---

def translate_example_with_context(example_in):
    """
    Chama o Llama 3 no Bedrock para traduzir um EXEMPLO INTEIRO (contexto + 3 alvos).
    Instrui o modelo a retornar um JSON estruturado.
    """
    
    # Extrai os textos em inglês do exemplo original
    context_en = example_in['context']
    # O 'sentences' é um dicionário de listas paralelas
    sentences_en = example_in['sentences']['sentence']

    # Prompt de sistema que instrui o modelo a ser um tradutor e a retornar JSON
    system_prompt = (
        "Você é um tradutor especialista de Inglês para Português do Brasil. "
        "Sua tarefa é traduzir o contexto e a lista de frases-alvo, mantendo a "
        "relação semântica entre eles. "
        "Responda *apenas* com um objeto JSON válido, seguindo o esquema fornecido."
    )
    
    # O prompt do usuário agora envia todos os dados de uma vez
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
    
    # Formato do prompt Llama 3 Instruct
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{""" # Inicia a resposta com "{" para forçar o JSON

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 1024, # Aumenta o tamanho para caber o JSON completo
        "temperature": 0.1,
        "top_p": 0.9
    })

    retries = 3
    delay = 5
    for i in range(retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Limpa e tenta decodificar a resposta JSON do LLM
            # A resposta do Llama 3 pode ser: `{"contexto_traduzido": ... }}`
            json_response_str = "{" + response_body['generation']
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
            logging.error(f"Erro ao processar exemplo (ID: {example_in['id']}): {e}")
            logging.error(f"Resposta JSON inválida: {response_body['generation']}")
            return None # Retorna None em caso de falha
    
    logging.error(f"Excedeu retentativas para o exemplo (ID: {example_in['id']}).")
    return None


# --- 3. FUNÇÃO AUXILIAR (COMO ANTES) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (MODIFICADA) ---

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

    # --- ETAPA DE SALVAMENTO (COMO ANTES) ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_llama3_contexto_correto.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido via LLM) é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
