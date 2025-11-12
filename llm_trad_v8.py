# -*- coding: utf-8 -*-

import boto3
import botocore.exceptions
import json
import re
import time
import logging
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES (Sem alterações) ---

MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
REGION_NAME = 'us-east-1'
client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (Modificada para prompts adaptativos) ---

def translate_example_with_context(example_in, task_type):
    """
    Chama o Claude 3.5 Sonnet com um prompt adaptativo dependendo da tarefa.
    """
    
    context_en = example_in['context']
    sentences_en = example_in['sentences']['sentence']

    # --- INÍCIO DA CORREÇÃO: PROMPTS ADAPTATIVOS ---
    if task_type == 'intersentence':
        # Prompt "Forte" para a tarefa intersentence (a que está falhando)
        system_prompt = (
            "Você é um tradutor especialista de Inglês para Português do Brasil. "
            "Sua tarefa é traduzir um CONTEXTO e, em seguida, traduzir três FRASES-ALVO separadas que "
            "são uma *continuação* direta desse contexto. "
            "É absolutamente crucial que a tradução das frases-alvo mantenha a relação semântica "
            "de continuação com o contexto traduzido. "
            "Responda *apenas* com um objeto JSON válido, começando com '{' e terminando com '}'. "
            "Não inclua nenhum texto explicativo."
        )
        
        user_prompt = f"""Traduza o seguinte conjunto de textos, onde as "Frases-Alvo" são uma *continuação* do "Contexto".

Contexto em Inglês:
"{context_en}"

Frases-Alvo em Inglês (que seguem o contexto):
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
    
    else: # task_type == 'intrasentence'
        # Prompt Padrão (o que já funcionava bem para intrasentence)
        system_prompt = (
            "Você é um tradutor especialista de Inglês para Português do Brasil. "
            "Sua tarefa é traduzir o contexto e a lista de frases-alvo, mantendo a "
            "relação semântica entre eles. "
            "Responda *apenas* com um objeto JSON válido, começando com '{' e terminando com '}'."
        )
        
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
    # --- FIM DA CORREÇÃO ---

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{"}]}
    ]
    
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
            
            json_response_str = "{" + response_body['content'][0]['text']
            translated_data = json.loads(json_response_str)
            
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
            return None
    
    logging.error(f"Excedeu retentativas para o exemplo (ID: {example_in['id']}).")
    return None


# --- 3. FUNÇÃO AUXILIAR (Sem alterações) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (Modificada para passar o 'config') ---

def traduzir_e_recriar_estrutura_com_llm():
    """
    Executa o pipeline completo de tradução usando o Bedrock
    e recria a estrutura original do Stereoset.
    """
    
    reconstructed_data = {}
    
    for config in CONFIGS:
        print(f"Carregando e processando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        
        new_examples_list = []
        
        for original_example in tqdm(dataset, desc=f"Traduzindo {config}"):
            
            # --- INÍCIO DA CORREÇÃO ---
            # Passa o nome da configuração (config) para a função de tradução
            # para que ela possa escolher o prompt correto.
            translated_data = translate_example_with_context(original_example, config)
            # --- FIM DA CORREÇÃO ---
            
            if translated_data is None:
                continue
                
            context_pt = sanitize_text(translated_data['contexto_traduzido'])
            alvos_pt = [sanitize_text(t) for t in translated_data['alvos_traduzidos']]
            
            # O resto da lógica de reconstrução é o mesmo
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
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({
                        "human_id": human_ids[j],
                        "label": INNER_LABEL_MAP[inner_int_labels[j]]
                    })

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": alvos_pt[i],
                    "labels": recreated_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_claude3-5-sonnet_prompt_corrigido.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido com prompts adaptativos) está pronto.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
