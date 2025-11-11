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


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (Modificada para tradução simples) ---

def translate_text(text_en):
    """
    Chama o Llama 3 no Bedrock para traduzir um ÚNICO trecho de texto.
    """
    
    # Prompt de sistema simples para tradução literal
    system_prompt = (
        "Você é um tradutor especialista de Inglês para Português do Brasil. "
        "Sua tarefa é traduzir o texto fornecido de forma literal, sem adicionar "
        "comentários ou contexto adicional. Responda *apenas* com a tradução."
    )
    
    user_prompt = f"""Traduza o seguinte texto:
"{text_en}"
"""
    
    # Formato do prompt Llama 3 Instruct
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" # O modelo responderá apenas com o texto traduzido

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.1,
        "top_p": 0.9
    })

    retries = 3
    delay = 5
    for i in range(retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Pega a tradução e remove aspas extras que o modelo pode adicionar
            translated_text = response_body['generation'].strip().strip('"')
            
            if translated_text:
                return translated_text
            else:
                raise Exception("Tradução retornou vazia.")

        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"Throttling... retentativa em {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return None # Falha na tradução deste texto
        except Exception as e:
            logging.error(f"Erro ao processar texto '{text_en[:50]}...': {e}")
            logging.error(f"Resposta inválida: {response_body.get('generation', 'N/A')}")
            return None
    
    logging.error(f"Excedeu retentativas para o texto '{text_en[:50]}...'.")
    return None


# --- 3. FUNÇÃO AUXILIAR (COMO ANTES) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (LÓGICA CORRIGIDA) ---

def traduzir_e_recriar_estrutura_com_llm():
    """
    Executa o pipeline completo de tradução usando o Bedrock
    com chamadas ISOLADAS para preservar a integridade do teste.
    """
    
    reconstructed_data = {}
    
    for config in CONFIGS:
        print(f"Carregando e processando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        
        new_examples_list = []
        
        for original_example in tqdm(dataset, desc=f"Traduzindo {config}"):
            
            # --- INÍCIO DA LÓGICA CORRIGIDA ---
            # 1. Traduz o contexto de forma isolada
            context_pt = translate_text(original_example['context'])
            if context_pt is None:
                continue # Pula o exemplo se o contexto falhar
            
            # 2. Traduz cada alvo de forma isolada
            original_sents_data = original_example['sentences']
            alvos_en = original_sents_data['sentence']
            alvos_pt = []
            
            translation_failed = False
            for alvo_en in alvos_en:
                alvo_pt = translate_text(alvo_en)
                if alvo_pt is None:
                    translation_failed = True
                    break # Se um alvo falhar, o exemplo todo é inválido
                alvos_pt.append(alvo_pt)
            
            if translation_failed:
                continue # Pula o exemplo
            # --- FIM DA LÓGICA CORRIGIDA ---

            # Monta a nova estrutura do exemplo
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": sanitize_text(context_pt),
                "sentences": []
            }
            
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

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": sanitize_text(alvos_pt[i]), # Usa o texto da lista de alvos traduzidos
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
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_llama3_isolado.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido isoladamente) é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
