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
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Mapeamentos para recriar o formato original
GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

# --- 2. FUNÇÕES ESSENCIAIS (Mínimo Necessário) ---

def invoke_llama_translation(text_to_translate):
    """
    Chama o Llama 3 no Bedrock para traduzir um texto.
    """
    # Prompt de sistema focado em tradução pura
    system_prompt = (
        "Você é um tradutor especialista. Traduza o texto de Inglês para Português do Brasil. "
        "Responda *apenas* com o texto traduzido, sem explicações ou introduções."
    )
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{text_to_translate}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.1
    })

    retries = 3
    delay = 5
    for i in range(retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            # Limpa a saída para pegar apenas a tradução
            return response_body['generation'].strip().strip('"')
        
        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"Throttling... retentativa em {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return f"ERRO_DE_TRADUCAO"
    
    return "ERRO_DE_TRADUCAO: Excedeu retentativas."

def sanitize_text(text):
    """
    Necessário para remover caracteres inválidos que o LLM pode gerar,
    garantindo que o JSON final seja válido.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

# --- 3. FUNÇÃO PRINCIPAL (Mínima e Combinada) ---

def traduzir_e_reconstruir():
    """
    Carrega o dataset, traduz e reconstrói a estrutura em uma única passagem.
    """
    
    reconstructed_data = {}
    for config in CONFIGS:
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        new_examples_list = []
        
        print(f"Processando e traduzindo a configuração: {config}...")
        for original_example in tqdm(dataset):
            
            # Traduz o contexto
            translated_context = invoke_llama_translation(original_example['context'])
            
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": sanitize_text(translated_context),
                "sentences": []
            }
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # Traduz a sentença
                original_sentence = original_sents_data['sentence'][i]
                translated_sentence = invoke_llama_translation(original_sentence)

                # Recria a estrutura interna de "labels" (Necessário para o dataloader.py)
                recreated_labels = []
                labels_data = original_sents_data['labels'][i]
                human_ids = labels_data['human_id']
                inner_labels = labels_data['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({
                        "human_id": human_ids[j],
                        "label": INNER_LABEL_MAP[inner_labels[j]]
                    })

                # Monta o objeto da sentença traduzida
                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": sanitize_text(translated_sentence),
                    "labels": recreated_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- 4. ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1", # Necessário para o dataloader.py
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_llama3_formato_original.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_reconstruir()
