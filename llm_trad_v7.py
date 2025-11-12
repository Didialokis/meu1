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


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (Modificada para Texto Único) ---

def translate_single_text(text_en):
    """
    Chama o Llama 3 no Bedrock para traduzir um ÚNICO pedaço de texto.
    """
    
    # Prompt de sistema simples e direto
    system_prompt = (
        "Você é um tradutor especialista de Inglês para Português do Brasil. "
        "Traduza o texto fornecido de forma precisa e o mais literal possível, "
        "mantendo o significado e a estrutura. "
        "Responda *apenas* com o texto traduzido, sem qualquer preâmbulo, explicação ou aspas."
    )
    
    user_prompt = f"Traduza o seguinte texto para o Português do Brasil: \"{text_en}\""
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 512, # Pode ser menor para textos únicos
        "temperature": 0.1,
        "top_p": 0.9
    })

    retries = 3
    delay = 5
    for i in range(retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Pega a tradução e limpa espaços/aspas extras
            translated_text = response_body['generation'].strip().strip('"')
            
            if translated_text:
                return translated_text
            else:
                raise Exception("Resposta de tradução vazia.")

        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"Throttling em '{text_en[:20]}'... retentativa em {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return None
        except Exception as e:
            logging.error(f"Erro ao processar texto '{text_en[:20]}...': {e}")
            logging.error(f"Resposta inválida: {response_body['generation']}")
            return None
    
    logging.error(f"Excedeu retentativas para o texto: '{text_en[:20]}...'")
    return None


# --- 3. FUNÇÃO AUXILIAR (COMO ANTES) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    if text is None:
        return None
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (MODIFICADA) ---

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
        
        # --- ETAPA DE TRADUÇÃO E RECONSTRUÇÃO (LÓGICA CORRIGIDA) ---
        for original_example in tqdm(dataset, desc=f"Traduzindo {config}"):
            
            # --- 1. TRADUÇÃO INDEPENDENTE ---
            # Cada chamada é separada e sem contaminação.
            
            context_pt = sanitize_text(translate_single_text(original_example['context']))
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])
            
            # Traduz cada sentença alvo independentemente
            alvos_pt = []
            for i in range(num_sentences):
                texto_alvo_en = original_sents_data['sentence'][i]
                texto_alvo_pt = sanitize_text(translate_single_text(texto_alvo_en))
                alvos_pt.append(texto_alvo_pt)

            # --- 2. VERIFICAÇÃO DE FALHA ---
            # Pula o exemplo se alguma tradução crítica falhar
            if context_pt is None or any(t is None for t in alvos_pt):
                logging.warning(f"Pulando exemplo {original_example['id']} devido à falha na tradução.")
                continue
                
            # --- 3. RECONSTRUÇÃO (COMO ANTES) ---
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": context_pt,
                "sentences": []
            }

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
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_llama3_isolado.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido isoladamente) é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
