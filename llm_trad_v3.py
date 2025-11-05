# -*- coding: utf-8 -*-

import boto3
import botocore.exceptions
import json
import re
import time
import logging
# from datasets import load_dataset # Removido - não é mais necessário
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# ATENÇÃO: Nome do seu arquivo JSON local
LOCAL_JSON_FILE = 'dev.json' 

# Configurações do Amazon Bedrock
MODEL_ID = 'meta.llama3-8b-instruct-v1:0'
REGION_NAME = 'us-east-1' # Ajuste para sua região, se necessário
client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

# Configs de navegação do JSON local (substituindo configs do Hugging Face)
CONFIGS = ['intersentence', 'intrasentence'] 

# Mapeamentos de labels não são mais necessários, pois o dev.json já tem labels de texto.
# GOLD_LABEL_MAP = ... (Removido)
# INNER_LABEL_MAP = ... (Removido)

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 2. FUNÇÃO DE CHAMADA DO BEDROCK (PROMPT ATUALIZADO) ---

def invoke_llama_translation(text_to_translate):
    """
    Chama o modelo Llama 3 8B no Bedrock para traduzir um único texto.
    Inclui lógica de retentativa para throttling.
    """
    
    # --- PROMPT ATUALIZADO ---
    # Adicionada a instrução para ignorar (preservar) o placeholder "55555"
    system_prompt = (
        "Você é um tradutor especialista. Traduza o texto de Inglês para Português do Brasil. "
        "Responda *apenas* com o texto traduzido, sem frases introdutórias, explicações, "
        "aspas ou qualquer outro texto. "
        "IMPORTANTE: Se você encontrar o placeholder '55555' no texto, mantenha-o exatamente como '55555' na tradução."
    )
    # --- FIM DA ATUALIZAÇÃO ---
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
Texto em Inglês: "{text_to_translate}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
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
            translated_text = response_body['generation'].strip().strip('"')
            return translated_text
        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"ThrottlingException... retentativa em {delay}s. Texto: {text_to_translate[:50]}...")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return f"ERRO_DE_TRADUCAO: {e}"
        except Exception as e:
            logging.error(f"Erro ao invocar o modelo: {e}")
            return f"ERRO_DE_TRADUCAO: {e}"
    
    return "ERRO_DE_TRADUCAO: Excedeu retentativas de throttling."


# --- 3. FUNÇÃO AUXILIAR (COMO ANTES) ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (MODIFICADA) ---

def traduzir_e_recriar_estrutura_com_llm():
    """
    Executa o pipeline completo de tradução usando o Bedrock
    a partir de um arquivo dev.json local.
    """
    
    # --- ETAPA DE EXTRAÇÃO (MODIFICADA PARA LER dev.json LOCAL) ---
    print(f"Carregando o arquivo local '{LOCAL_JSON_FILE}'...")
    try:
        with open(LOCAL_JSON_FILE, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            # O dataloader original espera esta estrutura
            original_data = full_data['data']
    except FileNotFoundError:
        logging.error(f"ERRO: Arquivo '{LOCAL_JSON_FILE}' não encontrado.")
        return
    except KeyError:
        logging.error(f"ERRO: O arquivo '{LOCAL_JSON_FILE}' não contém a chave 'data'. Verifique o formato.")
        return

    # A "datasets_dict" agora armazena os dados brutos do JSON
    datasets_dict = {} 
    sentences_to_translate = []
    
    for config in CONFIGS:
        if config not in original_data:
            logging.warning(f"Aviso: Chave '{config}' não encontrada no JSON.")
            datasets_dict[config] = [] # Garante que a chave exista
            continue
        
        examples = original_data[config]
        datasets_dict[config] = examples # Armazena a lista de exemplos
        
        # Extrai sentenças para a tradução em lote
        for example in examples:
            sentences_to_translate.append(example['context'])
            for sent_obj in example['sentences']:
                sentences_to_translate.append(sent_obj['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- ETAPA DE TRADUÇÃO (Via Bedrock, como antes) ---
    print(f"Iniciando a tradução via Bedrock (Modelo: {MODEL_ID})...")
    translated_sentences = []

    for sentence_to_translate in tqdm(sentences_to_translate, desc="Traduzindo sentenças"):
        translated_text_raw = invoke_llama_translation(sentence_to_translate)
        batch_sanitized = [sanitize_text(translated_text_raw)]
        translated_sentences.extend(batch_sanitized)

    print("Tradução finalizada.")

    # --- ETAPA DE RECONSTRUÇÃO MANUAL (LÓGICA AJUSTADA) ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    reconstructed_data = {}
    for config in CONFIGS:
        original_dataset = datasets_dict[config] # Agora é a lista do JSON
        new_examples_list = []
        for original_example in tqdm(original_dataset, desc=f"Reconstruindo {config}"):
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": next(translated_iter), # Pega o contexto traduzido
                "sentences": []
            }
            
            # Itera sobre a lista de dicionários de sentenças
            for original_sentence_obj in original_example['sentences']:
                new_sentence_obj = {
                    "id": original_sentence_obj['id'],
                    "sentence": next(translated_iter), # Pega a sentença traduzida
                    "labels": original_sentence_obj['labels'], # Preserva os labels originais
                    "gold_label": original_sentence_obj['gold_label'] # Preserva o gold_label original (string)
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO (COMO ANTES) ---
    final_output_structure = {
        "version": "1.1", # Mantém a compatibilidade com o dataloader
        "data": reconstructed_data
    }
    
    output_path = f"dev_pt_llama3_formato_original.json" # Nome de arquivo atualizado
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido via LLM) é compatível com o dataloader.py.")


# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_e_recriar_estrutura_com_llm()
