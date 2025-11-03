# -*- coding: utf-8 -*-

import torch # Necessário para o device, mas a lógica de GPU foi removida
import boto3
import json
import re
import time
import logging
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# Configurações do AWS Bedrock
BEDROCK_REGION = 'us-east-1' # CONFIRA E MUDE SUA REGIÃO SE NECESSÁRIO
MODEL_ID = 'meta.llama3-8b-instruct-v1:0' # Modelo Llama 3 8B
MAX_RETRIES = 5 # Tentativas em caso de rate limit
RETRY_DELAY_SECONDS = 10 # Tempo de espera entre tentativas

# Configurações do Dataset (como antes)
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Mapeamentos de Label (como antes)
GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

# Configura logging para o Boto3
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)

# --- 2. INICIALIZAÇÃO DO CLIENTE BEDROCK ---

try:
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=BEDROCK_REGION
    )
    print(f"✅ Cliente Bedrock inicializado na região: {BEDROCK_REGION}")
except Exception as e:
    print(f"❌ ERRO: Não foi possível inicializar o cliente Bedrock. Verifique suas credenciais e região. {e}")
    exit()


# --- 3. FUNÇÕES AUXILIARES ---

def find_blank_index(context_str):
    """Encontra o índice da palavra 'BLANK' no contexto."""
    words = context_str.split()
    for i, word in enumerate(words):
        if "BLANK" in word:
            return i
    return None

def sanitize_text(text):
    """Limpa o texto de caracteres de controle."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def invoke_llama_translation(prompt_text):
    """
    Envia o prompt para o Llama 3 via Bedrock, com lógica de retry
    e parsing de JSON.
    """
    # Llama 3 usa um formato de prompt específico
    # Usamos <|begin_of_text|> e <|eot_id|> para delimitar
    system_prompt = "You are a precise, technical translator. Your task is to translate English text to Brazilian Portuguese, following instructions exactly. You must return ONLY a valid JSON object as your response, without any introductory text."
    
    # Formato do Llama 3 Instruct
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    body = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 2048, # Aumentar para garantir que caiba a tradução
        "temperature": 0.0, # Temperatura 0.0 para tradução determinística
        "top_p": 0.9
    })

    for attempt in range(MAX_RETRIES):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # O Llama 3 retorna a resposta em 'generation'
            generated_text = response_body['generation']
            
            # Tenta extrair o JSON do texto gerado
            try:
                # O LLM pode, às vezes, adicionar "```json ... ```"
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("Nenhum JSON encontrado na resposta do LLM.", generated_text, 0)
                
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as json_e:
                print(f"  ⚠️ Erro de parsing do JSON do LLM (Tentativa {attempt + 1}): {json_e}")
                print(f"  Resposta recebida: {generated_text}")
                continue # Tenta novamente
                
        except client.exceptions.ProvisionedThroughputExceededException as e:
            print(f"  ⏳ Rate limit atingido. Aguardando {RETRY_DELAY_SECONDS}s... (Tentativa {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            print(f"  ❌ Erro ao invocar o modelo (Tentativa {attempt + 1}): {e}")
            time.sleep(RETRY_DELAY_SECONDS)
    
    print(f"  ❌ FALHA: Não foi possível traduzir o prompt após {MAX_RETRIES} tentativas.")
    return None # Falha após todas as tentativas


# --- 4. FUNÇÃO PRINCIPAL DE TRADUÇÃO (MODIFICADA) ---

def traduzir_com_llama():
    
    # Carrega os datasets originais (como antes)
    datasets_dict = {}
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset

    print("--- Iniciando Tradução via Llama 3 no Bedrock ---")
    
    reconstructed_data = {}
    
    # --- Loop 1: INTERSENTENCE (Tradução simples) ---
    config_inter = 'intersentence'
    if config_inter in datasets_dict:
        original_dataset_inter = datasets_dict[config_inter]
        new_examples_list = []
        print(f"Reconstruindo {config_inter}...")
        
        for original_example in tqdm(original_dataset_inter, desc=f"Traduzindo {config_inter}"):
            # Pega as sentenças originais
            original_sentences = [s['sentence'] for s in original_example['sentences']]
            
            # Constrói o prompt de tradução simples
            prompt = f"""
            Por favor, traduza o seguinte JSON de inglês para português do Brasil.
            Mantenha a estrutura JSON de saída exatamente como solicitado.

            JSON de Entrada:
            {{
                "context": {json.dumps(original_example['context'])},
                "sentences": {json.dumps(original_sentences)}
            }}

            Retorne APENAS o JSON traduzido com as chaves "translated_context" e "translated_sentences".
            """
            
            # Chama o LLM
            parsed_response = invoke_llama_translation(prompt)
            if not parsed_response:
                continue # Pula este exemplo se a tradução falhar

            # Inicia a reconstrução do exemplo
            new_example = original_example.copy() # Preserva IDs, bias_type, etc.
            new_example['context'] = sanitize_text(parsed_response.get('translated_context', ''))
            
            translated_sentences = parsed_response.get('translated_sentences', [])
            
            # Recria a estrutura interna das sentenças (Lógica CRÍTICA)
            rebuilt_sentences = []
            original_sents_data = original_example['sentences']
            for i in range(len(original_sents_data['id'])):
                rebuilt_labels = []
                labels_data = original_sents_data['labels'][i]
                for j in range(len(labels_data['human_id'])):
                    rebuilt_labels.append({
                        "human_id": labels_data['human_id'][j],
                        "label": INNER_LABEL_MAP[labels_data['label'][j]]
                    })

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": sanitize_text(translated_sentences[i]) if i < len(translated_sentences) else "",
                    "labels": rebuilt_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                rebuilt_sentences.append(new_sentence_obj)
                
            new_example['sentences'] = rebuilt_sentences
            new_examples_list.append(new_example)
            
        reconstructed_data[config_inter] = new_examples_list

    # --- Loop 2: INTRASENTENCE (Tradução com regra "BLANK") ---
    config_intra = 'intrasentence'
    if config_intra in datasets_dict:
        original_dataset_intra = datasets_dict[config_intra]
        new_examples_list = []
        print(f"Reconstruindo {config_intra}...")
        
        for original_example in tqdm(original_dataset_intra, desc=f"Traduzindo {config_intra}"):
            original_context = original_example['context']
            original_sentences = [s['sentence'] for s in original_example['sentences']]
            
            # Encontra a posição do BLANK
            blank_idx = find_blank_index(original_context)
            if blank_idx is None:
                print(f"  AVISO: Não foi possível encontrar 'BLANK' no contexto: {original_context}. Pulando...")
                continue
                
            # Constrói o prompt de tradução complexo com as regras
            prompt = f"""
            Por favor, traduza o seguinte JSON de inglês para português do Brasil, seguindo regras MUITO estritas.
            
            REGRAS CRÍTICAS:
            1. A palavra "BLANK" NÃO DEVE ser traduzida. Deve permanecer "BLANK".
            2. No "context" original, "BLANK" é a palavra de índice {blank_idx} (contando a partir de 0).
            3. Na tradução do "context", "BLANK" DEVE OBRIGATORIAMENTE ser a palavra de índice {blank_idx}.
            4. As "sentences" traduzidas também DEVEM ter a palavra que substitui "BLANK" no índice {blank_idx}.
            5. Retorne APENAS o JSON traduzido com as chaves "translated_context" e "translated_sentences".

            JSON de Entrada:
            {{
                "context": {json.dumps(original_context)},
                "sentences": {json.dumps(original_sentences)}
            }}
            """
            
            # Chama o LLM
            parsed_response = invoke_llama_translation(prompt)
            if not parsed_response:
                continue

            # Inicia a reconstrução do exemplo (lógica idêntica à anterior)
            new_example = original_example.copy()
            new_example['context'] = sanitize_text(parsed_response.get('translated_context', ''))
            
            translated_sentences = parsed_response.get('translated_sentences', [])
            
            rebuilt_sentences = []
            original_sents_data = original_example['sentences']
            for i in range(len(original_sents_data['id'])):
                rebuilt_labels = []
                labels_data = original_sents_data['labels'][i]
                for j in range(len(labels_data['human_id'])):
                    rebuilt_labels.append({
                        "human_id": labels_data['human_id'][j],
                        "label": INNER_LABEL_MAP[labels_data['label'][j]]
                    })

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": sanitize_text(translated_sentences[i]) if i < len(translated_sentences) else "",
                    "labels": rebuilt_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                rebuilt_sentences.append(new_sentence_obj)
                
            new_example['sentences'] = rebuilt_sentences
            new_examples_list.append(new_example)
            
        reconstructed_data[config_intra] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_llama3_formato_original.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída (traduzido via Llama 3) é 100% compatível com o dataloader.py.")


if __name__ == "__main__":
    traduzir_com_llama()
