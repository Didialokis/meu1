Script de Tradução Corrigido
Python

# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

MODEL_NAME = "facebook/nllb-200-1.3B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Códigos de idioma para o padrão NLLB (Flores-200)
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"

BATCH_SIZE = 8
PLACEHOLDER = "__BLANK_PLACEHOLDER__"

# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """
    Limpa o texto, removendo caracteres de controle que podem quebrar o JSON.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_dataset_completo():
    """
    Executa o pipeline completo de tradução e salva um único arquivo de saída.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    datasets_dict = {}
    sentences_to_translate = []

    for config in CONFIGS:
        print(f"Baixando e carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        batch_com_placeholder = [s.replace("BLANK", PLACEHOLDER) for s in batch]
        inputs = tokenizer(batch_com_placeholder, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )
        batch_translated_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        batch_translated_final = [s.replace(PLACEHOLDER, "BLANK") for s in batch_translated_raw]
        batch_sanitized = [sanitize_text(text) for text in batch_translated_final]
        translated_sentences.extend(batch_sanitized)

    print("Tradução finalizada.")

    # --- 4. RECONSTRUÇÃO DO DATASET ---
    print("Reconstruindo o dataset...")
    translated_iter = iter(translated_sentences)
    
    reconstructed_data = {}
    for config in CONFIGS:
        dataset_original = datasets_dict[config]
        def replace_sentences(example):
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example
        translated_dataset = dataset_original.map(replace_sentences)
        reconstructed_data[config] = list(translated_dataset)

    # --- 5. SALVANDO O RESULTADO NO FORMATO OFICIAL ---
    
    # --- INÍCIO DA CORREÇÃO ---
    # Criamos um dicionário final que imita a estrutura do arquivo original,
    # com as chaves "version" e "data".
    final_output_structure = {
        "version": "1.1", # Adiciona a chave de versão esperada
        "data": reconstructed_data # Aninha os dados traduzidos sob a chave "data"
    }
    # --- FIM DA CORREÇÃO ---
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_nllb_completo.json"
    print(f"Salvando o dataset combinado em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Salva a nova estrutura completa no arquivo JSON
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")


# --- 6. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_dataset_completo()
Seu Fluxo de Trabalho Corrigido
Agora, seu fluxo de trabalho deve ser o seguinte:

Execute o Script de Tradução (acima): Use este script corrigido para gerar o arquivo stereoset_validation_pt_nllb_completo.json. Ele agora terá o formato perfeito.

Use o dataloader.py: Após gerar o arquivo no formato correto, o script dataloader.py não dará mais o erro KeyError: 'version', pois a chave agora existe.

Execute o Script de Predição: O seu script de predição também precisa de um pequeno ajuste para ler a nova estrutura aninhada.

Ajuste necessário no seu script de predição (predicao.py):

Python

# No script de predição, mude como os dados são lidos:

try:
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        # Carrega o JSON inteiro
        full_data = json.load(f)
        # ACESSA OS DADOS DENTRO DA CHAVE "data"
        gold_data = full_data['data'] 
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{GOLD_FILE}' não encontrado. Verifique o nome do arquivo.")
    return
# ... o resto do script continua igual ...
