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
PLACEHOLDER = "__BLANK_PLACEHOLDER__" # Placeholder para proteger o token "BLANK"

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

    # Carrega os dados e extrai todas as sentenças
    for config in CONFIGS:
        print(f"Baixando e carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # Executa a tradução em lotes
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

    # --- 4. RECONSTRUÇÃO DO DATASET NA ESTRUTURA ORIGINAL ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    # Dicionário final que irá espelhar a estrutura do JSON original
    final_output_structure = {}

    for config in CONFIGS:
        dataset_original = datasets_dict[config]

        def replace_sentences(example):
            # Esta função garante que apenas o texto seja alterado,
            # preservando IDs, labels, etc.
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example

        # Aplica a tradução e armazena o resultado no dicionário final
        translated_dataset = dataset_original.map(replace_sentences)
        final_output_structure[config] = list(translated_dataset)

    # --- 5. SALVANDO O RESULTADO EM UM ÚNICO ARQUIVO ---
    output_path = f"stereoset_{DATASET_SPLIT}_pt_nllb_completo.json"
    print(f"Salvando o dataset combinado em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Usa json.dump para salvar o dicionário completo em um único arquivo
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

# --- 6. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_dataset_completo()
