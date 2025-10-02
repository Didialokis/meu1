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
    Executa o pipeline completo de tradução, gerando um único arquivo de saída
    em formato JSON válido.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device.upper()}")

    print(f"💾 Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("✅ Modelo carregado com sucesso.")

    datasets_dict = {}
    sentences_to_translate = []

    # Carrega os dados e extrai todas as sentenças
    for config in CONFIGS:
        print(f"💿 Baixando e carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"📊 Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # Executa a tradução em lotes
    print("⏳ Iniciando a tradução em lotes...")
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

    print("✅ Tradução finalizada.")

    # --- MUDANÇA: RECONSTRUÇÃO E SALVAMENTO UNIFICADOS ---

    print("🏗️ Reconstruindo o dataset na estrutura final...")
    translated_iter = iter(translated_sentences)
    
    # Dicionário que irá conter os dados de 'intrasentence' e 'intersentence'
    reconstructed_data = {}

    for config in CONFIGS:
        dataset_original = datasets_dict[config]

        def replace_sentences(example):
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example

        # Aplica a tradução e converte o resultado para uma lista
        translated_dataset = dataset_original.map(replace_sentences)
        reconstructed_data[config] = list(translated_dataset)

    # Cria a estrutura final que imita o arquivo original, com "version" e "data"
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }

    # Salva tudo em um único arquivo JSON válido usando json.dump
    output_path = f"stereoset_{DATASET_SPLIT}_pt_nllb_completo.json"
    print(f"💾 Salvando o dataset combinado em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # json.dump garante um único arquivo JSON bem formatado
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n🎉 Sucesso! Processo concluído.")

# --- 4. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_dataset_completo()
