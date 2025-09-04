# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import re

# --- 1. CONFIGURAÇÕES ---

MODEL_NAME = "facebook/nllb-200-1.3B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Códigos de idioma para o padrão NLLB (Flores-200)
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"

BATCH_SIZE = 8
PLACEHOLDER = "__BLANK_PLACEHOLDER__" # Define o placeholder como uma constante

# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """
    Função para "limpar" o texto, removendo caracteres de controle
    que podem quebrar o formato JSON. ESTA FUNÇÃO ESTAVA FALTANDO.
    """
    # Remove caracteres de controle C0 e C1, exceto os comuns como tab e newline
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_dataset_completo():
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
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para todas as configurações.")

    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    
    # Método correto para obter o ID do idioma de destino para o NLLB
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        
        # Lógica para proteger o token "BLANK" antes de traduzir
        batch_com_placeholder = [s.replace("BLANK", PLACEHOLDER) for s in batch]
        
        inputs = tokenizer(batch_com_placeholder, return_tensors="pt", padding=True, truncation=True).to(device)
        
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )
        
        batch_translated_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Restaura o token "BLANK" após a tradução
        batch_translated_final = [s.replace(PLACEHOLDER, "BLANK") for s in batch_translated_raw]
        
        # Chama a função sanitize_text para limpar a saída
        batch_sanitized = [sanitize_text(text) for text in batch_translated_final]
        translated_sentences.extend(batch_sanitized)
        
        print(f"  Lote {i//BATCH_SIZE + 1} de {len(sentences_to_translate)//BATCH_SIZE + 1} concluído...")

    print("Tradução finalizada.")

    print("Reconstruindo os datasets com as sentenças traduzidas...")
    translated_iter = iter(translated_sentences)

    for config in CONFIGS:
        dataset_original = datasets_dict[config]

        def replace_sentences(example):
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example

        translated_dataset = dataset_original.map(replace_sentences)
        
        # Salva com um nome de arquivo claro para indicar que é a versão final
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt_nllb_final.json"
        print(f"Salvando o dataset '{config}' traduzido em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

# --- 4. EXECUÇÃO ---
if __name__ == "__main__":
    traduzir_dataset_completo()
    
    /////////////////////////////////////////





# --- CONFIGURAÇÕES ---

# Modelo foi trocado para o NLLB-200 com 1.3 bilhão de parâmetros
MODEL_NAME = "facebook/nllb-200-1.3B" 
# Para uma versão mais rápida, use: "facebook/nllb-200-distilled-600M"

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# ATENÇÃO: Os códigos de idioma foram atualizados para o padrão do NLLB
SOURCE_LANG = "eng_Latn"  # Código para Inglês no NLLB
TARGET_LANG = "por_Latn"  # Código para Português no NLLB

BATCH_SIZE = 8 # É prudente diminuir o batch size inicial para um modelo maior

# ... o resto do script permanece o mesmo ...

def traduzir_dataset_huggingface_corrigido():
    """
    Função corrigida para carregar as configurações corretas do dataset,
    traduzir e salvar os resultados.
    """
    # --- 1. PREPARAÇÃO DO MODELO ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- 2. CARREGAMENTO E EXTRAÇÃO DAS SENTENÇAS (DE AMBAS AS CONFIGS) ---
    datasets = {}
    sentences_to_translate = []

    for config in CONFIGS:
        print(f"Baixando e carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets[config] = dataset
        
        # Extrai sentenças de cada dataset e adiciona à lista geral
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas de todas as configurações.")

    # --- 3. TRADUÇÃO EM LOTE ---
    # A lógica de tradução é a mesma, pois processamos tudo de uma vez
    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    # Linhas Corretas
    tokenizer.src_lang = SOURCE_LANG # Define o idioma de origem
    forced_bos_token_id = tokenizer.lang_code_to_id[TARGET_LANG] # Pega o ID do idioma de destino

    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=128)
        batch_translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_sentences.extend(batch_translated)
        print(f"  Lote {i//BATCH_SIZE + 1} de {len(sentences_to_translate)//BATCH_SIZE + 1} concluído...")

    print("Tradução finalizada.")

    # --- 4. RECONSTRUÇÃO DE CADA DATASET ---
    print("Reconstruindo os datasets com as sentenças traduzidas...")
    translated_iter = iter(translated_sentences)

    for config in CONFIGS:
        dataset_original = datasets[config]

        def replace_sentences(example):
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example

        # Aplica o mapeamento para criar o dataset traduzido
        translated_dataset = dataset_original.map(replace_sentences)
        
        # --- 5. SALVANDO O RESULTADO ---
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt.json"
        print(f"Salvando o dataset '{config}' traduzido em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

if __name__ == "__main__":
    traduzir_dataset_huggingface_corrigido()
