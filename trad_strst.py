# -*- coding: utf-8 -*-

pip install torch transformers sentencepiece datasets
python traduzir_online.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset # Importação da nova biblioteca

# --- CONFIGURAÇÕES ---
# Modelo de tradução do Hugging Face
MODEL_NAME = "facebook/m2m100_418M"

# Identificador do dataset no Hugging Face Hub
DATASET_NAME = "McGill-NLP/stereoset"
DATASET_SPLIT = "validation" # O split de desenvolvimento é chamado de 'validation'

# Idiomas (ISO 639-1 codes)
SOURCE_LANG = "en"  # Inglês
TARGET_LANG = "pt"  # Português

# Caminho do arquivo de saída
OUTPUT_JSON_PATH = "stereoset_validation_pt.json"

# Parâmetros de processamento
BATCH_SIZE = 16  # Ajuste conforme a memória da sua GPU

def traduzir_dataset_huggingface():
    """
    Função principal para carregar o dataset Stereoset do Hugging Face,
    traduzi-lo e salvá-lo como um novo arquivo JSON.
    """
    # --- 1. PREPARAÇÃO DO MODELO ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- 2. CARREGAMENTO E EXTRAÇÃO DAS SENTENÇAS ---

    print(f"Baixando e carregando o dataset '{DATASET_NAME}' do Hugging Face Hub...")
    dataset = load_dataset(DATASET_NAME, "en", split=DATASET_SPLIT)
    print("Dataset carregado.")

    # Coleta todas as strings de texto em uma lista para traduzir em lote
    sentences_to_translate = []
    for example in dataset:
        # Adiciona o contexto (seja uma palavra ou uma sentença)
        sentences_to_translate.append(example['context'])
        # Adiciona todas as sentenças alvo associadas
        sentences_to_translate.extend(example['sentences']['sentence'])

    if not sentences_to_translate:
        print("Nenhuma sentença encontrada para traduzir.")
        return

    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- 3. TRADUÇÃO EM LOTE ---
    # (Esta seção permanece a mesma, pois é a mais eficiente)

    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    forced_bos_token_id = tokenizer.get_lang_id(TARGET_LANG)

    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=128)
        batch_translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_sentences.extend(batch_translated)
        print(f"  Lote {i//BATCH_SIZE + 1} de {len(sentences_to_translate)//BATCH_SIZE + 1} concluído...")

    print("Tradução finalizada.")

    # --- 4. RECONSTRUÇÃO DO DATASET COM O MÉTODO .MAP() ---

    print("Reconstruindo o dataset com as sentenças traduzidas...")
    
    # Cria um iterador para fornecer as sentenças traduzidas em ordem
    translated_iter = iter(translated_sentences)

    def replace_sentences(example):
        """
        Função auxiliar que substitui o texto em um exemplo do dataset.
        """
        # Substitui o contexto
        example['context'] = next(translated_iter)
        
        # Substitui as sentenças alvo
        num_target_sentences = len(example['sentences']['sentence'])
        translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
        example['sentences']['sentence'] = translated_target_sentences
        
        return example

    # O método .map() aplica a função 'replace_sentences' a cada exemplo no dataset
    translated_dataset = dataset.map(replace_sentences)

    # --- 5. SALVANDO O RESULTADO FINAL ---

    print(f"Salvando o dataset traduzido em: {OUTPUT_JSON_PATH}")
    # O objeto 'Dataset' tem um método conveniente para salvar em JSON
    translated_dataset.to_json(OUTPUT_JSON_PATH, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

if __name__ == "__main__":
    traduzir_dataset_huggingface()
