# -*- coding: utf-8 -*-

# É recomendado ter as versões mais recentes das bibliotecas.
# pip install torch transformers datasets accelerate
# Para recursos muito recentes, pode ser necessário instalar do source:
# pip install git+https://github.com/huggingface/transformers.git

import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import re
import json

# --- 1. CONFIGURAÇÕES ---

# Modelo de geração de texto para tradução
MODEL_ID = "Unbabel/Tower-Plus-2B"

# Informações do dataset Stereoset
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Idioma de destino
TARGET_LANGUAGE = "Portuguese (Brazil)"

# Tamanho do lote. Para um modelo grande (2B), um valor baixo é crucial
# para evitar erros de falta de memória (Out of Memory).
# Ajuste conforme a VRAM da sua GPU. Comece com 4 ou 8.
BATCH_SIZE = 8

# Template do prompt, adaptado para português do Brasil
PROMPT_TEMPLATE = f"Translate the following English source text to {TARGET_LANGUAGE}:\nEnglish: {{text}}\n{TARGET_LANGUAGE}: "


# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """Limpa o texto de caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def translate_batch(pipe, batch_texts):
    """
    Formata um lote de textos no template de chat e os traduz usando o pipeline.
    """
    # 1. Cria a lista de prompts para cada texto no lote
    prompts = [PROMPT_TEMPLATE.format(text=text) for text in batch_texts]

    # 2. Formata os prompts no formato de 'messages' que o pipeline espera
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    
    # 3. Executa o pipeline. O batch_size aqui ajuda o pipeline a otimizar.
    outputs = pipe(messages_batch, max_new_tokens=128, do_sample=False, batch_size=len(batch_texts))
    
    translations = []
    for output in outputs:
        # O resultado inclui o prompt. Precisamos extrair apenas a tradução.
        generated_text = output[0]['generated_text']
        
        # Encontra o final do prompt e pega o que vem depois
        try:
            translation = generated_text.split(f"{TARGET_LANGUAGE}: ")[-1].strip()
            translations.append(translation)
        except IndexError:
            # Caso o parsing falhe, adiciona uma string vazia para não quebrar o pipeline
            translations.append("")
            
    return translations


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def translate_stereoset_with_tower():
    """
    Função principal que executa todo o pipeline de tradução.
    """
    print(f"--- INICIANDO TRADUÇÃO COM O MODELO: {MODEL_ID} ---")
    
    # Carrega o pipeline. `device_map="auto"` distribui o modelo pelas GPUs.
    # `torch_dtype=torch.bfloat16` acelera a inferência e economiza memória.
    print("Carregando o pipeline de geração de texto...")
    try:
        pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        print(f"ERRO ao carregar o pipeline. Verifique se a biblioteca 'accelerate' está instalada.")
        print(f"Detalhe do erro: {e}")
        return
        
    print("Pipeline carregado com sucesso.")

    # Carrega e extrai todas as sentenças do dataset
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])

    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # Traduz em lotes
    translated_sentences = []
    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo lotes"):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        
        # Traduz o lote usando a nova função
        translated_batch = translate_batch(pipe, batch)
        
        # Sanitiza e armazena os resultados
        sanitized_batch = [sanitize_text(text) for text in translated_batch]
        translated_sentences.extend(sanitized_batch)

    print("Tradução finalizada.")

    # Reconstrói os datasets com os textos traduzidos
    print("Reconstruindo os datasets...")
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
        
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.json"
        print(f"Salvando dataset '{config}' em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

# --- 4. EXECUÇÃO ---

if __name__ == "__main__":
    translate_stereoset_with_tower()
