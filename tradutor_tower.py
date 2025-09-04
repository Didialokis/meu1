# -*- coding: utf-8 -*-

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
import re

# --- CONFIGURAÇÕES E DEFINIÇÕES (podem ficar no escopo global) ---
MODEL_NAME = "Unbabel/Tower-Plus-9B" 
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"
BATCH_SIZE = 32 

def sanitize_text(text):
    """
    Função para "limpar" o texto, removendo caracteres de controle.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def traduzir_dataset_com_vllm(llm, sampling_params):
    """
    A função de tradução agora recebe o modelo 'llm' e os parâmetros
    de amostragem como argumentos para evitar o escopo global.
    """
    datasets_dict = {}
    sentences_to_translate = []

    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    print("Iniciando a tradução em lotes com vLLM...")
    translated_sentences = []

    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        
        prompts = [
            f"Translate the following English source text to Portuguese (Brazil):\nEnglish: {sentence}\nPortuguese (Brazil): "
            for sentence in batch
        ]
        
        outputs = llm.generate(prompts, sampling_params)
        
        batch_translated_raw = [output.outputs[0].text.strip() for output in outputs]
        
        batch_sanitized = [sanitize_text(text) for text in batch_translated_raw]
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
        
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.json"
        print(f"Salvando o dataset '{config}' traduzido em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

# --- INÍCIO DA CORREÇÃO ---
# O bloco de proteção 'if __name__ == "__main__":' garante que o código
# pesado de inicialização do modelo só seja executado pelo processo principal.
if __name__ == "__main__":
    # 1. Parâmetros de amostragem
    sampling_params = SamplingParams(
      best_of=1,
      temperature=0,
      max_tokens=256,
    )

    # 2. Carregamento do modelo
    print(f"Carregando o modelo '{MODEL_NAME}' com vLLM...")
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)
    print("Modelo carregado com sucesso.")
    
    # 3. Chamada da função de tradução
    traduzir_dataset_com_vllm(llm, sampling_params)
# --- FIM DA CORREÇÃO ---
