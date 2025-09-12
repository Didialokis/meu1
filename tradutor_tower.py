# -*- coding: utf-8 -*-

import torch
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- 1. CONFIGURAÇÕES ---

MODEL_ID = "Unbabel/Tower-Plus-2B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# --- 2. FUNÇÕES AUXILIARES ---

def create_translation_prompt(english_text):
    """
    Formata uma sentença em inglês no template que o modelo espera.
    """
    return (
        "Translate the following English text to Brazilian Portuguese.\n"
        f"English source: \"{english_text}\"\n"
        "Brazilian Portuguese translation: "
    )

def clean_translation(text):
    """
    Função de limpeza robusta para remover ruídos e padrões indesejados
    da saída do modelo de tradução.
    """
    if not isinstance(text, str):
        return ""

    # Remove aspas extras no início e no fim do texto
    text = text.strip().strip('"')

    # Remove padrões como "0:12", "1. I", "10.", etc., no início da string.
    # Esta expressão regular procura por:
    # ^                  - início da string
    # (["\s]*)?         - aspas e/ou espaços opcionais
    # (\d+[:.]\s*|\d+\.\s*I\s*)? - padrões como "0:", "12.", "1. I "
    # ([\s"]*)?         - mais espaços e/ou aspas opcionais
    pattern = r'^([\'"\s]*)?(\d+[:.]\s*|\d+\.\s*I\s*)?([\'"\s]*)?'
    cleaned_text = re.sub(pattern, '', text, count=1)

    return cleaned_text.strip()


def save_translated_dataset(config_name, dataset, translation_map, split_name):
    """
    Função dedicada para reconstruir e salvar um dataset traduzido.
    """
    output_filename = f"stereoset_{config_name}_{split_name}_pt_tower_vllm.jsonl"
    print(f"Reconstruindo e salvando o dataset '{config_name}' em: {output_filename}")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc=f"Salvando '{config_name}'"):
            clean_example = {
                "id": example["id"],
                "target": example["target"],
                "bias_type": example["bias_type"],
                "context": translation_map.get(example["context"], example["context"]),
                "sentences": {
                    "sentence": [
                        translation_map.get(sent, sent)
                        for sent in example["sentences"]["sentence"]
                    ],
                    "gold_label": list(example["sentences"]["gold_label"]),
                },
            }
            f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')

# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO COM VLLM ---

def translate_stereoset_with_vllm():
    print(f"--- INICIANDO TRADUÇÃO COM O MODELO: {MODEL_ID} (via vLLM) ---")
    
    # Aumentar um pouco a temperatura pode ajudar a evitar repetições de padrões,
    # mas para tradução, 0 ou um valor muito baixo é geralmente melhor.
    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("Nenhuma GPU encontrada! vLLM requer pelo menos uma GPU CUDA.")
    
    print(f"Detectadas {num_gpus} GPUs. O modelo será distribuído entre elas.")

    print("Carregando o modelo com vLLM...")
    llm = LLM(model=MODEL_ID, tensor_parallel_size=num_gpus)
    print("Modelo carregado com sucesso.")

    original_datasets = {}
    all_sentences_to_translate = set()

    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        original_datasets[config] = dataset
        for example in dataset:
            if example['context']: all_sentences_to_translate.add(example['context'])
            all_sentences_to_translate.update([s for s in example['sentences']['sentence'] if s])
    
    unique_english_sentences = sorted(list(all_sentences_to_translate))
    print(f"Total de {len(unique_english_sentences)} sentenças únicas para traduzir.")

    all_prompts = [create_translation_prompt(text) for text in unique_english_sentences]
    
    print("Iniciando a tradução com vLLM...")
    outputs = llm.generate(all_prompts, sampling_params)
    
    # --- INÍCIO DA CORREÇÃO ---
    # Aplicamos a função de limpeza a cada saída do modelo.
    all_translations = [clean_translation(output.outputs[0].text) for output in outputs]
    # --- FIM DA CORREÇÃO ---

    translation_map = dict(zip(unique_english_sentences, all_translations))
    print("Tradução e limpeza de todas as sentenças concluída.")

    # --- 4. RECONSTRUÇÃO E SALVAMENTO ---
    print("\n--- Reconstruindo e salvando os datasets traduzidos ---")
    
    save_translated_dataset(
        config_name='intersentence',
        dataset=original_datasets['intersentence'],
        translation_map=translation_map,
        split_name=DATASET_SPLIT
    )
    
    save_translated_dataset(
        config_name='intrasentence',
        dataset=original_datasets['intrasentence'],
        translation_map=translation_map,
        split_name=DATASET_SPLIT
    )

    print("\n--- PROCESSO DE TRADUÇÃO CONCLUÍDO COM SUCESSO! ---")

# --- 5. EXECUÇÃO ---
if __name__ == "__main__":
    translate_stereoset_with_vllm()
