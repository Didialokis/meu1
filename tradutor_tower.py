# -*- coding: utf-8 -*-

import torch
from transformers import pipeline
from datasets import load_dataset
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

MODEL_ID = "Unbabel/Tower-Plus-2B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"
BATCH_SIZE = 8

# --- 2. FUNÇÕES AUXILIARES ---

def create_translation_prompt(english_text):
    """
    Formata uma sentença em inglês no template de chat que o modelo espera.
    """
    prompt = (
        "Translate the following English text to Brazilian Portuguese.\n"
        f"English source: \"{english_text}\"\n"
        "Brazilian Portuguese translation: "
    )
    return [{"role": "user", "content": prompt}]

def parse_assistant_response(assistant_response):
    """
    Extrai a tradução da resposta do assistente de forma robusta.
    """
    # A resposta do assistente já é o texto traduzido.
    # Esta função pode ser usada para limpezas futuras, se necessário.
    # Por exemplo, remover aspas que o modelo possa adicionar no início/fim.
    return assistant_response.strip().strip('"')


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def translate_stereoset_with_tower():
    print(f"--- INICIANDO TRADUÇÃO COM O MODELO: {MODEL_ID} ---")
    
    print("Carregando o pipeline...")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
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

    prompts_for_model = [create_translation_prompt(text) for text in unique_english_sentences]
    
    all_translations = []
    print(f"Iniciando a tradução em lotes de tamanho {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(prompts_for_model), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch_prompts_model = prompts_for_model[i:i + BATCH_SIZE]
        
        outputs = pipe(batch_prompts_model, max_new_tokens=256, do_sample=False)
        
        for output in outputs:
            # --- INÍCIO DA CORREÇÃO ---
            # A saída é uma conversa (lista de dicionários).
            conversation = output[0]['generated_text']
            
            # A tradução é o conteúdo da última mensagem na conversa (a resposta do assistente).
            # Adicionamos uma verificação para garantir que a conversa não está vazia.
            if isinstance(conversation, list) and len(conversation) > 0:
                assistant_response = conversation[-1]['content']
            else:
                # Fallback caso a saída não seja no formato esperado
                assistant_response = ""

            parsed_translation = parse_assistant_response(assistant_response)
            # --- FIM DA CORREÇÃO ---
            all_translations.append(parsed_translation)

    translation_map = dict(zip(unique_english_sentences, all_translations))
    print("Tradução de todas as sentenças concluída.")

    for config, dataset in original_datasets.items():
        output_filename = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.jsonl"
        print(f"Reconstruindo e salvando o dataset traduzido em: {output_filename}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Salvando '{config}'"):
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

    print("\n--- PROCESSO DE TRADUÇÃO CONCLUÍDO COM SUCESSO! ---")

# --- 4. EXECUÇÃO ---
if __name__ == "__main__":
    translate_stereoset_with_tower()
