# -*- coding: utf-8 -*-

import torch
from transformers import pipeline
from datasets import load_dataset
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# Modelo instrucional da Unbabel
MODEL_ID = "Unbabel/Tower-Plus-2B"

# Dataset a ser traduzido
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Parâmetros de processamento
BATCH_SIZE = 8

# --- 2. FUNÇÕES AUXILIARES ---

def create_translation_prompt(english_text):
    """
    Formata uma sentença em inglês no template de chat que o modelo espera.
    """
    prompt = (
        "Translate the following English source text to Portuguese (Brazil):\n"
        f"English: {english_text}\n"
        "Portuguese (Brazil): "
    )
    return [{"role": "user", "content": prompt}]

def parse_generated_text(generated_output):
    """
    Extrai apenas a tradução do texto completo gerado pelo modelo.
    Esta versão é robusta para lidar com saídas que são listas.
    """
    # --- INÍCIO DA CORREÇÃO ---
    text_to_parse = generated_output

    # 1. Verifica se a saída é uma lista e extrai o primeiro elemento.
    if isinstance(text_to_parse, list):
        text_to_parse = text_to_parse[0] if text_to_parse else ""

    # 2. Garante que o que sobrou é uma string antes de continuar.
    if not isinstance(text_to_parse, str):
        return "" # Retorna string vazia para formatos inesperados

    # 3. Lógica original para extrair a tradução do prompt.
    marker = "Portuguese (Brazil): "
    if marker in text_to_parse:
        return text_to_parse.split(marker)[-1].strip()
    else:
        return text_to_parse.strip()
    # --- FIM DA CORREÇÃO ---


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def translate_stereoset_with_tower():
    """
    Função principal que carrega o dataset, o modelo, e executa a tradução em lote
    usando múltiplas GPUs.
    """
    print(f"--- INICIANDO TRADUÇÃO COM O MODELO: {MODEL_ID} ---")
    
    print("Carregando o pipeline... Isso pode levar alguns minutos e consumir bastante memória.")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Modelo carregado com sucesso em todas as GPUs.")

    original_datasets = {}
    all_sentences_to_translate = set()

    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        original_datasets[config] = dataset
        for example in dataset:
            all_sentences_to_translate.add(example['context'])
            all_sentences_to_translate.update(example['sentences']['sentence'])
    
    unique_english_sentences = list(all_sentences_to_translate)
    print(f"Total de {len(unique_english_sentences)} sentenças únicas para traduzir.")

    prompts = [create_translation_prompt(text) for text in unique_english_sentences]
    
    all_translations = []
    print(f"Iniciando a tradução em lotes de tamanho {BATCH_SIZE}...")
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        outputs = pipe(batch_prompts, max_new_tokens=256, do_sample=False)
        
        for output in outputs:
            generated_text = output[0]['generated_text'] 
            parsed_translation = parse_generated_text(generated_text)
            all_translations.append(parsed_translation)

    translation_map = dict(zip(unique_english_sentences, all_translations))
    print("Tradução de todas as sentenças concluída.")

# ... (todo o código anterior permanece o mesmo) ...

    # Reconstrói e salva os datasets traduzidos
    for config, dataset in original_datasets.items():
        output_filename = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.jsonl"
        print(f"Reconstruindo e salvando o dataset traduzido em: {output_filename}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Salvando '{config}'"):
                
                # --- INÍCIO DA CORREÇÃO ---
                # Em vez de modificar o 'example' original, construímos um novo dicionário Python
                # "puro" a partir dele. Isso garante que não haja tipos de dados complexos
                # que possam quebrar a função json.dumps.
                
                clean_example = {
                    "id": example["id"],
                    "target": example["target"],
                    "bias_type": example["bias_type"],
                    "context": translation_map.get(example["context"], example["text"]),
                    "sentences": {
                        "sentence": [
                            translation_map.get(sent, sent)
                            for sent in example["sentences"]["sentence"]
                        ],
                        # Copia explicitamente os labels para garantir que são listas puras.
                        # O .tolist() é uma segurança extra se for um array numpy, por exemplo.
                        "gold_label": list(example["sentences"]["gold_label"]),
                    },
                }
                # --- FIM DA CORREÇÃO ---
                
                # Salva o dicionário limpo como uma linha no arquivo .jsonl
                f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')

    print("\n--- PROCESSO DE TRADUÇÃO CONCLUÍDO COM SUCESSO! ---")


# --- 4. EXECUÇÃO ---

if __name__ == "__main__":
    translate_stereoset_with_tower()
