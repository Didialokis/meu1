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
CONFIGS = ['intersentence', 'intrasentence'] # Processará ambos corretamente
DATASET_SPLIT = "validation"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"
BATCH_SIZE = 8

# Mapeamento para converter os labels numéricos de volta para texto
GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

# --- 2. FUNÇÃO AUXILIAR ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_e_recriar_estrutura_corretamente():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- ETAPA DE EXTRAÇÃO (AJUSTADA) ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            # <--- MUDANÇA 1: SÓ EXTRAI O CONTEXTO SE ELE EXISTIR (para intersentence) --->
            if 'context' in example and example['context']:
                sentences_to_translate.append(example['context'])
            # As frases sempre existem, então as adicionamos incondicionalmente
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- ETAPA DE TRADUÇÃO (SEM MUDANÇAS) ---
    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=128)
        batch_translated_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        batch_sanitized = [sanitize_text(text) for text in batch_translated_raw]
        translated_sentences.extend(batch_sanitized)
    print("Tradução finalizada.")

    # --- ETAPA DE RECONSTRUÇÃO (AJUSTADA) ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    reconstructed_data = {}
    for config in CONFIGS:
        original_dataset = datasets_dict[config]
        new_examples_list = []
        for original_example in tqdm(original_dataset, desc=f"Reconstruindo {config}"):
            # Cria a base do exemplo
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "sentences": [] # Será preenchido a seguir
            }
            
            # <--- MUDANÇA 2: SÓ ADICIONA A CHAVE 'context' SE FOR UM EXEMPLO 'intersentence' --->
            if config == 'intersentence':
                new_example["context"] = next(translated_iter)
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # Recria a lista de dicionários para o campo 'labels'
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({
                        "human_id": human_ids[j],
                        "label": INNER_LABEL_MAP[inner_int_labels[j]]
                    })

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": next(translated_iter),
                    "labels": recreated_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO (SEM MUDANÇAS) ---
    final_output_structure = {
        "version": "1.1",
        "data": {
            # O arquivo final terá ambas as seções, como o original dev.json
            "intrasentence": reconstructed_data.get('intrasentence', []),
            "intersentence": reconstructed_data.get('intersentence', [])
        }
    }
    
    output_path = f"dev_pt.json" # Nome mais padronizado
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída agora contém ambas as estruturas (intra/inter) e é compatível com o script de avaliação.")


if __name__ == "__main__":
    traduzir_e_recriar_estrutura_corretamente()
