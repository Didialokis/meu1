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
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"
BATCH_SIZE = 8

# Mapeamento para converter os labels numéricos de volta para texto
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}

# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_e_recriar_estrutura_final():
    """
    Executa o pipeline de tradução e recria a estrutura original do Stereoset
    com precisão para garantir compatibilidade com o dataloader oficial.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- ETAPA DE EXTRAÇÃO ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        # A flag `keep_in_memory=True` pode ajudar na estabilidade do acesso aos dados
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- ETAPA DE TRADUÇÃO ---
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

    # --- ETAPA DE RECONSTRUÇÃO MANUAL (LÓGICA CORRIGIDA) ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    reconstructed_data = {}
    for config in CONFIGS:
        original_dataset = datasets_dict[config]
        new_examples_list = []
        for original_example in tqdm(original_dataset, desc=f"Reconstruindo {config}"):
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "context": next(translated_iter),
                "sentences": []
            }
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # --- INÍCIO DA CORREÇÃO FINAL ---
                # Recria a lista de dicionários para o campo 'labels'
                original_labels_list_of_dicts = original_sents_data['labels'][i]
                recreated_labels = []
                # O campo 'labels' no dataset do HF já é uma lista de dicionários,
                # mas vamos garantir a estrutura correta explicitamente.
                if isinstance(original_labels_list_of_dicts, list) and all(isinstance(item, dict) for item in original_labels_list_of_dicts):
                     recreated_labels = original_labels_list_of_dicts
                else:
                    # Caso de fallback se a estrutura for inesperada (pouco provável)
                    # Isso garante que o código não quebre, mesmo que a estrutura mude.
                    if 'human_id' in original_labels_list_of_dicts and 'label' in original_labels_list_of_dicts:
                        ids = original_labels_list_of_dicts['human_id']
                        lbls = original_labels_list_of_dicts['label']
                        recreated_labels = [{"human_id": hid, "label": lbl} for hid, lbl in zip(ids, lbls)]

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": next(translated_iter),
                    "labels": recreated_labels, # Usa a lista de dicionários recriada
                    "gold_label": LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                # --- FIM DA CORREÇÃO FINAL ---
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_nllb_formato_original_final.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída agora é 100% compatível com o dataloader.py.")


if __name__ == "__main__":
    traduzir_e_recriar_estrutura_final()
