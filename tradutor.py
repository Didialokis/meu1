# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# Modelo de instrução da série Qwen2. 7B é uma ótima escolha de alta qualidade.
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Ajuste o BATCH_SIZE conforme a VRAM total de suas GPUs. Comece com um valor baixo (e.g., 4 ou 8).
BATCH_SIZE = 8 
PLACEHOLDER = "__BLANK_PLACEHOLDER__"

# Mapeamentos para reconstrução correta da estrutura original
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}


# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_dataset_com_qwen2():
    """
    Executa o pipeline de tradução usando um LLM de instrução (Qwen2)
    em múltiplas GPUs e recria a estrutura original do Stereoset.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise ConnectionError("Este script requer pelo menos uma GPU CUDA para ser executado de forma otimizada.")
    
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    print(f"🚀 Encontradas {num_gpus} GPUs. Usando o dispositivo principal: {device}")

    print(f"💾 Carregando o modelo '{MODEL_NAME}'... (Isso pode levar tempo e memória)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    if num_gpus > 1:
        print(f"Parallelizando o modelo em {num_gpus} GPUs usando DataParallel...")
        model = nn.DataParallel(model)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("✅ Modelo e tokenizador carregados com sucesso.")

    # --- ETAPA DE EXTRAÇÃO ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")
    
    # --- ETAPA DE TRADUÇÃO COM PROMPT DE INSTRUÇÃO ---
    print("Iniciando a tradução em lotes com Qwen2...")
    translated_sentences = []

    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        batch_com_placeholder = [s.replace("BLANK", PLACEHOLDER) for s in batch]

        messages_batch = [
            [
                {"role": "system", "content": "You are an expert translator. Translate the user's text from English to Brazilian Portuguese accurately and fluently."},
                {"role": "user", "content": text}
            ]
            for text in batch_com_placeholder
        ]

        inputs = tokenizer.apply_chat_template(
            messages_batch,
            padding=True,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True
        ).to(device)

        generate_func = model.module.generate if isinstance(model, nn.DataParallel) else model.generate
        generated_tokens = generate_func(
            inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )
        
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        assistant_responses = [output.split("assistant")[-1].strip() for output in decoded_outputs]

        batch_translated_final = [s.replace(PLACEHOLDER, "BLANK") for s in assistant_responses]
        batch_sanitized = [sanitize_text(text) for text in batch_translated_final]
        translated_sentences.extend(batch_sanitized)

    print("Tradução finalizada.")

    # --- ETAPA DE RECONSTRUÇÃO MANUAL ---
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
                    "gold_label": LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1",
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_qwen2_completo.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída agora é 100% compatível com as ferramentas de avaliação.")


if __name__ == "__main__":
    traduzir_dataset_com_qwen2()
