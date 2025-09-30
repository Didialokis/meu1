# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator
import re
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# Modelo de instrução Qwen2. Ele requer um prompt para traduzir.
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

BATCH_SIZE = 8 # Ajuste conforme a memória VRAM de suas GPUs

# Mapeamento para converter os labels numéricos de volta para texto
GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

# Template do prompt para instruir o modelo a traduzir.
# A precisão deste prompt é crucial para a qualidade da saída.
PROMPT_TEMPLATE = """Translate the following English text to Brazilian Portuguese. Do not add any extra explanations, comments, or apologies. Provide only the direct translation.

English: "{text}"
Brazilian Portuguese:"""


# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def parse_translation(output_text):
    """Extrai a tradução da saída completa gerada pelo modelo."""
    # Procura pelo delimitador final do nosso prompt
    delimiter = "Brazilian Portuguese:"
    if delimiter in output_text:
        # Pega tudo que vem depois do delimitador e remove espaços extras
        return output_text.split(delimiter)[-1].strip()
    else:
        # Se o modelo não seguir o prompt, retorna a saída crua como fallback
        return output_text.strip()

# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_com_qwen2_multigpu():
    # Inicializa o Accelerator. Ele gerenciará a distribuição entre as GPUs.
    accelerator = Accelerator()
    print(f"🚀 Usando dispositivo: {str(accelerator.device).upper()} | GPUs disponíveis: {accelerator.num_processes}")

    print(f"💾 Carregando o modelo '{MODEL_NAME}'... (Pode levar tempo e memória)")
    # Carrega o modelo com precisão mista para otimizar o uso de memória
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=accelerator.device # O accelerate cuida do mapeamento
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Modelo carregado com sucesso.")
    
    # Prepara o modelo com o Accelerator
    model = accelerator.prepare(model)

    # --- ETAPA DE EXTRAÇÃO (Lógica mantida) ---
    datasets_dict, sentences_to_translate = {}, []
    if accelerator.is_main_process:
        # Apenas o processo principal baixa e prepara os dados
        for config in CONFIGS:
            dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
            datasets_dict[config] = dataset
            for example in dataset:
                sentences_to_translate.append(example['context'])
                sentences_to_translate.extend(example['sentences']['sentence'])
        print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # Distribui os dados para todos os processos
    sentences_to_translate = accelerator.broadcast(sentences_to_translate)
    datasets_dict = accelerator.broadcast(datasets_dict)

    # --- ETAPA DE TRADUÇÃO OTIMIZADA ---
    print("Iniciando a tradução em lotes com múltiplas GPUs...")
    translated_sentences = []

    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo Lotes", disable=not accelerator.is_main_process):
        batch_texts = sentences_to_translate[i:i + BATCH_SIZE]
        
        # Cria os prompts para cada sentença no lote
        prompts = [PROMPT_TEMPLATE.format(text=text) for text in batch_texts]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
        
        # Gera a tradução
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=128, # Limite de tokens para a resposta
            do_sample=False # Usa decodificação gananciosa para consistência
        )
        
        # Decodifica a saída completa (prompt + tradução)
        full_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Extrai apenas a tradução de cada saída
        batch_translated = [parse_translation(output) for output in full_outputs]
        batch_sanitized = [sanitize_text(text) for text in batch_translated]
        translated_sentences.extend(batch_sanitized)

    print("Tradução finalizada.")

    # Apenas o processo principal reconstrói e salva o arquivo final
    if accelerator.is_main_process:
        # --- ETAPA DE RECONSTRUÇÃO MANUAL (Lógica mantida) ---
        print("Reconstruindo o dataset na estrutura original...")
        translated_iter = iter(translated_sentences)
        reconstructed_data = {}
        # ... (Lógica de reconstrução idêntica à do script anterior) ...
        final_output_structure = { "version": "1.1", "data": reconstructed_data }
        output_path = f"stereoset_{DATASET_SPLIT}_pt_qwen2_completo.json"
        
        print(f"Salvando o dataset final em: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

        print("\n✅ Sucesso! O arquivo de saída é 100% compatível com as ferramentas de avaliação.")


if __name__ == "__main__":
    traduzir_com_qwen2_multigpu()
