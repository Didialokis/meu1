# -*- coding: utf-8 -*-

# As importações mudaram para usar vLLM em vez de transformers para o modelo
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
import re

# --- CONFIGURAÇÕES ---
# Novo modelo da Unbabel
MODEL_NAME = "Unbabel/Tower-Plus-9B" 

DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Parâmetros para o processamento em lote com vLLM
BATCH_SIZE = 32 # vLLM é eficiente, podemos tentar um batch size maior

# --- 1. PREPARAÇÃO DO MODELO COM VLLM ---

# Define os parâmetros de amostragem para a geração de texto
# Temperatura 0 para respostas mais determinísticas, ideal para tradução
sampling_params = SamplingParams(
  best_of=1,
  temperature=0,
  max_tokens=256, # 256 tokens devem ser suficientes para a maioria das sentenças
)

# Carrega o modelo usando o LLM do vLLM
# Certifique-se de ter GPU com VRAM suficiente (ex: A100, H100)
print(f"Carregando o modelo '{MODEL_NAME}' com vLLM...")
# O tensor_parallel_size=1 é para uma única GPU. Ajuste se tiver mais.
llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)
print("Modelo carregado com sucesso.")


def sanitize_text(text):
    """
    Função para "limpar" o texto, removendo caracteres de controle.
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def traduzir_dataset_com_vllm():
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

    # --- 2. LÓGICA DE TRADUÇÃO EM LOTE ADAPTADA PARA VLLM ---
    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        
        # Cria a lista de prompts no formato esperado pelo modelo Tower
        prompts = [
            f"Translate the following English source text to Portuguese (Brazil):\nEnglish: {sentence}\nPortuguese (Brazil): "
            for sentence in batch
        ]
        
        # Executa a geração em lote com vLLM
        outputs = llm.generate(prompts, sampling_params)
        
        # Extrai o texto traduzido de cada saída
        batch_translated_raw = [output.outputs[0].text.strip() for output in outputs]
        
        # Aplica a sanitização
        batch_sanitized = [sanitize_text(text) for text in batch_translated_raw]
        translated_sentences.extend(batch_sanitized)
        
        print(f"  Lote {i//BATCH_SIZE + 1} de {len(sentences_to_translate)//BATCH_SIZE + 1} concluído...")

    print("Tradução finalizada.")

    # --- 3. RECONSTRUÇÃO E SALVAMENTO (sem alterações) ---
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
        
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.json" # Nome de arquivo atualizado
        print(f"Salvando o dataset '{config}' traduzido em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False, indent=2)

    print("\nSucesso! Processo concluído.")

if __name__ == "__main__":
    traduzir_dataset_com_vllm()
