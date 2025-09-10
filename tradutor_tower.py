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
    # Template ajustado para ser mais claro para o modelo
    prompt = (
        "Translate the following English text to Brazilian Portuguese.\n"
        f"English source: \"{english_text}\"\n"
        "Brazilian Portuguese translation: "
    )
    return [{"role": "user", "content": prompt}]

def parse_generated_text(generated_text, prompt_text):
    """
    Extrai a tradução da saída completa do modelo de forma robusta.
    """
    # Caso 1: A saída do modelo inclui o prompt. Removemos o prompt.
    # Usamos o prompt original (sem o dicionário 'role'/'user') para a verificação.
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].strip()
    
    # Caso 2: A saída do modelo é apenas a tradução.
    # Removemos o marcador final caso ele esteja presente na saída.
    marker = "Brazilian Portuguese translation: "
    if marker in generated_text:
        # Pega a última ocorrência para evitar problemas se o marcador aparecer no texto.
        return generated_text.split(marker)[-1].strip()
    
    # Caso 3 (Fallback): Se nenhum dos casos acima funcionar, retorna a saída como está.
    return generated_text.strip()


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
        # Garante que apenas strings não-vazias sejam adicionadas
        for example in dataset:
            if example['context']: all_sentences_to_translate.add(example['context'])
            all_sentences_to_translate.update([s for s in example['sentences']['sentence'] if s])
    
    unique_english_sentences = sorted(list(all_sentences_to_translate)) # Ordenado para consistência
    print(f"Total de {len(unique_english_sentences)} sentenças únicas para traduzir.")

    # Gera os prompts formatados para o modelo
    prompts_for_model = [create_translation_prompt(text) for text in unique_english_sentences]
    
    # Guarda também o texto puro do prompt para a função de parse
    raw_prompts = [p[0]['content'] for p in prompts_for_model]

    all_translations = []
    print(f"Iniciando a tradução em lotes de tamanho {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(prompts_for_model), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch_prompts_model = prompts_for_model[i:i + BATCH_SIZE]
        batch_raw_prompts = raw_prompts[i:i + BATCH_SIZE]
        
        # O pipeline retorna uma lista de listas, uma para cada prompt no lote
        outputs = pipe(batch_prompts_model, max_new_tokens=256, do_sample=False)
        
        # Itera sobre os prompts e as saídas correspondentes
        for prompt_text, output in zip(batch_raw_prompts, outputs):
            # A saída real está dentro de uma estrutura aninhada
            generated_text = output[0]['generated_text']
            
            # --- LINHA DE DEPURAÇÃO ---
            # Descomente a linha abaixo para ver a saída bruta do modelo para cada sentença.
            # print(f"\n--- MODELO GEROU ---\n{generated_text}\n--------------------")
            
            parsed_translation = parse_generated_text(generated_text, prompt_text)
            all_translations.append(parsed_translation)

    # Cria o mapa de tradução: {sentença_original: sentença_traduzida}
    translation_map = dict(zip(unique_english_sentences, all_translations))
    print("Tradução de todas as sentenças concluída.")

    # Reconstrói e salva os datasets traduzidos
    for config, dataset in original_datasets.items():
        output_filename = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.jsonl"
        print(f"Reconstruindo e salvando o dataset traduzido em: {output_filename}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Salvando '{config}'"):
                # Constrói um dicionário "limpo" para garantir a validade do JSON
                clean_example = {
                    "id": example["id"],
                    "target": example["target"],
                    "bias_type": example["bias_type"],
                    # Usa .get() com um fallback para a sentença original se a tradução falhar
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
