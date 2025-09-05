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
# Ajuste o BATCH_SIZE conforme a VRAM total de suas GPUs.
# Comece com um valor baixo (ex: 4 ou 8) e aumente se não houver erros de memória.
BATCH_SIZE = 8

# --- 2. FUNÇÕES AUXILIARES ---

def create_translation_prompt(english_text):
    """
    Formata uma sentença em inglês no template de chat que o modelo espera.
    """
    # Usamos "Portuguese (Brazil)" para especificar o dialeto desejado.
    prompt = (
        "Translate the following English source text to Portuguese (Brazil):\n"
        f"English: {english_text}\n"
        "Portuguese (Brazil): "
    )
    # O modelo usa o formato de chat, então encapsulamos em uma lista de mensagens.
    return [{"role": "user", "content": prompt}]

def parse_generated_text(generated_output):
    """
    Extrai apenas a tradução do texto completo gerado pelo modelo.
    """
    # O modelo repete o prompt na saída, então pegamos apenas o texto após o marcador final.
    marker = "Portuguese (Brazil): "
    if marker in generated_output:
        return generated_output.split(marker)[-1].strip()
    else:
        # Caso o modelo não repita o prompt, retorna a saída como está.
        return generated_output.strip()


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def translate_stereoset_with_tower():
    """
    Função principal que carrega o dataset, o modelo, e executa a tradução em lote
    usando múltiplas GPUs.
    """
    print(f"--- INICIANDO TRADUÇÃO COM O MODELO: {MODEL_ID} ---")
    
    # Carrega o pipeline. `device_map="auto"` distribui o modelo pelas GPUs disponíveis.
    # A biblioteca `accelerate` é necessária para esta funcionalidade.
    print("Carregando o pipeline... Isso pode levar alguns minutos e consumir bastante memória.")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16, # Usa bfloat16 para melhor performance e menor uso de memória
        device_map="auto",
    )
    print("Modelo carregado com sucesso em todas as GPUs.")

    # Carrega os datasets originais
    original_datasets = {}
    all_sentences_to_translate = set()

    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        original_datasets[config] = dataset
        # Extrai todas as sentenças únicas para evitar traduções repetidas
        for example in dataset:
            all_sentences_to_translate.add(example['context'])
            all_sentences_to_translate.update(example['sentences']['sentence'])
    
    unique_english_sentences = list(all_sentences_to_translate)
    print(f"Total de {len(unique_english_sentences)} sentenças únicas para traduzir.")

    # Cria todos os prompts para tradução
    prompts = [create_translation_prompt(text) for text in unique_english_sentences]
    
    # Traduz em lotes para eficiência
    all_translations = []
    print(f"Iniciando a tradução em lotes de tamanho {BATCH_SIZE}...")
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        # O pipeline lida com a lista de prompts diretamente
        outputs = pipe(batch_prompts, max_new_tokens=256, do_sample=False, temperature=0.0)
        
        # Extrai e limpa a tradução de cada item do lote
        for output in outputs:
            # Acessa o texto gerado de dentro da estrutura de lista/dicionário
            generated_text = output[0]['generated_text'] 
            parsed_translation = parse_generated_text(generated_text)
            all_translations.append(parsed_translation)

    # Cria um mapa de tradução para reconstruir o dataset
    translation_map = dict(zip(unique_english_sentences, all_translations))
    print("Tradução de todas as sentenças concluída.")

    # Reconstrói e salva os datasets traduzidos
    for config, dataset in original_datasets.items():
        output_filename = f"stereoset_{config}_{DATASET_SPLIT}_pt_tower.jsonl"
        print(f"Reconstruindo e salvando o dataset traduzido em: {output_filename}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Salvando '{config}'"):
                # Cria uma cópia para não modificar o objeto original
                translated_example = example.copy()
                
                # Traduz o contexto
                translated_example['context'] = translation_map.get(example['context'], example['context'])
                
                # Traduz as sentenças alvo
                translated_sentences = [
                    translation_map.get(sent, sent) for sent in example['sentences']['sentence']
                ]
                translated_example['sentences']['sentence'] = translated_sentences
                
                # Salva o exemplo traduzido como uma linha no arquivo .jsonl
                f.write(json.dumps(translated_example, ensure_ascii=False) + '\n')

    print("\n--- PROCESSO DE TRADUÇÃO CONCLUÍDO COM SUCESSO! ---")


# --- 4. EXECUÇÃO ---

if __name__ == "__main__":
    translate_stereoset_with_tower()
