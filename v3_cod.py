import json
import csv
import random
from datasets import load_dataset
from collections import defaultdict

# --- CONFIGURAÇÕES ---
# Arquivo "ouro" traduzido, gerado pelo seu script de tradução.
# Verifique se o nome do arquivo está correto.
TRADUCTION_FILE = 'stereoset_validation_pt_nllb_formato_original_final.json' 

# Nome do arquivo CSV de saída.
OUTPUT_CSV_FILE = 'amostra_avaliacao_stereoset.csv'

# Número de exemplos a serem amostrados por cada tipo de viés.
NUM_SAMPLES_PER_BIAS = 10

# Mapeamento de labels numéricos para texto (se necessário, mas o arquivo já deve ter texto)
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
# ---------------------

def create_sample_csv():
    """
    Função principal que carrega os datasets original e traduzido,
    seleciona amostras aleatórias por tipo de viés e salva em um arquivo CSV.
    """
    print("🚀 Iniciando a geração da amostra CSV...")

    # --- 1. Carregar o dataset original do Hugging Face ---
    print("💾 Carregando dataset original do Hugging Face...")
    original_datasets = {}
    for config in ['intersentence', 'intrasentence']:
        original_datasets[config] = load_dataset("McGill-NLP/stereoset", config, split="validation")
    
    # Mapeia cada ID de exemplo para seus dados originais para busca rápida
    original_data_map = {}
    for task_type, dataset in original_datasets.items():
        for example in dataset:
            original_data_map[example['id']] = {
                "context": example['context'],
                # Converte a estrutura de listas paralelas para lista de dicionários
                "sentences": [
                    {"id": sid, "sentence": s_text, "gold_label": LABEL_MAP[s_label]}
                    for sid, s_text, s_label in zip(
                        example['sentences']['id'],
                        example['sentences']['sentence'],
                        example['sentences']['gold_label']
                    )
                ]
            }
    print(f"✅ Encontrados {len(original_data_map)} exemplos originais.")

    # --- 2. Carregar o dataset traduzido ---
    try:
        with open(TRADUCTION_FILE, 'r', encoding='utf-8') as f:
            translated_full_data = json.load(f)
            translated_data = translated_full_data['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo traduzido '{TRADUCTION_FILE}' não encontrado.")
        return

    # --- 3. Agrupar exemplos traduzidos por tipo de viés ---
    examples_by_bias = defaultdict(list)
    for task_type in translated_data:
        for example in translated_data[task_type]:
            examples_by_bias[example['bias_type']].append(example)

    # --- 4. Selecionar amostras aleatórias e preparar dados para o CSV ---
    csv_rows = []
    
    print(f" sampling {NUM_SAMPLES_PER_BIAS} exemplos para cada tipo de viés...")
    for bias_type, examples in examples_by_bias.items():
        # Embaralha e seleciona os N primeiros exemplos
        random.shuffle(examples)
        sampled_examples = examples[:NUM_SAMPLES_PER_BIAS]
        
        for example_pt in sampled_examples:
            example_id = example_pt['id']
            # Busca o exemplo original correspondente
            example_en = original_data_map.get(example_id)
            
            if not example_en:
                continue

            # Para cada tipo de sentença (stereotype, anti-stereotype, unrelated)
            for sentence_pt in example_pt['sentences']:
                label = sentence_pt['gold_label']
                
                # Encontra a sentença original com o mesmo label
                sentence_en = next((s for s in example_en['sentences'] if s['gold_label'] == label), None)
                
                if sentence_en:
                    csv_rows.append({
                        'bias_type': bias_type,
                        'example_id': example_id,
                        'sentence_type': label,
                        'contexto_original_EN': example_en['context'],
                        'contexto_traduzido_PT': example_pt['context'],
                        'sentenca_original_EN': sentence_en['sentence'],
                        'sentenca_traduzida_PT': sentence_pt['sentence'],
                    })

    # --- 5. Escrever o arquivo CSV ---
    if not csv_rows:
        print("⚠️ Nenhuma linha foi preparada para o CSV. Verifique os arquivos de entrada.")
        return

    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            # Define os cabeçalhos das colunas
            fieldnames = [
                'bias_type', 'example_id', 'sentence_type', 
                'contexto_original_EN', 'contexto_traduzido_PT',
                'sentenca_original_EN', 'sentenca_traduzida_PT'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n🎉 Sucesso! Arquivo '{OUTPUT_CSV_FILE}' foi gerado com {len(csv_rows)} linhas.")
        
    except Exception as e:
        print(f"❌ ERRO ao escrever o arquivo CSV: {e}")


if __name__ == "__main__":
    create_sample_csv()
