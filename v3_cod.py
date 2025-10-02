import json
import csv
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# ATENÇÃO: Verifique se este é o nome do arquivo gerado pelo seu último script de tradução
TRANSLATED_FILE = 'stereoset_validation_pt_nllb_completo.json' 

# Nome do arquivo CSV de saída que será gerado.
OUTPUT_CSV_FILE = 'amostra_avaliacao.csv'

# Quantos exemplos selecionar para cada categoria de viés.
SAMPLES_PER_BIAS_TYPE = 10

# Configurações do dataset original no Hugging Face.
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"

# Mapeamento para converter os labels numéricos de volta para texto
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}

# --- 2. FUNÇÃO PRINCIPAL ---

def generate_evaluation_csv():
    """
    Função principal para carregar os dados, criar os mapas de correspondência,
    selecionar as amostras e gerar o arquivo CSV.
    """
    print("🚀 Iniciando a geração do arquivo CSV de amostra para avaliação.")

    # --- Carregando o dataset traduzido (Português) ---
    print(f"📖 Lendo o arquivo traduzido: {TRANSLATED_FILE}")
    try:
        with open(TRANSLATED_FILE, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo traduzido '{TRANSLATED_FILE}' não encontrado.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ ERRO: Falha ao ler o arquivo JSON. Erro: {e}")
        return
        
    # --- Carregando o dataset original (Inglês) e criando mapas para busca rápida ---
    print("📚 Baixando o dataset original em Inglês para comparação...")
    en_context_map = {}
    en_sentence_map = {}

    for config in CONFIGS:
        en_dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        for example in en_dataset:
            en_context_map[example['id']] = example['context']
            sentence_ids = example['sentences']['id']
            sentence_texts = example['sentences']['sentence']
            for i in range(len(sentence_ids)):
                en_sentence_map[sentence_ids[i]] = sentence_texts[i]
    
    print(f"✅ {len(en_context_map)} contextos e {len(en_sentence_map)} sentenças em Inglês foram mapeados.")

    # --- Selecionando 10 amostras de cada categoria de viés ---
    print(f"🔍 Selecionando {SAMPLES_PER_BIAS_TYPE} exemplos de cada categoria de viés...")
    
    sampled_examples = defaultdict(list)
    
    for task_type in ['intrasentence', 'intersentence']:
        for translated_example in tqdm(translated_data.get(task_type, []), desc=f"Processando {task_type}"):
            bias_type = translated_example['bias_type']
            
            if len(sampled_examples[bias_type]) < SAMPLES_PER_BIAS_TYPE:
                sampled_examples[bias_type].append(translated_example)

    print(f"📦 Amostras selecionadas para as categorias: {list(sampled_examples.keys())}")

    # --- Montando as linhas do CSV com dados em Inglês e Português ---
    print("✍️ Montando o arquivo CSV com dados lado a lado...")
    csv_rows = []
    for bias_type, examples in sampled_examples.items():
        for example in examples:
            example_id = example['id']
            task_type = 'intrasentence' if any(ex['id'] == example_id for ex in translated_data.get('intrasentence', [])) else 'intersentence'
            context_en = en_context_map.get(example_id, "N/A")
            
            # --- INÍCIO DA CORREÇÃO ---
            # Acessa o dicionário de listas paralelas
            sentences_data = example['sentences']
            sent_ids = sentences_data['id']
            sent_texts_pt = sentences_data['sentence']
            sent_gold_labels = sentences_data['gold_label']

            # Itera sobre as listas usando um índice para "desfazer" a estrutura paralela
            for i in range(len(sent_ids)):
                sentence_id = sent_ids[i]
                sentence_pt = sent_texts_pt[i]
                gold_label_int = sent_gold_labels[i]
                
                sentence_en = en_sentence_map.get(sentence_id, "N/A")

                row = {
                    'task_type': task_type,
                    'bias_type': bias_type,
                    'example_id': example_id,
                    'context_en': context_en,
                    'context_pt': example['context'],
                    'sentence_id': sentence_id,
                    'sentence_en': sentence_en,
                    'sentence_pt': sentence_pt,
                    'gold_label': LABEL_MAP.get(gold_label_int, "N/A") # Converte o número para texto
                }
                csv_rows.append(row)
            # --- FIM DA CORREÇÃO ---

    # --- Salvando o arquivo CSV ---
    if not csv_rows:
        print("⚠️ Nenhuma amostra foi gerada. Verifique os arquivos de entrada.")
        return

    print(f"💾 Salvando {len(csv_rows)} linhas de amostra em '{OUTPUT_CSV_FILE}'...")
    
    headers = [
        'task_type', 'bias_type', 'example_id', 
        'context_en', 'context_pt', 'sentence_id', 
        'sentence_en', 'sentence_pt', 'gold_label'
    ]

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n🎉 Arquivo '{OUTPUT_CSV_FILE}' gerado com sucesso!")


# --- 3. EXECUÇÃO ---

if __name__ == "__main__":
    generate_evaluation_csv()
