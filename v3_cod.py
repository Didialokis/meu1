

import json
import csv
from datasets import load_dataset
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# ATENÇÃO: Verifique se este é o nome do arquivo gerado pelo seu último script de tradução
# O script deve ser o que gera o arquivo no formato original e completo.
TRANSLATED_FILE = 'stereoset_validation_pt_nllb_formato_original_final.json' 

# Nome do arquivo CSV de saída que será gerado.
OUTPUT_CSV_FILE = 'avaliacao_completa.csv'

# Configurações do dataset original no Hugging Face.
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"


# --- 2. FUNÇÃO PRINCIPAL ---

def generate_full_evaluation_csv():
    """
    Função principal para carregar os dados originais e traduzidos,
    combiná-los e gerar um único arquivo CSV com todo o conteúdo.
    """
    print("🚀 Iniciando a geração do arquivo CSV completo para avaliação.")

    # --- Carregando o dataset traduzido (Português) ---
    print(f"📖 Lendo o arquivo traduzido: {TRANSLATED_FILE}")
    try:
        with open(TRANSLATED_FILE, 'r', encoding='utf-8') as f:
            # Acessa os dados dentro da chave "data"
            translated_data = json.load(f)['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo traduzido '{TRANSLATED_FILE}' não encontrado.")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ ERRO: Falha ao ler o arquivo JSON. Verifique se o arquivo tem a estrutura correta com a chave 'data'. Erro: {e}")
        return
        
    # --- Carregando o dataset original (Inglês) e criando mapas para busca rápida ---
    print("📚 Baixando o dataset original em Inglês para comparação...")
    en_context_map = {}  # Mapeia example_id -> contexto em inglês
    en_sentence_map = {} # Mapeia sentence_id -> sentença em inglês

    for config in CONFIGS:
        en_dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        for example in en_dataset:
            en_context_map[example['id']] = example['context']
            # Para o dataset original do HF, a estrutura é de listas paralelas
            sentence_ids = example['sentences']['id']
            sentence_texts = example['sentences']['sentence']
            for i in range(len(sentence_ids)):
                en_sentence_map[sentence_ids[i]] = sentence_texts[i]
    
    print(f"✅ {len(en_context_map)} contextos e {len(en_sentence_map)} sentenças em Inglês foram mapeados.")

    # --- Montando as linhas do CSV com dados em Inglês e Português ---
    print("✍️  Montando o arquivo CSV com todos os dados lado a lado...")
    csv_rows = []
    
    for task_type in ['intrasentence', 'intersentence']:
        # Itera sobre TODOS os exemplos, sem amostragem
        for translated_example in tqdm(translated_data.get(task_type, []), desc=f"Processando {task_type}"):
            example_id = translated_example['id']
            bias_type = translated_example['bias_type']
            
            # Busca o contexto original em inglês usando o ID do exemplo
            context_en = en_context_map.get(example_id, "N/A")
            
            # --- LÓGICA CORRIGIDA ---
            # Itera sobre a lista de dicionários de sentenças, que é a estrutura correta do arquivo traduzido
            for sentence_obj in translated_example['sentences']:
                sentence_id = sentence_obj['id']
                
                # Busca a sentença original em inglês usando o ID da sentença
                sentence_en = en_sentence_map.get(sentence_id, "N/A")

                # Monta uma linha (dicionário) para o CSV
                row = {
                    'task_type': task_type,
                    'bias_type': bias_type,
                    'example_id': example_id,
                    'context_en': context_en,
                    'context_pt': translated_example['context'],
                    'sentence_id': sentence_id,
                    'sentence_en': sentence_en,
                    'sentence_pt': sentence_obj['sentence'],
                    'gold_label': sentence_obj['gold_label'] # O label já está em formato de texto
                }
                csv_rows.append(row)

    # --- Salvando o arquivo CSV ---
    if not csv_rows:
        print("⚠️ Nenhuma linha foi gerada para o CSV. Verifique os arquivos de entrada.")
        return

    print(f"💾 Salvando {len(csv_rows)} linhas no arquivo '{OUTPUT_CSV_FILE}'...")
    
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
    generate_full_evaluation_csv()
///////////////////////////////////////////////////////////////
import json
import csv
import random
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# Arquivo JSON traduzido, gerado pelo script de tradução final
TRANSLATED_FILE = 'stereoset_validation_pt_nllb_formato_original_final.json' 

# Nomes dos arquivos de saída
FULL_OUTPUT_CSV = 'avaliacao_completa.csv'
SAMPLED_OUTPUT_CSV = 'amostra_aleatoria_avaliacao.csv'

# Quantos exemplos (contextos) selecionar aleatoriamente para cada categoria de viés
SAMPLES_PER_BIAS_TYPE = 10

# Configurações do dataset original no Hugging Face
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"


# --- 2. FUNÇÃO PRINCIPAL ---

def generate_csv_outputs():
    """
    Função principal que:
    1. Carrega os dados em português e inglês.
    2. Gera um CSV com a conversão completa.
    3. Gera um segundo CSV com uma amostra aleatória.
    """
    print("🚀 Iniciando a geração dos arquivos CSV.")

    # --- Carregando o dataset traduzido (Português) ---
    print(f"📖 Lendo o arquivo traduzido: {TRANSLATED_FILE}")
    try:
        with open(TRANSLATED_FILE, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo traduzido '{TRANSLATED_FILE}' não encontrado.")
        return
        
    # --- Carregando o dataset original (Inglês) e criando mapas para busca rápida ---
    print("📚 Baixando o dataset original em Inglês para comparação...")
    en_context_map = {}
    en_sentence_map = {}

    for config in CONFIGS:
        en_dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        for example in en_dataset:
            en_context_map[example['id']] = example['context']
            # CORREÇÃO: Itera sobre a lista de dicionários, que é a estrutura correta
            for sentence_obj in example['sentences']:
                en_sentence_map[sentence_obj['id']] = sentence_obj['sentence']
    
    print(f"✅ {len(en_context_map)} contextos e {len(en_sentence_map)} sentenças em Inglês foram mapeados.")

    # --- Processando dados e agrupando para amostragem ---
    all_csv_rows = []
    examples_by_bias_type = defaultdict(lambda: defaultdict(list))

    print("🧩 Processando e combinando dados de tradução...")
    for task_type in ['intrasentence', 'intersentence']:
        for translated_example in translated_data.get(task_type, []):
            example_id = translated_example['id']
            bias_type = translated_example['bias_type']
            
            # Agrupa o exemplo completo para a amostragem posterior
            examples_by_bias_type[task_type][bias_type].append(translated_example)
            
            context_en = en_context_map.get(example_id, "N/A")
            
            for sentence_obj in translated_example['sentences']:
                sentence_id = sentence_obj['id']
                sentence_en = en_sentence_map.get(sentence_id, "N/A")
                row = {
                    'task_type': task_type, 'bias_type': bias_type, 'example_id': example_id,
                    'context_en': context_en, 'context_pt': translated_example['context'],
                    'sentence_id': sentence_id, 'sentence_en': sentence_en,
                    'sentence_pt': sentence_obj['sentence'], 'gold_label': sentence_obj['gold_label']
                }
                all_csv_rows.append(row)

    # --- Salvando o arquivo CSV COMPLETO ---
    print(f"\n💾 Salvando a conversão completa ({len(all_csv_rows)} linhas) em '{FULL_OUTPUT_CSV}'...")
    headers = list(all_csv_rows[0].keys())
    with open(FULL_OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"✅ Arquivo '{FULL_OUTPUT_CSV}' gerado com sucesso!")

    # --- Gerando a AMOSTRA ALEATÓRIA ---
    print("\n🎲 Gerando amostra aleatória...")
    sampled_csv_rows = []
    for task_type, bias_types_dict in examples_by_bias_type.items():
        for bias_type, example_list in bias_types_dict.items():
            # Seleciona aleatoriamente N exemplos (ou menos, se não houver N)
            num_to_sample = min(SAMPLES_PER_BIAS_TYPE, len(example_list))
            random_sample = random.sample(example_list, num_to_sample)
            
            # Processa apenas os exemplos da amostra aleatória
            for example in random_sample:
                example_id = example['id']
                context_en = en_context_map.get(example_id, "N/A")
                for sentence_obj in example['sentences']:
                    sentence_id = sentence_obj['id']
                    sentence_en = en_sentence_map.get(sentence_id, "N/A")
                    row = {
                        'task_type': task_type, 'bias_type': bias_type, 'example_id': example_id,
                        'context_en': context_en, 'context_pt': example['context'],
                        'sentence_id': sentence_id, 'sentence_en': sentence_en,
                        'sentence_pt': sentence_obj['sentence'], 'gold_label': sentence_obj['gold_label']
                    }
                    sampled_csv_rows.append(row)

    # --- Salvando o arquivo CSV da AMOSTRA ---
    if not sampled_csv_rows:
        print("⚠️ Nenhuma amostra aleatória foi gerada.")
        return

    print(f"💾 Salvando a amostra aleatória ({len(sampled_csv_rows)} linhas) em '{SAMPLED_OUTPUT_CSV}'...")
    with open(SAMPLED_OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sampled_csv_rows)
    print(f"✅ Arquivo '{SAMPLED_OUTPUT_CSV}' gerado com sucesso!")
    
    print("\n🎉 Processo concluído!")


# --- 3. EXECUÇÃO ---
if __name__ == "__main__":
    generate_csv_outputs()




















////////////////////////////////////////////
import json
import csv
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

# ATENÇÃO: Verifique se este é o nome do arquivo gerado pelo seu script de tradução final
TRANSLATED_FILE = 'stereoset_validation_pt_nllb_formato_original_final.json' 

# Nome do arquivo CSV de saída que será gerado.
OUTPUT_CSV_FILE = 'amostra_avaliacao_final.csv'

# Quantos exemplos (contextos) selecionar para cada categoria de viés.
SAMPLES_PER_BIAS_TYPE = 10

# Configurações do dataset original no Hugging Face.
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"


# --- 2. FUNÇÃO PRINCIPAL ---

def generate_evaluation_csv():
    """
    Função principal para carregar os dados, criar os mapas de correspondência,
    selecionar as amostras corretamente e gerar o arquivo CSV.
    """
    print("🚀 Iniciando a geração do arquivo CSV de amostra para avaliação.")

    # --- Carregando o dataset traduzido (Português) ---
    print(f"📖 Lendo o arquivo traduzido: {TRANSLATED_FILE}")
    try:
        with open(TRANSLATED_FILE, 'r', encoding='utf-8') as f:
            # Acessa a chave 'data' na estrutura do arquivo
            translated_data = json.load(f)['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo traduzido '{TRANSLATED_FILE}' não encontrado.")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ ERRO: Falha ao ler o arquivo JSON. Verifique se ele contém a chave 'data'. Erro: {e}")
        return
        
    # --- Carregando o dataset original (Inglês) e criando mapas para busca rápida ---
    print("📚 Baixando o dataset original em Inglês para comparação...")
    en_context_map = {}
    en_sentence_map = {}

    for config in CONFIGS:
        en_dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        for example in en_dataset:
            en_context_map[example['id']] = example['context']
            # Acessa a estrutura de dicionário de listas do dataset original
            for i in range(len(example['sentences']['id'])):
                sent_id = example['sentences']['id'][i]
                sent_text = example['sentences']['sentence'][i]
                en_sentence_map[sent_id] = sent_text
    
    print(f"✅ {len(en_context_map)} contextos e {len(en_sentence_map)} sentenças em Inglês foram mapeados.")

    # --- INÍCIO DA CORREÇÃO 1: LÓGICA DE AMOSTRAGEM ---
    # A amostragem agora é separada por 'task_type' para garantir a contagem correta.
    print(f"🔍 Selecionando {SAMPLES_PER_BIAS_TYPE} exemplos de cada categoria de viés por tarefa...")
    
    # Estrutura aninhada: { 'task_type': { 'bias_type': [lista_de_exemplos] } }
    sampled_examples = defaultdict(lambda: defaultdict(list))
    
    for task_type in ['intrasentence', 'intersentence']:
        for translated_example in translated_data.get(task_type, []):
            bias_type = translated_example['bias_type']
            
            # Adiciona à amostra APENAS se tivermos menos de 10 para esta tarefa E este tipo de viés
            if len(sampled_examples[task_type][bias_type]) < SAMPLES_PER_BIAS_TYPE:
                sampled_examples[task_type][bias_type].append(translated_example)
    # --- FIM DA CORREÇÃO 1 ---

    # --- Montando as linhas do CSV com dados em Inglês e Português ---
    print("✍️ Montando o arquivo CSV com dados lado a lado...")
    csv_rows = []
    
    # Itera sobre a estrutura de amostragem corrigida
    for task_type, bias_types_dict in sampled_examples.items():
        print(f"\n--- Tarefa: {task_type} ---")
        total_task_samples = 0
        for bias_type, examples in bias_types_dict.items():
            print(f"  - Categoria '{bias_type}': {len(examples)} exemplos selecionados.")
            total_task_samples += len(examples)
            for example in examples:
                example_id = example['id']
                context_en = en_context_map.get(example_id, "N/A")
                
                # --- INÍCIO DA CORREÇÃO 2: LÓGICA DE LEITURA DOS DADOS ---
                # O loop agora itera sobre a 'lista de dicionários', que é a estrutura correta do arquivo.
                for sentence_obj in example['sentences']:
                    sentence_id = sentence_obj['id']
                    sentence_pt = sentence_obj['sentence']
                    
                    # O 'gold_label' já é texto no arquivo final, não precisa de conversão.
                    gold_label_str = sentence_obj['gold_label']
                    
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
                        'gold_label': gold_label_str
                    }
                    csv_rows.append(row)
                # --- FIM DA CORREÇÃO 2 ---
        print(f"  Total para {task_type}: {total_task_samples} contextos.")

    # --- Salvando o arquivo CSV ---
    if not csv_rows:
        print("⚠️ Nenhuma amostra foi gerada. Verifique os arquivos de entrada.")
        return

    print(f"\n💾 Salvando {len(csv_rows)} linhas de amostra em '{OUTPUT_CSV_FILE}'...")
    
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
