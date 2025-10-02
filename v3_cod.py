import json
import pandas as pd
import os

# --- 1. CONFIGURAÇÕES ---

# Nomes dos arquivos de entrada gerados pelo script de tradução.
# O script irá procurar por arquivos que terminem com estes nomes.
INPUT_FILES_SUFFIX = [
    "intrasentence_validation_pt_nllb_final.json",
    "intersentence_validation_pt_nllb_final.json"
]

# Número de exemplos a serem amostrados para cada tipo de viés.
SAMPLES_PER_BIAS_TYPE = 10

# Mapeamento para converter os labels numéricos de volta para texto (para clareza no CSV).
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}

# --- 2. FUNÇÃO PRINCIPAL ---

def criar_amostra_csv(input_path, output_path):
    """
    Carrega um arquivo JSON do Stereoset, cria uma amostra aleatória de 10 exemplos
    por tipo de viés e salva o resultado como um arquivo CSV.
    """
    print(f"📄 Processando arquivo: {input_path}...")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  ❌ ERRO: Arquivo não encontrado. Pulando.")
        return
    except json.JSONDecodeError as e:
        print(f"  ❌ ERRO: Arquivo JSON malformado. Detalhe: {e}. Pulando.")
        return

    # --- Achatando a estrutura JSON para um formato de tabela ---
    flat_data = []
    task_type = "intrasentence" if "intrasentence" in input_path else "intersentence"

    for cluster in data:
        # Informações do "cluster" (exemplo principal)
        cluster_id = cluster.get('id', 'N/A')
        bias_type = cluster.get('bias_type', 'N/A')
        target = cluster.get('target', 'N/A')
        context = cluster.get('context', 'N/A')
        
        # Informações das sentenças individuais
        sentences_data = cluster.get('sentences', {})
        sentence_texts = sentences_data.get('sentence', [])
        sentence_ids = sentences_data.get('id', [])
        gold_labels = sentences_data.get('gold_label', [])

        for i in range(len(sentence_texts)):
            flat_data.append({
                "task_type": task_type,
                "cluster_id": cluster_id,
                "bias_type": bias_type,
                "target": target,
                "context": context,
                "sentence_id": sentence_ids[i],
                "sentence_text": sentence_texts[i],
                # Converte o label numérico para texto para facilitar a leitura
                "gold_label": LABEL_MAP.get(gold_labels[i], 'desconhecido') 
            })
    
    if not flat_data:
        print("  ⚠️ AVISO: Nenhum dado foi extraído do arquivo.")
        return

    # Converte para um DataFrame do Pandas
    df = pd.DataFrame(flat_data)
    print(f"  📊 Total de {len(df)} sentenças carregadas.")

    # --- Amostragem: 10 exemplos por tipo de viés ---
    print(f"  🎲 Selecionando {SAMPLES_PER_BIAS_TYPE} exemplos aleatórios por 'bias_type'...")

    # Agrupa por 'bias_type' e, para cada grupo, pega uma amostra.
    # Usamos uma função lambda para pegar no máximo 10, ou menos se o grupo for menor.
    sampled_df = df.groupby('bias_type').apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_BIAS_TYPE), random_state=42)
    ).reset_index(drop=True)

    # Ordena o resultado para facilitar a visualização
    sampled_df = sampled_df.sort_values(by=['bias_type', 'cluster_id']).reset_index(drop=True)

    print(f"  📝 Total de {len(sampled_df)} sentenças na amostra final.")
    
    # --- Salvando em CSV ---
    # Usamos encoding='utf-8-sig' para garantir compatibilidade com Excel
    sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Amostra salva com sucesso em: {output_path}")


# --- 3. EXECUÇÃO ---

if __name__ == "__main__":
    print("--- Iniciando Script de Amostragem e Conversão para CSV ---\n")
    
    # Encontra os arquivos na pasta atual que correspondem aos sufixos
    files_to_process = [f for f in os.listdir('.') if any(f.endswith(suffix) for suffix in INPUT_FILES_SUFFIX)]

    if not files_to_process:
        print("❌ Nenhum arquivo JSON de entrada encontrado na pasta atual. Verifique os nomes dos arquivos.")
    else:
        for input_file in files_to_process:
            # Cria um nome de arquivo de saída correspondente
            output_file = input_file.replace('.json', '_amostra.csv')
            criar_amostra_csv(input_file, output_file)
            print("-" * 50)
    
    print("\n--- Script concluído. ---")
