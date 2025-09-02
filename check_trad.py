import json
from datasets import load_dataset

# Nomes dos arquivos de saída gerados pelo script de tradução
ARQUIVOS_TRADUZIDOS = [
    "stereoset_intersentence_validation_pt.json",
    "stereoset_intrasentence_validation_pt.json",
]

def checar_integridade():
    print("--- INICIANDO VERIFICAÇÃO ESTRUTURAL E QUANTITATIVA ---")
    
    for arquivo_pt in ARQUIVOS_TRADUZIDOS:
        print(f"\nVerificando arquivo: {arquivo_pt}...")
        
        # Extrai a configuração ('intersentence' ou 'intrasentence') do nome do arquivo
        try:
            config = arquivo_pt.split('_')[1]
        except IndexError:
            print(f"  ERRO: Não foi possível determinar a configuração pelo nome do arquivo '{arquivo_pt}'.")
            continue

        # 1. Carrega o dataset original do Hugging Face
        try:
            dataset_original = load_dataset("McGill-NLP/stereoset", config, split="validation")
            print(f"  [OK] Dataset original '{config}' carregado com sucesso.")
        except Exception as e:
            print(f"  ERRO ao carregar dataset original '{config}': {e}")
            continue
            
        # 2. Carrega o dataset traduzido do arquivo JSON
        try:
            with open(arquivo_pt, 'r', encoding='utf-8') as f:
                dataset_traduzido = json.load(f)
            print(f"  [OK] Arquivo traduzido é um JSON válido.")
        except json.JSONDecodeError:
            print(f"  ERRO: O arquivo '{arquivo_pt}' não é um arquivo JSON válido.")
            continue
        except FileNotFoundError:
            print(f"  ERRO: Arquivo '{arquivo_pt}' não encontrado.")
            continue

        # 3. Compara a quantidade de exemplos
        num_original = len(dataset_original)
        num_traduzido = len(dataset_traduzido)

        if num_original == num_traduzido:
            print(f"  [OK] Número de exemplos corresponde: {num_original}")
        else:
            print(f"  ERRO: Divergência no número de exemplos! Original: {num_original}, Traduzido: {num_traduzido}")
            continue # Pula para o próximo arquivo se a contagem principal estiver errada

        # 4. Checa a estrutura de um exemplo aleatório
        exemplo_original = dataset_original[0]
        exemplo_traduzido = dataset_traduzido[0]
        
        num_sentencas_orig = len(exemplo_original['sentences']['sentence'])
        num_sentencas_trad = len(exemplo_traduzido['sentences']['sentence'])

        if num_sentencas_orig == num_sentencas_trad:
             print(f"  [OK] Número de sentenças no primeiro exemplo corresponde: {num_sentencas_orig}")
        else:
            print(f"  ERRO: Divergência no número de sentenças do primeiro exemplo! Original: {num_sentencas_orig}, Traduzido: {num_sentencas_trad}")

    print("\n--- VERIFICAÇÃO AUTOMÁTICA CONCLUÍDA ---")


if __name__ == "__main__":
    checar_integridade()
