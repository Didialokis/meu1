import json
from datasets import load_dataset
import itertools

# --- 1. CONFIGURAÇÕES ---

# Nomes dos arquivos .jsonl gerados pelo seu script de tradução
FILENAMES_TO_CHECK = [
    "stereoset_intersentence_validation_pt_tower.jsonl",
    "stereoset_intrasentence_validation_pt_tower.jsonl",
]

# Quantidade de exemplos a serem exibidos de cada arquivo
NUM_EXAMPLES_TO_SHOW = 50

# --- 2. SCRIPT DE VERIFICAÇÃO ---

def display_translation_samples():
    """
    Carrega os datasets original e traduzido e exibe uma comparação
    lado a lado para verificação da qualidade.
    """
    for filename in FILENAMES_TO_CHECK:
        print(f"\n{'='*80}")
        print(f"VERIFICANDO AMOSTRAS DO ARQUIVO: {filename}")
        print(f"{'='*80}\n")

        try:
            # Determina a configuração ('intersentence' ou 'intrasentence') a partir do nome do arquivo
            config = filename.split('_')[1]

            # Carrega o dataset original correspondente do Hugging Face
            print(f"Carregando dataset original '{config}' do Hugging Face para comparação...")
            original_dataset = load_dataset("McGill-NLP/stereoset", config, split="validation")
            
            # Abre o arquivo .jsonl traduzido
            with open(filename, 'r', encoding='utf-8') as f:
                # Usamos islice para pegar apenas as primeiras N linhas sem carregar o arquivo inteiro
                translated_lines = list(itertools.islice(f, NUM_EXAMPLES_TO_SHOW))

            if not translated_lines:
                print("Arquivo de tradução está vazio ou não foi encontrado.")
                continue

            # Itera sobre os primeiros N exemplos de ambos os datasets
            for i, (original_example, translated_line) in enumerate(zip(original_dataset, translated_lines)):
                translated_example = json.loads(translated_line)
                
                # Extrai os dados para comparação
                original_context = original_example['context']
                translated_context = translated_example['context']
                
                original_sents_data = original_example['sentences']
                translated_sents_data = translated_example['sentences']

                print(f"--- Amostra {i+1}/{NUM_EXAMPLES_TO_SHOW} (ID: {original_example['id']}) ---")
                
                # Exibe o contexto
                print(f"Contexto Original (EN):    {original_context}")
                print(f"Contexto Traduzido (PT):   {translated_context}\n")

                try:
                    # Mapeamento de labels: 0=stereotype, 1=anti-stereotype, 2=unrelated
                    labels = original_sents_data['gold_label']
                    
                    # Encontra os índices para exibir na ordem correta
                    idx_stereotype = labels.index(0)
                    idx_anti_stereotype = labels.index(1)
                    idx_unrelated = labels.index(2)

                    # Exibe as sentenças na ordem lógica para fácil comparação
                    print("[Estereótipo]")
                    print(f"  Original (EN):    {original_sents_data['sentence'][idx_stereotype]}")
                    print(f"  Traduzido (PT):   {translated_sents_data['sentence'][idx_stereotype]}\n")

                    print("[Anti-Estereótipo]")
                    print(f"  Original (EN):    {original_sents_data['sentence'][idx_anti_stereotype]}")
                    print(f"  Traduzido (PT):   {translated_sents_data['sentence'][idx_anti_stereotype]}\n")

                    print("[Não Relacionado]")
                    print(f"  Original (EN):    {original_sents_data['sentence'][idx_unrelated]}")
                    print(f"  Traduzido (PT):   {translated_sents_data['sentence'][idx_unrelated]}")

                except (KeyError, ValueError):
                    print("  ERRO: Estrutura de sentenças inesperada neste exemplo.")
                
                print(f"{'-'*80}\n")

        except FileNotFoundError:
            print(f"ERRO: Arquivo '{filename}' não encontrado. Verifique o nome e o caminho do arquivo.")
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao processar o arquivo '{filename}': {e}")


# --- 3. EXECUÇÃO ---

if __name__ == "__main__":
    display_translation_samples()
