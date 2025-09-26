import json
import re
import sys
import tty
import termios
import csv # Biblioteca para manipulação de CSV
from datasets import load_dataset

# --- CONFIGURAÇÕES ---
# Arquivos JSON gerados pelo script de tradução
FILES_TO_INSPECT = [
    "stereoset_intersentence_validation_pt_nllb_final.json",
    "stereoset_intrasentence_validation_pt_nllb_final.json"
]

# Nome do arquivo de saída
OUTPUT_CSV_FILE = "avaliacao_manual_traducoes.csv"

# Limite de exemplos a serem avaliados por arquivo (coloque None para avaliar todos)
EXAMPLES_PER_FILE = 50 

# Mapeamento de labels numéricos para texto
LABEL_MAP = {0: "Estereótipo", 1: "Anti-Estereótipo", 2: "Não Relacionado"}

# --- FUNÇÕES AUXILIARES ---

def getch():
    """Captura uma única tecla pressionada pelo usuário."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def load_repaired_json(file_path):
    """Carrega o arquivo JSON traduzido."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        repaired_content = re.sub(r'}\s*{', '},{', content)
        final_json_string = f"[{repaired_content}]"
        data = json.loads(final_json_string)
        return data
    except FileNotFoundError:
        print(f"||! AVISO: O arquivo '{file_path}' não foi encontrado. Pulando. !!|")
        return None
    except json.JSONDecodeError as e:
        print(f"||! ERRO: Falha ao carregar o arquivo '{file_path}'. Detalhe: {e} !!|")
        return None

# --- FUNÇÃO PRINCIPAL DE AVALIAÇÃO ---

def interactive_evaluation_to_csv():
    stats = {"Correto": 0, "Incorreto": 0, "Pulado": 0}
    
    # Abre o arquivo CSV para escrita
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        # Define o cabeçalho das colunas
        header = [
            'id_exemplo', 'id_sentenca', 'contexto_original', 'contexto_traduzido', 
            'sentenca_original', 'sentenca_traduzida', 'label', 'avaliacao'
        ]
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header) # Escreve o cabeçalho

        try:
            for file_path in FILES_TO_INSPECT:
                print("=" * 80)
                print(f"INICIANDO AVALIAÇÃO DO ARQUIVO: {file_path}")
                print("=" * 80)

                data_pt = load_repaired_json(file_path)
                if data_pt is None:
                    continue

                config = "intersentence" if "intersentence" in file_path else "intrasentence"
                print(f"Carregando dataset original '{config}' do Hugging Face para comparação...")
                data_en = load_dataset("McGill-NLP/stereoset", config, split="validation")
                
                num_examples = min(EXAMPLES_PER_FILE or len(data_pt), len(data_pt))

                for i in range(num_examples):
                    example_pt = data_pt[i]
                    example_en = data_en[i]
                    
                    # Itera sobre cada sentença dentro do exemplo
                    num_sentences = len(example_en['sentences']['sentence'])
                    for j in range(num_sentences):
                        sent_en = example_en['sentences']['sentence'][j]
                        sent_pt = example_pt['sentences']['sentence'][j]
                        label_num = example_en['sentences']['gold_label'][j]
                        label_text = LABEL_MAP.get(label_num, "Desconhecido")
                        
                        print("\n" + "=" * 40 + f" Exemplo {i+1}/{num_examples} | Sentença {j+1}/{num_sentences} " + "=" * 40)
                        print(f"[ CONTEXTO ORIGINAL (EN) ]\n{example_en['context']}")
                        print(f"\n[ CONTEXTO TRADUZIDO (PT) ]\n{example_pt['context']}")
                        print("-" * 30)
                        print(f"[ SENTENÇA ORIGINAL (EN) - {label_text} ]\n\"{sent_en}\"")
                        print(f"\n[ SENTENÇA TRADUZIDA (PT) ]\n\"{sent_pt}\"")
                        print("-" * 30)
                        print("A tradução da SENTENÇA parece correta? Pressione a tecla:")
                        print("[s] Sim  |  [n] Não  |  [p] Pular  |  [q] Sair e Salvar")

                        while True:
                            char = getch().lower()
                            if char == 's':
                                evaluation = "Correto"
                                stats[evaluation] += 1
                                print("-> Marcado como CORRETO.")
                                break
                            elif char == 'n':
                                evaluation = "Incorreto"
                                stats[evaluation] += 1
                                print("-> Marcado como INCORRETO.")
                                break
                            elif char == 'p':
                                evaluation = "Pulado"
                                stats[evaluation] += 1
                                print("-> Exemplo PULADO.")
                                break
                            elif char == 'q':
                                raise KeyboardInterrupt
                        
                        # Prepara a linha para salvar no CSV
                        row_to_save = [
                            example_en['id'],
                            example_en['sentences']['id'][j],
                            example_en['context'],
                            example_pt['context'],
                            sent_en,
                            sent_pt,
                            label_text,
                            evaluation
                        ]
                        csv_writer.writerow(row_to_save) # Salva a linha no arquivo

        except KeyboardInterrupt:
            print("\n\nSaindo da avaliação...")
        
        finally:
            print("\n" + "=" * 80)
            print(f"AVALIAÇÃO CONCLUÍDA. Resultados salvos em '{OUTPUT_CSV_FILE}'")
            print("=" * 80)
            total = stats["Correto"] + stats["Incorreto"]
            print(f"Exemplos Corretos: {stats['Correto']}")
            print(f"Exemplos Incorretos: {stats['Incorreto']}")
            print(f"Exemplos Pulados: {stats['Pulado']}")
            if total > 0:
                accuracy = (stats["Correto"] / total) * 100
                print(f"\nTaxa de Aprovação (Corretos / (Corretos + Incorretos)): {accuracy:.2f}%")
            print("=" * 80)


if __name__ == "__main__":
    interactive_evaluation_to_csv()
