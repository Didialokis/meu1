import json
import re
import sys
import tty
import termios
import csv
from datasets import load_dataset

# --- CONFIGURAÇÕES ---
FILES_TO_INSPECT = [
    "stereoset_intersentence_validation_pt_nllb_final.json",
    "stereoset_intrasentence_validation_pt_nllb_final.json"
]
EXAMPLES_PER_FILE = 50
CSV_OUTPUT_FILE = 'avaliacao_traducoes.csv' # Nome do arquivo de saída
LABEL_MAP = {
    1: "Estereótipo",
    0: "Anti-Estereótipo",
    2: "Não Relacionado"
}
# ---------------------

def getch():
    """Captura um único caractere do terminal sem precisar pressionar Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def load_repaired_json(file_path):
    """Carrega um arquivo JSON que pode estar mal formatado (com múltiplos arrays)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Corrige o JSON que tem '][' entre os objetos
        repaired_content = re.sub(r']\s*\[', ',', content)
        # Envolve tudo em um único array
        final_json_string = f"[{repaired_content}]"
        
        data = json.loads(final_json_string)
        return data
    except FileNotFoundError:
        print(f"||! AVISO: O arquivo '{file_path}' não foi encontrado. Pulando. !!|")
        return None
    except json.JSONDecodeError as e:
        print(f"I!! ERRO: Falha ao carregar o arquivo '{file_path}'. Detalhe: {e} !!!")
        return None

def interactive_evaluation():
    """Função principal para avaliação interativa com salvamento em CSV."""
    stats = {"correto": 0, "incorreto": 0, "pulado": 0}
    
    # Define o cabeçalho para o arquivo CSV
    csv_header = [
        'arquivo_origem', 'id_exemplo', 'contexto_en', 
        'sentenca_en', 'label_en', 'sentenca_pt', 'avaliacao'
    ]

    try:
        # Abre o arquivo CSV para escrita
        with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header) # Escreve o cabeçalho

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

                    print("\n" + "-" * 40 + f" Exemplo {i + 1}/{num_examples} " + "-" * 40)
                    print(f"\n[ CONTEXTO ORIGINAL (EN) ]\n{example_en['context']}")
                    print("-" * 30)

                    # Itera sobre cada sentença dentro do exemplo
                    for sent_en, label_en, sent_pt in zip(
                        example_en['sentences']['sentence'], 
                        example_en['sentences']['gold_label'], 
                        example_pt['sentences']['sentence']
                    ):
                        print(f"  [EN] - {LABEL_MAP[label_en]}: \"{sent_en}\"")
                        print(f"  [PT] - Tradução: \"{sent_pt}\"\n")
                        
                        print("A tradução parece correta? Pressione a tecla:")
                        print("[s] Sim | [n] Não | [p] Pular | [q] Sair e Salvar")

                        while True:
                            char = getch().lower()
                            avaliacao = None
                            
                            if char == 's':
                                stats["correto"] += 1
                                avaliacao = "correto"
                                print("-> Marcado como CORRETO.\n")
                            elif char == 'n':
                                stats["incorreto"] += 1
                                avaliacao = "incorreto"
                                print("-> Marcado como INCORRETO.\n")
                            elif char == 'p':
                                stats["pulado"] += 1
                                avaliacao = "pulado"
                                print("-> Sentença PULADA.\n")
                            elif char == 'q':
                                raise KeyboardInterrupt # Usa a interrupção para sair do loop

                            if avaliacao:
                                # Prepara a linha de dados para o CSV
                                data_row = [
                                    file_path,
                                    example_en['id'],
                                    example_en['context'],
                                    sent_en,
                                    LABEL_MAP[label_en],
                                    sent_pt,
                                    avaliacao
                                ]
                                csv_writer.writerow(data_row)
                                csvfile.flush() # Garante que a linha seja salva imediatamente
                                break
                        print("-" * 30)


    except KeyboardInterrupt:
        print("\n\nSaindo... Avaliações salvas em '{}'.".format(CSV_OUTPUT_FILE))
    
    finally:
        print("\n" + "=" * 80)
        print("RESUMO DA SESSÃO")
        print("=" * 80)
        total = stats["correto"] + stats["incorreto"]
        print(f"Exemplos Corretos: {stats['correto']}")
        print(f"Exemplos Incorretos: {stats['incorreto']}")
        print(f"Exemplos Pulados: {stats['pulado']}")
        if total > 0:
            accuracy = (stats["correto"] / total) * 100
            print(f"\nTaxa de Aprovação (Correto / (Correto + Incorreto)): {accuracy:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    interactive_evaluation()
