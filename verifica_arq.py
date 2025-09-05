import json
import re
import sys
import tty
import termios
from datasets import load_dataset

# --- 1. CONFIGURAÇÕES ---

# Arquivos traduzidos em português para inspecionar
FILES_TO_INSPECT = [
    "stereoset_intersentence_validation_pt.json",
    "stereoset_intrasentence_validation_pt.json"
]

# Quantidade de exemplos de cada arquivo que você deseja avaliar
# Mude para um número maior ou para None se quiser avaliar o arquivo inteiro
EXAMPLES_PER_FILE = 50

# Mapeamento de rótulos numéricos para texto
LABEL_MAP = {
    0: "Estereótipo",
    1: "Anti-Estereótipo",
    2: "Não Relacionado"
}

# --- 2. FUNÇÕES AUXILIARES ---

def getch():
    """
    Função para capturar um único caracter do teclado sem precisar de 'Enter'.
    Funciona em sistemas Linux/macOS.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def load_repaired_json(file_path):
    """Carrega e repara o arquivo JSON traduzido."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        repaired_content = re.sub(r'}\s*{', '},{', content)
        final_json_string = f"[{repaired_content}]"
        data = json.loads(final_json_string)
        return data
    except FileNotFoundError:
        print(f"!!! AVISO: O arquivo '{file_path}' não foi encontrado. Pulando. !!!")
        return None
    except json.JSONDecodeError as e:
        print(f"!!! ERRO: Falha ao carregar o arquivo '{file_path}'. Detalhe: {e} !!!")
        return None

# --- 3. FUNÇÃO PRINCIPAL DE AVALIAÇÃO ---

def interactive_evaluation():
    """
    Executa o processo de avaliação interativa lado a lado.
    """
    # Contadores para o resumo final
    stats = {"correto": 0, "incorreto": 0, "pulado": 0}

    try:
        for file_path in FILES_TO_INSPECT:
            print("=" * 80)
            print(f"INICIANDO AVALIAÇÃO DO ARQUIVO: {file_path}")
            print("=" * 80)
            
            # Carrega o arquivo traduzido (português)
            data_pt = load_repaired_json(file_path)
            if data_pt is None:
                continue

            # Carrega o arquivo original (inglês) correspondente do Hugging Face
            config = "intersentence" if "intersentence" in file_path else "intrasentence"
            print(f"Carregando dataset original '{config}' do Hugging Face para comparação...")
            data_en = load_dataset("McGill-NLP/stereoset", config, split="validation")
            
            num_examples = min(EXAMPLES_PER_FILE or len(data_pt), len(data_pt))

            for i in range(num_examples):
                example_pt = data_pt[i]
                example_en = data_en[i]
                
                # Exibe o exemplo lado a lado
                print("\n" + "-" * 40 + f" Exemplo {i + 1}/{num_examples} " + "-" * 40)
                
                # Original em Inglês
                print("\n[ VERSÃO ORIGINAL - INGLÊS ]")
                print(f"Contexto: {example_en['context']}")
                for sent, label in zip(example_en['sentences']['sentence'], example_en['sentences']['gold_label']):
                    print(f"  - {LABEL_MAP[label]}: \"{sent}\"")

                # Tradução em Português
                print("\n[ SUA TRADUÇÃO - PORTUGUÊS ]")
                print(f"Contexto: {example_pt['context']}")
                for sent, label in zip(example_pt['sentences']['sentence'], example_pt['sentences']['gold_label']):
                    print(f"  - {LABEL_MAP[label]}: \"{sent}\"")
                
                # Pergunta interativa
                print("\n" + "-" * 30)
                print("A tradução parece correta e natural? Pressione a tecla:")
                print("[s] Sim  |  [n] Não  |  [p] Pular  |  [q] Sair")
                
                # Captura a resposta sem 'Enter'
                while True:
                    char = getch().lower()
                    if char == 's':
                        stats["correto"] += 1
                        print("-> Marcado como CORRETO.")
                        break
                    elif char == 'n':
                        stats["incorreto"] += 1
                        print("-> Marcado como INCORRETO.")
                        break
                    elif char == 'p':
                        stats["pulado"] += 1
                        print("-> Exemplo PULADO.")
                        break
                    elif char == 'q':
                        # Se apertar 'q', sai e mostra o resumo
                        raise KeyboardInterrupt
                        
    except KeyboardInterrupt:
        # Permite sair com Ctrl+C ou 'q' e ainda ver o resumo
        print("\n\nSaindo da avaliação interativa...")

    finally:
        # Exibe o resumo final
        print("\n" + "=" * 80)
        print("RESUMO DA AVALIAÇÃO MANUAL")
        print("=" * 80)
        total = stats["correto"] + stats["incorreto"]
        print(f"Exemplos Corretos:   {stats['correto']}")
        print(f"Exemplos Incorretos: {stats['incorreto']}")
        print(f"Exemplos Pulados:     {stats['pulado']}")
        if total > 0:
            accuracy = (stats["correto"] / total) * 100
            print(f"\nTaxa de Aprovação (Corretos / (Corretos + Incorretos)): {accuracy:.2f}%")
        print("=" * 80)

# --- 4. EXECUÇÃO ---

if __name__ == "__main__":
    interactive_evaluation()
