import json
import re

# --- CONFIGURAÇÕES ---

# Arquivos JSON traduzidos para inspecionar
FILES_TO_INSPECT = [
    "stereoset_intersentence_validation_pt.json",
    "stereoset_intrasentence_validation_pt.json"
]

# Quantidade máxima de exemplos a serem exibidos por arquivo
RESULTS_TO_SHOW = 50

# Mapeamento dos rótulos numéricos para texto em português
LABEL_MAP = {
    0: "Estereótipo",
    1: "Anti-Estereótipo",
    2: "Não Relacionado"
}

# --- FUNÇÕES AUXILIARES ---

def load_repaired_json(file_path):
    """
    Carrega e repara um arquivo que contém múltiplos blocos JSON concatenados.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # Adiciona uma vírgula entre os objetos JSON
        repaired_content = re.sub(r'}\s*{', '},{', content)
        # Adiciona colchetes para formar uma lista JSON válida
        final_json_string = f"[{repaired_content}]"
        data = json.loads(final_json_string)
        return data
    except FileNotFoundError:
        print(f"!!! AVISO: O arquivo '{file_path}' não foi encontrado. Pulando. !!!")
        return None
    except json.JSONDecodeError as e:
        print(f"!!! ERRO: Falha ao carregar o arquivo '{file_path}'. Detalhe: {e} !!!")
        return None

# --- FUNÇÃO PRINCIPAL ---

def inspect_and_evaluate_results():
    """
    Exibe os exemplos traduzidos e solicita uma avaliação manual para cada um.
    """
    # Dicionário para armazenar o placar da avaliação
    evaluation_summary = {"correct": 0, "incorrect": 0, "skipped": 0}

    for file_path in FILES_TO_INSPECT:
        print("=" * 80)
        print(f"VERIFICANDO O ARQUIVO: {file_path}")
        print("=" * 80)
        
        data = load_repaired_json(file_path)
        
        if data is None:
            continue
        
        num_examples_to_show = min(RESULTS_TO_SHOW, len(data))
        
        if num_examples_to_show == 0:
            print("Nenhum exemplo encontrado neste arquivo.")
            continue

        for i in range(num_examples_to_show):
            example = data[i]
            
            try:
                # Extrai as informações do exemplo
                context = example.get('context', 'Contexto não encontrado')
                bias_type = example.get('bias_type', 'N/A')
                sentences_data = example.get('sentences', {})
                labels = sentences_data.get('gold_label', [])
                sents = sentences_data.get('sentence', [])
                
                # Exibe as informações na tela
                print(f"\n--- Exemplo {i + 1}/{num_examples_to_show} ---")
                print(f"Tipo de Viés: {bias_type}")
                print(f"Contexto: {context}")
                
                for label_num, sentence_text in zip(labels, sents):
                    label_text = LABEL_MAP.get(label_num, "Rótulo Desconhecido")
                    print(f'  - {label_text} ({label_num}): "{sentence_text}"')
                
                # --- INÍCIO DA MODIFICAÇÃO: AVALIAÇÃO INTERATIVA ---
                while True:
                    feedback = input("\nA tradução parece correta? [y]sim / [n]não / [s]saltar: ").lower()
                    if feedback in ['y', 'sim']:
                        evaluation_summary["correct"] += 1
                        break
                    elif feedback in ['n', 'nao', 'não']:
                        evaluation_summary["incorrect"] += 1
                        break
                    elif feedback in ['s', 'saltar']:
                        evaluation_summary["skipped"] += 1
                        break
                    else:
                        print("Opção inválida. Por favor, digite 'y', 'n' ou 's'.")
                # --- FIM DA MODIFICAÇÃO ---
            
            except Exception as e:
                print(f"\n--- ERRO ao processar o exemplo {i + 1} ---")
                print(f"Conteúdo do exemplo: {example}")
                print(f"Erro: {e}")

        print("\n" + "=" * 80)
        print(f"Fim da verificação para {file_path}")
        print("=" * 80 + "\n")

    # --- EXIBIÇÃO DO RESUMO FINAL ---
    print("\n" + "#" * 80)
    print("### AVALIAÇÃO MANUAL CONCLUÍDA ###")
    print("#" * 80 + "\n")

    total_evaluated = evaluation_summary["correct"] + evaluation_summary["incorrect"]
    print("Resumo da sua Avaliação:")
    print(f"- Traduções Marcadas como Corretas  : {evaluation_summary['correct']}")
    print(f"- Traduções Marcadas como Incorretas: {evaluation_summary['incorrect']}")
    print(f"- Exemplos Saltados (não avaliados): {evaluation_summary['skipped']}")
    print("-" * 40)
    print(f"Total de Exemplos Efetivamente Avaliados: {total_evaluated}")

    if total_evaluated > 0:
        accuracy = (evaluation_summary["correct"] / total_evaluated) * 100
        print(f"Taxa de Acerto Percebida: {accuracy:.2f}%")
    print("\n" + "#" * 80)


if __name__ == "__main__":
    inspect_and_evaluate_results()
