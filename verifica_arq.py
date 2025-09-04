import json
import re

FILES_TO_INSPECT = [
    "stereoset_intersentence_validation_pt.json",
    "stereoset_intrasentence_validation_pt.json"
]

RESULTS_TO_SHOW = 50

LABEL_MAP = {
    0: "Estereótipo",
    1: "Anti-Estereótipo",
    2: "Não Relacionado"
}

def load_repaired_json(file_path):

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

def inspect_translated_results():

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
                context = example.get('context', 'Contexto não encontrado')
                bias_type = example.get('bias_type', 'N/A')
                sentences_data = example.get('sentences', {})
                labels = sentences_data.get('gold_label', [])
                sents = sentences_data.get('sentence', [])
                
                print(f"\n--- Exemplo {i + 1}/{num_examples_to_show} ---")
                print(f"Tipo de Viés: {bias_type}")
                print(f"Contexto: {context}")
                
                for label_num, sentence_text in zip(labels, sents):
                    label_text = LABEL_MAP.get(label_num, "Rótulo Desconhecido")
                    print(f'  - {label_text} ({label_num}): "{sentence_text}"')
            
            except Exception as e:
                print(f"\n--- ERRO ao processar o exemplo {i + 1} ---")
                print(f"Conteúdo do exemplo: {example}")
                print(f"Erro: {e}")

        print("\n" + "=" * 80)
        print(f"Fim da verificação para {file_path}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    inspect_translated_results()
