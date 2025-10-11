import json

# Altere para o caminho do seu arquivo traduzido
input_filename = 'dev_pt.json' 
# Nome do novo arquivo que será salvo com os índices corrigidos
output_filename = 'dev_pt_corrigido.json' 

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Erro: O arquivo '{input_filename}' não foi encontrado.")
    exit()

total_examples = 0
corrected_count = 0
errors = []

# Itera sobre os dados do Stereoset (intrasentence)
for bias_type in data.get('intrasentence', []):
    for example in bias_type.get('examples', []):
        for sentence in example.get('sentences', []):
            total_examples += 1
            words = sentence['sentence'].split(' ')
            target_word = sentence['target']
            
            try:
                # Tenta encontrar o índice correto da palavra-alvo
                new_idx = words.index(target_word)
                if sentence['word_idx'] != new_idx:
                    sentence['word_idx'] = new_idx
                    corrected_count += 1
            except ValueError:
                # Ocorre se a palavra-alvo não for encontrada na frase após o split
                # Isso pode acontecer por causa de pontuação, ex: "negro" vs "negro."
                # Tentamos encontrar uma correspondência parcial
                found = False
                for i, word in enumerate(words):
                    if target_word in word:
                        sentence['word_idx'] = i
                        corrected_count += 1
                        found = True
                        break
                if not found:
                    errors.append({
                        "id": sentence.get('id', 'N/A'),
                        "sentence": sentence['sentence'],
                        "target": target_word
                    })

print(f"Total de exemplos verificados: {total_examples}")
print(f"Índices corrigidos: {corrected_count}")

if errors:
    print(f"\nAVISO: Não foi possível encontrar o alvo em {len(errors)} exemplos:")
    for error in errors[:5]: # Mostra os 5 primeiros erros
        print(f"  - ID: {error['id']}, Frase: '{error['sentence']}', Alvo: '{error['target']}'")

# Salva o novo arquivo JSON com os índices corrigidos
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nArquivo corrigido salvo como '{output_filename}'. Use este arquivo na sua avaliação.")
