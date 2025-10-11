import json

# Altere para o caminho do seu arquivo traduzido
input_filename = 'dev_pt.json' 
# Nome do novo arquivo que será salvo com os índices corrigidosClaro, entendi perfeitamente. Esse novo erro, `KeyError: 'context'`, confirma que o problema está em um bug no arquivo `dataloader.py` original do Stereoset.

A boa notícia é que o seu script de tradução e o seu arquivo `dev_pt.json` agora estão **corretos**. O problema não é mais com seus dados, mas sim com a forma como o script de avaliação os lê.

-----

### Análise do Erro: Por que isso acontece?

O erro `KeyError: 'context'` acontece dentro da função `_create_intrasentence_examples`. Como o nome sugere, essa função foi feita para processar **apenas** os exemplos do tipo "intrasentence".

O problema é que, por engano, o código dentro dessa função tenta acessar a chave `'context'` (`example['context']`), que **só existe** nos exemplos do tipo "**inter**sentence". É um bug no script do repositório.

A correção é simples: precisamos editar o `dataloader.py` para que ele procure as palavras na chave correta, que é `sentence['sentence']`.

-----

### 💡 A Solução: Corrigir o `dataloader.py`

Você precisa fazer uma pequena alteração no arquivo `stereoset/code/dataloader.py`.

1.  **Abra o arquivo:** `stereoset/code/dataloader.py`.
2.  **Encontre a função:** `_create_intrasentence_examples`.
3.  **Localize e substitua o bloco de código problemático.**

**Procure por este bloco de código (por volta da linha 135):**

```python
# CÓDIGO ORIGINAL (COM ERRO)
for sentence in example['sentences']:
    word_idx = -1
    for idx, word in enumerate(example['context'].split(" ")):
        if "BLANK" in word:
            word_idx = idx
```

**Substitua-o por este bloco corrigido:**

```python
# CÓDIGO CORRIGIDO
for sentence in example['sentences']:
    word_idx = -1
    # A correção é usar sentence['sentence'] em vez de example['context']
    words = sentence['sentence'].split(" ")
    try:
        word_idx = words.index("BLANK")
    except ValueError:
        # Se 'BLANK' não for encontrado, pulamos esta sentença.
        # Isso evita que o programa quebre se houver um erro de formatação.
        continue
```

#### Resumo da Mudança:

  * **Linha incorreta:** `for idx, word in enumerate(example['context'].split(" ")):`
  * **Lógica correta:** A nova lógica divide a frase correta (`sentence['sentence']`) e encontra o índice da palavra `"BLANK"` de forma segura usando `words.index("BLANK")`.

Depois de fazer essa alteração e salvar o arquivo `dataloader.py`, seu script de avaliação `eval_discriminative_models.py` finalmente conseguirá processar ambos os tipos de exemplos do seu arquivo `dev_pt.json` sem erros.

Agora você pode executar o comando de avaliação completo, **sem** a flag `--skip-intrasentence`, e tudo deve funcionar como esperado. 👍
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
