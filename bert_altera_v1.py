Bash

python eval_discriminative_models.py \
    --pretrained-class "neuralmind/bert-base-portuguese-cased" \
    --tokenizer "BertTokenizer" \
    --intrasentence-model "BertLM" \
    --intersentence-model "BertNextSentence" \
    --input-file "../data/dev_pt.json" \
    --output-file "predictions_bertimbau.json"
Para Avaliar o BERT Multilingual:
O comando é quase idêntico, apenas mudando o nome do modelo e o arquivo de saída.

Bash

python eval_discriminative_models.py \
    --pretrained-class "bert-base-multilingual-cased" \
    --tokenizer "BertTokenizer" \
    --intrasentence-model "BertLM" \
    --intersentence-model "BertNextSentence" \
    --input-file "../data/dev_pt.json" \
    --output-file "predictions_mbert.json"
//////////////////////////////////////////////////////////////

def __create_intrasentence_examples__(self, examples):
        created_examples = []
        # Loop externo: itera sobre cada CLUSTER de sentenças
        for example in examples:
            sentences = []
            
            # Loop interno: itera sobre cada SENTENÇA dentro do cluster
            for sentence in example['sentences']:
                # Cria o objeto Sentença básico
                labels = [Label(**label) for label in sentence['labels']]
                sentence_obj = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])

                # --- INÍCIO DA LÓGICA CORRIGIDA COM DIFF ---
                # Limpa e tokeniza a frase de contexto, removendo a palavra BLANK
                context_tokens = [w for w in example['context'].lower().split() if 'blank' not in w]
                
                # Limpa e tokeniza a frase completa (usando o dict 'sentence' do loop interno)
                sentence_tokens = sentence['sentence'].lower().split()

                # Usa o SequenceMatcher para encontrar a maior sequência de palavras em comum
                matcher = SequenceMatcher(None, context_tokens, sentence_tokens)
                
                # A "palavra-alvo" é composta por todas as palavras da sentença que NÃO fazem parte do bloco comum
                diff_words = []
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != 'equal': # Captura inserções ('insert') e substituições ('replace')
                        # Pega as palavras com a capitalização original da sentença correta
                        diff_words.extend(sentence['sentence'].split()[j1:j2]) 

                if diff_words:
                    template_word = " ".join(diff_words)
                    sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                    sentences.append(sentence_obj)
                else:
                    # Se ainda assim falhar, imprime o aviso (agora muito mais raro)
                    print(f"AVISO: Não foi possível encontrar a diferença para o ID {sentence['id']}.")
                    print(f"  Contexto: {example['context']}")
                    print(f"  Sentença: {sentence['sentence']}")
                # --- FIM DA LÓGICA CORRIGIDA ---

            # Cria o objeto Exemplo com a lista de sentenças processadas
            created_example = IntrasentenceExample(
                example['id'], example['bias_type'], 
                example['target'], example['context'], sentences) 
            created_examples.append(created_example)
            
        return created_examples

///////////////////////////////////////////////////

        Com certeza\! Analisando o seu traceback, o problema fica bem claro. A solução é **modificar o arquivo `dataloader.py`** para torná-lo mais robusto à tradução.

-----

### Diagnóstico do Erro 💡

O erro `IndexError: list index out of range` acontece na linha:
`template_word = sentence['sentence'].split(" ")[word_idx]`

O problema é um pressuposto frágil no código original do Stereoset:

1.  O script primeiro encontra o índice da palavra `"BLANK"` na frase de contexto (ex: "Meu amigo é um BLANK."). Vamos dizer que o índice (`word_idx`) seja `4`.
2.  Em seguida, ele assume que a palavra-alvo (ex: "cientista") estará **exatamente no mesmo índice** na frase preenchida (ex: "Meu amigo é cientista.").
3.  **A tradução quebra isso.** Em português, a frase "Meu amigo é cientista" tem apenas 4 palavras (índices 0 a 3). Quando o código tenta acessar o índice `4`, a lista é menor que o esperado, causando o erro `IndexError`.

Tentar consertar isso no script de tradução é inviável. A solução correta é tornar o `dataloader.py` mais inteligente.

-----

### A Solução: Modificar o `dataloader.py`

Vamos alterar a lógica para que, em vez de depender de um índice, ele encontre a palavra-alvo descobrindo qual palavra é a **diferença** entre a frase de contexto e a frase preenchida.

#### Passo 1: Abra o arquivo `dataloader.py`

Navegue até o arquivo `/home/sagemaker-user/stereoset/code/dataloader.py`.

#### Passo 2: Localize a função `__create_intrasentence_examples__`

Dentro da classe `StereoSet`, encontre esta função.

#### Passo 3: Substitua a lógica de busca por índice

Você substituirá um bloco de código dentro do loop `for sentence in example['sentences']:` por uma versão mais robusta.

**SUBSTITUA ESTE BLOCO DE CÓDIGO ORIGINAL:**

```python
                word_idx = None
                for idx, word in enumerate(example['context'].split(" ")):
                    if "BLANK" in word: 
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                template_word = sentence['sentence'].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                sentences.append(sentence_obj)
```

**POR ESTE NOVO BLOCO DE CÓDIGO ROBUSTO:**

```python
                # --- INÍCIO DA NOVA LÓGICA ---
                # Remove a pontuação e divide as frases em conjuntos de palavras em minúsculas
                context_words = set(example['context'].replace("BLANK", "").lower().translate(str.maketrans('', '', string.punctuation)).split())
                sentence_words = set(sentence['sentence'].lower().translate(str.maketrans('', '', string.punctuation)).split())

                # A palavra-alvo é a que está no conjunto da sentença, mas não no do contexto
                difference = sentence_words.difference(context_words)

                # Verifica se encontrou exatamente uma palavra de diferença
                if len(difference) == 1:
                    template_word = difference.pop()
                    sentence_obj.template_word = template_word
                    sentences.append(sentence_obj)
                else:
                    # Se a lógica falhar para um exemplo, imprime um aviso em vez de quebrar a execução
                    print(f"AVISO: Não foi possível encontrar uma única palavra de diferença para o ID {sentence['id']}.")
                    print(f"  Contexto: {example['context']}")
                    print(f"  Sentença: {sentence['sentence']}")
                    # Isso permite que o script continue com os outros exemplos
```

-----

### Próximos Passos ✅

1.  **Aplique a alteração** no seu arquivo `dataloader.py`.
2.  **Não é necessário** gerar novamente o arquivo `dev_pt.json`. O problema estava na leitura do arquivo, não no arquivo em si.
3.  **Execute o script `eval_discriminative_models.py` novamente.**

O erro `IndexError` será resolvido, pois o programa não depende mais da frágil suposição de que a contagem de palavras permanece a mesma após a tradução.
        
///////////////////////////////////////////////////
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import re
import json
from tqdm import tqdm

# --- 1. CONFIGURAÇÕES ---

MODEL_NAME = "facebook/nllb-200-1.3B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"
BATCH_SIZE = 8

GOLD_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
INNER_LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated', 3: 'related'}

# --- 2. FUNÇÃO AUXILIAR ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_e_recriar_estrutura_corretamente():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- ETAPA DE EXTRAÇÃO (SEM MUDANÇAS) ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            if 'context' in example and example['context']:
                sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- ETAPA DE TRADUÇÃO (SEM MUDANÇAS) ---
    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    for i in tqdm(range(0, len(sentences_to_translate), BATCH_SIZE), desc="Traduzindo Lotes"):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=128)
        batch_translated_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        batch_sanitized = [sanitize_text(text) for text in batch_translated_raw]
        translated_sentences.extend(batch_sanitized)
    print("Tradução finalizada.")

    # --- ETAPA DE RECONSTRUÇÃO (LÓGICA FINAL) ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    # MUDANÇA PRINCIPAL: Padrão Regex para encontrar e padronizar "BLANK"
    # \b garante que estamos pegando a palavra inteira. Cobre "branco", "branca", "blanco", "em branco", etc.
    BLANK_PATTERN = re.compile(r'\b(branco|branca|blanco|blanca|em branco|lacuna)\b', re.IGNORECASE)

    reconstructed_data = {}
    for config in CONFIGS:
        original_dataset = datasets_dict[config]
        new_examples_list = []
        for original_example in tqdm(original_dataset, desc=f"Reconstruindo {config}"):
            new_example = {
                "id": original_example['id'],
                "bias_type": original_example['bias_type'],
                "target": original_example['target'],
                "sentences": []
            }
            
            # Garante que o contexto seja preservado para ambos os tipos
            if 'context' in original_example and original_example['context']:
                translated_context = next(translated_iter)
                # Se for um exemplo intrasentence, padroniza a tradução de "BLANK" de volta para o original.
                if config == 'intrasentence':
                    translated_context = BLANK_PATTERN.sub("BLANK", translated_context)
                new_example["context"] = translated_context
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({
                        "human_id": human_ids[j],
                        "label": INNER_LABEL_MAP[inner_int_labels[j]]
                    })

                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": next(translated_iter),
                    "labels": recreated_labels,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1",
        "data": {
            "intrasentence": reconstructed_data.get('intrasentence', []),
            "intersentence": reconstructed_data.get('intersentence', [])
        }
    }
    
    output_path = "dev_pt.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n✅ Sucesso! O arquivo de saída agora é 100% compatível com a estrutura original do Stereoset.")


if __name__ == "__main__":
    traduzir_e_recriar_estrutura_corretamente()
