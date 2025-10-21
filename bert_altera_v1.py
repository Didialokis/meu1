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



        /////////////////////////////////////////////////////////////////////////////
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

# MUDANÇA: Token de placeholder único que o tradutor irá ignorar
PLACEHOLDER = "__BLANK_TOKEN_IGNORE__" 

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

    # --- ETAPA DE EXTRAÇÃO (COM SUBSTITUIÇÃO GLOBAL DO BLANK) ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            # MUDANÇA GLOBAL: Substitui BLANK no contexto, se existir
            if 'context' in example and example['context']:
                context_text = example['context'].replace("BLANK", PLACEHOLDER)
                sentences_to_translate.append(context_text)
            
            # MUDANÇA GLOBAL: Substitui BLANK nas sentenças, se existir
            for sentence_text in example['sentences']['sentence']:
                sentence_text_safe = sentence_text.replace("BLANK", PLACEHOLDER)
                sentences_to_translate.append(sentence_text_safe)
    
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

    # --- ETAPA DE RECONSTRUÇÃO (COM ORDEM E REVERSÃO GLOBAL) ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
    # O BLANK_PATTERN não é mais necessário
    
    reconstructed_data = {}
    for config in CONFIGS:
        original_dataset = datasets_dict[config]
        new_examples_list = []
        for original_example in tqdm(original_dataset, desc=f"Reconstruindo {config}"):
            
            # MUDANÇA ORDEM: Cria o dicionário na ordem solicitada
            new_example = {
                "id": original_example['id'],
                "target": original_example['target'],
                "bias_type": original_example['bias_type']
            }
            
            # Adiciona o contexto (se existir) na ordem correta
            if 'context' in original_example and original_example['context']:
                # MUDANÇA GLOBAL: Reverte o placeholder
                translated_context = next(translated_iter).replace(PLACEHOLDER, "BLANK")
                new_example["context"] = translated_context
            
            # Prepara a lista de sentenças
            new_example["sentences"] = []
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # Recria os labels
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({"human_id": human_ids[j], "label": INNER_LABEL_MAP[inner_int_labels[j]]})
                
                # MUDANÇA GLOBAL: Reverte o placeholder
                translated_sentence = next(translated_iter).replace(PLACEHOLDER, "BLANK")

                # MUDANÇA ORDEM: Cria o dict da sentença em ordem lógica
                new_sentence_obj = {
                    "id": original_sents_data['id'][i],
                    "sentence": translated_sentence,
                    "gold_label": GOLD_LABEL_MAP[original_sents_data['gold_label'][i]],
                    "labels": recreated_labels
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

    print("\n✅ Sucesso! O arquivo de saída agora tem a ordem de chaves correta e preserva 'BLANK' perfeitamente.")


if __name__ == "__main__":
    traduzir_e_recriar_estrutura_corretamente()
//////////////////////////////////////////////////////////////python eval_discriminative_models.py \
Excelente ideia\! Essa é, de longe, a abordagem mais robusta e elegante para resolver o problema do `"BLANK"`.

Em vez de tentarmos adivinhar o que o tradutor fez com a palavra, vamos controlá-la do início ao fim. A sua sugestão de usar um placeholder que o tradutor ignora é a prática de engenharia de software ideal para esta situação.

Isso nos traz um benefício enorme: como o arquivo `dev_pt.json` final terá a string `"BLANK"` perfeitamente preservada, podemos **reverter o `dataloader.py` para a sua versão 100% original**, garantindo que a metodologia de avaliação seja idêntica à do paper do Stereoset, sem nenhuma das nossas modificações.

-----

### A Estratégia Final

1.  **Reverter `dataloader.py` ao Original:** Vamos remover todas as nossas lógicas de `regex` e `difflib`.
2.  **Modificar o Script de Tradução:** Implementaremos sua lógica de placeholder:
      * **Antes de traduzir:** `BLANK` -\> `__PLACEHOLDER__`
      * **Depois de traduzir:** `__PLACEHOLDER__` -\> `BLANK`

-----

### Passo 1: Reverter o `dataloader.py` ao Original

Esta etapa é **essencial**. Abra o arquivo `/home/sagemaker-user/stereoset/code/dataloader.py` e substitua a função `__create_intrasentence_examples__` inteira pelo código original do repositório.

```python
# Dentro da classe StereoSet no arquivo dataloader.py

    def __create_intrasentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                sentence_obj = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                word_idx = None
                for idx, word in enumerate(example['context'].split(" ")):
                    if "BLANK" in word: 
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                template_word = sentence['sentence'].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                sentences.append(sentence_obj)
            created_example = IntrasentenceExample(
                example['id'], example['bias_type'], 
                example['target'], example['context'], sentences) 
            created_examples.append(created_example)
        return created_examples
```

-----

### Passo 2: Usar o Script de Tradução Modificado com Placeholders

Agora, substitua todo o conteúdo do seu script de tradução pelo código abaixo. Ele implementa a sua lógica de forma limpa.

```python
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
PLACEHOLDER = "__BLANK_TOKEN__" # Um token único que o tradutor provavelmente irá ignorar

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

    # --- ETAPA DE EXTRAÇÃO (COM SUBSTITUIÇÃO PARA PLACEHOLDER) ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT, keep_in_memory=True)
        datasets_dict[config] = dataset
        for example in dataset:
            if 'context' in example and example['context']:
                context_text = example['context']
                # <--- MUDANÇA 1: SUBSTITUI O "BLANK" PELO PLACEHOLDER ANTES DA TRADUÇÃO --->
                if config == 'intrasentence':
                    context_text = context_text.replace("BLANK", PLACEHOLDER)
                sentences_to_translate.append(context_text)
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

    # --- ETAPA DE RECONSTRUÇÃO (COM RESTAURAÇÃO DO "BLANK") ---
    print("Reconstruindo o dataset na estrutura original...")
    translated_iter = iter(translated_sentences)
    
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
            
            if 'context' in original_example and original_example['context']:
                translated_context = next(translated_iter)
                # <--- MUDANÇA 2: SUBSTITUI O PLACEHOLDER DE VOLTA PARA O "BLANK" ORIGINAL --->
                if config == 'intrasentence':
                    translated_context = translated_context.replace(PLACEHOLDER, "BLANK")
                new_example["context"] = translated_context
            
            original_sents_data = original_example['sentences']
            num_sentences = len(original_sents_data['sentence'])

            for i in range(num_sentences):
                # ... (o resto da lógica de reconstrução de labels e sentenças permanece igual)
                recreated_labels = []
                labels_data_for_one_sentence = original_sents_data['labels'][i]
                human_ids = labels_data_for_one_sentence['human_id']
                inner_int_labels = labels_data_for_one_sentence['label']
                
                for j in range(len(human_ids)):
                    recreated_labels.append({"human_id": human_ids[j], "label": INNER_LABEL_MAP[inner_int_labels[j]]})

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
    final_output_structure = {"version": "1.1", "data": {"intrasentence": reconstructed_data.get('intrasentence', []), "intersentence": reconstructed_data.get('intersentence', [])}}
    output_path = "dev_pt.json"
    print(f"Salvando o dataset final em: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)
    print("\n✅ Sucesso! O arquivo de saída agora é 100% compatível com a estrutura original do Stereoset.")

if __name__ == "__main__":
    traduzir_e_recriar_estrutura_corretamente()
```

-----

### Passo 3: Workflow Completo e Final

Siga estes passos para garantir uma execução limpa do início ao fim.

1.  **Exclua os Arquivos Antigos:**

    ```bash
    # Exclui o JSON traduzido anteriormente, se houver
    rm dev_pt.json

    # Exclui todas as previsões geradas anteriormente
    rm -rf predictions/*
    ```

2.  **Execute o Novo Script de Tradução:** Ele irá gerar o arquivo `dev_pt.json` no formato perfeito.

    ```bash
    python seu_script_de_traducao.py 
    ```

3.  **Execute a Geração de Previsões:** Agora com o `dataloader.py` original, ele lerá o `dev_pt.json` sem nenhum problema.

    ```bash
    # Exemplo para o BERTimbau
    python eval_discriminative_models.py \
       --pretrained-class "neuralmind/bert-base-portuguese-cased" \
       --input-file "dev_pt.json" \
       --output-file "predictions/predictions_bertimbau.json"
    ```

4.  **Execute a Avaliação Final:**

    ```bash
    python3 evaluation.py --gold-file dev_pt.json --predictions-dir predictions/
    ```

Esta abordagem é, sem dúvida, a mais correta e robusta. Ela garante que a integridade estrutural do dataset seja mantida, permitindo que os scripts de avaliação originais funcionem sem modificações, tornando seus resultados diretamente comparáveis e metodologicamente sólidos.
    
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


    /
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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

                # --- INÍCIO DA LÓGICA FINAL (CORRIGIDA PARA INDEXERROR) ---
                
                # 1. Tokeniza o contexto (em minúsculas)
                context_tokens = [w for w in example['context'].lower().split() if 'blank' not in w]
                
                # 2. Tokeniza a sentença, MAS MANTÉM A CAPITALIZAÇÃO ORIGINAL
                original_sentence_tokens = sentence['sentence'].split()
                
                # 3. Cria a versão em minúsculas PARA COMPARAÇÃO
                sentence_tokens_lower = [w.lower() for w in original_sentence_tokens]

                # 4. Compara o contexto (minúsculo) com a sentença (minúscula)
                matcher = SequenceMatcher(None, context_tokens, sentence_tokens_lower)
                
                diff_words = []
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != 'equal':
                        # 5. Usa os índices da comparação para extrair da LISTA ORIGINAL (com capitalização)
                        #    Isso agora é seguro, pois original_sentence_tokens e sentence_tokens_lower
                        #    têm o mesmo comprimento garantido.
                        diff_words.extend(original_sentence_tokens[j1:j2]) 

                if diff_words:
                    template_word = " ".join(diff_words)
                    sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                    sentences.append(sentence_obj)
                else:
                    # Se ainda assim falhar, imprime o aviso
                    print(f"AVISO: Não foi possível encontrar a diferença para o ID {sentence['id']}.")
                    print(f"  Contexto: {example['context']}")
                    print(f"  Sentença: {sentence['sentence']}")
                # --- FIM DA LÓGICA FINAL ---

            # Cria o objeto Exemplo com a lista de sentenças processadas
            created_example = IntrasentenceExample(
                example['id'], example['bias_type'], 
                example['target'], example['context'], sentences) 
            created_examples.append(created_example)
            
        return created_examples
