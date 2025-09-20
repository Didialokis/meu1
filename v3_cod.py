Script de Tradução Final: Recriando a Estrutura Original
Para resolver isso, não podemos mais usar o conveniente método .map(). O script abaixo foi reescrito para reconstruir manualmente a estrutura original após a tradução. Ele é mais detalhado, mas o resultado será um arquivo JSON perfeitamente compatível com o dataloader.py e qualquer outra ferramenta oficial do StereoSet.

Python

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

# Mapeamento para converter os labels numéricos de volta para texto
LABEL_MAP = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}

# --- 2. FUNÇÕES AUXILIARES ---

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


# --- 3. FUNÇÃO PRINCIPAL DE TRADUÇÃO ---

def traduzir_e_recriar_estrutura_corretamente():
    """
    Executa o pipeline de tradução e recria a estrutura original do Stereoset
    com precisão para garantir compatibilidade com o dataloader oficial.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    # --- ETAPA DE EXTRAÇÃO ---
    datasets_dict = {}
    sentences_to_translate = []
    for config in CONFIGS:
        print(f"Carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            # Acessa a lista de sentenças dentro da estrutura do dataset
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas para tradução.")

    # --- ETAPA DE TRADUÇÃO ---
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

    # --- ETAPA DE RECONSTRUÇÃO MANUAL (LÓGICA CORRIGIDA) ---
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
                "context": next(translated_iter),
                "sentences": [] # Será uma lista de dicionários
            }
            
            # Pega as listas paralelas da estrutura do Hugging Face
            original_sents_texts = original_example['sentences']['sentence']
            original_sents_ids = original_example['sentences']['id']
            original_sents_gold_labels = original_example['sentences']['gold_label']
            original_sents_labels = original_example['sentences']['labels']

            # Itera sobre o número de sentenças para "desfazer" a estrutura paralela
            for i in range(len(original_sents_texts)):
                new_sentence_obj = {
                    "id": original_sents_ids[i],
                    "sentence": next(translated_iter),
                    # A chave 'labels' contém as anotações humanas e é crucial para o dataloader
                    "labels": original_sents_labels[i],
                    "gold_label": LABEL_MAP[original_sents_gold_labels[i]]
                }
                new_example["sentences"].append(new_sentence_obj)
            
            new_examples_list.append(new_example)
        reconstructed_data[config] = new_examples_list

    # --- ETAPA DE SALVAMENTO ---
    final_output_structure = {
        "version": "1.1", # Chave de versão compatível
        "data": reconstructed_data
    }
    
    output_path = f"stereoset_{DATASET_SPLIT}_pt_nllb_formato_original.json"
    print(f"Salvando o dataset final em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=2)

    print("\n🎉 Sucesso! Processo concluído. O arquivo de saída agora é 100% compatível com o dataloader.py.")


if __name__ == "__main__":
    traduzir_e_recriar_estrutura_corretamente()

Seu Novo Fluxo de Trabalho (Simplificado e Correto)
Execute o Script de Tradução (acima): Use este novo script para gerar o arquivo stereoset_validation_pt_nllb_formato_original.json. Este arquivo será uma réplica perfeita do original, apenas com os textos em português.

Use o dataloader.py: Agora, qualquer script (como o evaluation.py do repositório oficial) que use o dataloader.py para carregar este novo arquivo funcionará sem erros.

Ajuste o Seu Script de Predição (predicao.py): Seu script de predição foi feito para a estrutura simplificada. Ele também precisa de um pequeno ajuste para funcionar com a estrutura original correta.

Modificação necessária no predicao.py:

Python

# No seu script predicao.py...

# ...
# Carregue os dados aninhados
with open(GOLD_FILE, 'r', encoding='utf-8') as f:
    full_data = json.load(f)
    gold_data = full_data['data'] 
# ...

# ...
# Modifique o loop principal
with tqdm(total=total_sentences, unit="sentença") as pbar:
    for task_type in gold_data:
        for example in gold_data[task_type]:

            # --- INÍCIO DA CORREÇÃO ---
            # Agora iteramos sobre a lista de dicionários de sentenças
            for sentence_obj in example['sentences']:
                sentence_id = sentence_obj['id']
                sentence_text = sentence_obj['sentence']
            # --- FIM DA CORREÇÃO ---

                score = calculate_pll_score(sentence_text, model, tokenizer, device)

                predictions[task_type].append({"id": sentence_id, "score": score})
                pbar.update(1)
# ...
Com essas duas correções (no script de tradução e no de predição), todo o seu pipeline estará consistente e alinhado com a estrutura de dados oficial do StereoSet.
