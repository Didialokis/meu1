3.1 Protocolo de Execução Passo a Passo
Para realizar a avaliação completa, siga os seguintes comandos no terminal, a partir de um diretório de trabalho.

Clonar o repositório StereoSet para obter os dados:

Bash

git clone https://github.com/moinnadeem/stereoset.git
Salvar os scripts: Crie e salve os quatro arquivos (requirements.txt, dataloader.py, generate_predictions.py, evaluation.py) no seu diretório de trabalho.

Instalar as dependências necessárias:

Bash

pip install -r requirements.txt
Gerar as previsões do modelo (etapa demorada):

Bash

mkdir -p predictions
python3 generate_predictions.py \
  --model_name_or_path bert-base-uncased \
  --output_file predictions/bert-base-uncased.json
Calcular e exibir as pontuações finais:

Bash

python3 evaluation.py \
  --gold-file stereoset/data/dev.json \
  --predictions-file predictions/bert-base-uncased.json

////////////////////////////////////////////////////////////////
Avaliar o Impacto da Qualidade do Corpus: Investigar e quantificar como a qualidade dos dados de pré-treinamento influencia o desempenho final de um modelo de linguagem para o português brasileiro.

Análise Comparativa de Modelos: Realizar uma comparação direta de performance entre o seu "modelA", treinado em um corpus de alta qualidade, e o modelo de referência BERTimbau.

Validação através de Tarefas Práticas: Medir o desempenho de ambos os modelos em um conjunto de tarefas de Processamento de Linguagem Natural (PLN) para obter métricas concretas sobre suas capacidades.

Demonstrar a Vantagem da Curadoria de Dados: Provar a hipótese de que um corpus mais limpo e bem estruturado resulta em um modelo de linguagem mais robusto e eficiente, capaz de superar baselines estabelecidos.

//// stereoset
  A Lacuna no Português: Atualmente, não existem datasets de avaliação amplamente adotados e específicos para medir este tipo de viés social em modelos de linguagem treinados para o português brasileiro, dificultando a análise de sua imparcialidade e segurança.

A Solução: Tradução e Adaptação Cultural: Para preencher essa lacuna, o projeto propõe a tradução e, crucialmente, a adaptação cultural do StereoSet para a realidade brasileira. Isso garante que os exemplos sejam relevantes e que os estereótipos avaliados façam sentido no contexto local.

Como a Avaliação Funciona: O dataset testa os modelos através de tarefas de preenchimento de lacunas e escolha de sentenças que revelam suas tendências associativas, fornecendo métricas claras sobre o nível de viés estereotipado que o modelo aprendeu durante o treinamento.

Objetivo Principal: O resultado, um "StereoSet-PT", servirá como uma ferramenta fundamental para comparar modelos como o "modelA" e o BERTimbau, permitindo analisar como a qualidade do corpus de treinamento impacta não apenas a performance, mas também o comportamento ético do modelo.
///////////////////////////////////////////////////////////////


python evaluate.py --gold-file stereoset_pt_gold.json --predictions-file predictions_bertimbau.json --output-file results_bertimbau.json


  

# -*- coding: utf-8 -*-
# NOME DO ARQUIVO: traduzir_final.py

# -*- coding: utf-8 -*-
# NOME DO ARQUIVO: traduzir_final.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import re

# --- CONFIGURAÇÕES ---
MODEL_NAME = "facebook/nllb-200-1.3B"
DATASET_NAME = "McGill-NLP/stereoset"
CONFIGS = ['intersentence', 'intrasentence']
DATASET_SPLIT = "validation"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "por_Latn"
BATCH_SIZE = 8
PLACEHOLDER = "__BLANK_PLACEHOLDER__"

def sanitize_text(text):
    """Limpa o texto, removendo caracteres de controle que podem quebrar o JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def traduzir_dataset_completo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    print(f"Carregando o modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Modelo carregado com sucesso.")

    datasets_dict = {}
    sentences_to_translate = []

    for config in CONFIGS:
        print(f"Baixando e carregando a configuração '{config}' do dataset...")
        dataset = load_dataset(DATASET_NAME, config, split=DATASET_SPLIT)
        datasets_dict[config] = dataset
        for example in dataset:
            sentences_to_translate.append(example['context'])
            sentences_to_translate.extend(example['sentences']['sentence'])
    
    print(f"Total de {len(sentences_to_translate)} sentenças extraídas.")

    print("Iniciando a tradução em lotes...")
    translated_sentences = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    for i in range(0, len(sentences_to_translate), BATCH_SIZE):
        batch = sentences_to_translate[i:i + BATCH_SIZE]
        batch_com_placeholder = [s.replace("BLANK", PLACEHOLDER) for s in batch]
        inputs = tokenizer(batch_com_placeholder, return_tensors="pt", padding=True, truncation=True).to(device)
        
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=128)
        batch_translated_raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        batch_translated_final = [s.replace(PLACEHOLDER, "BLANK") for s in batch_translated_raw]
        batch_sanitized = [sanitize_text(text) for text in batch_translated_final]
        translated_sentences.extend(batch_sanitized)
        
        print(f"  Lote {i//BATCH_SIZE + 1} de {len(sentences_to_translate)//BATCH_SIZE + 1} concluído...")

    print("Tradução finalizada.")
    print("Reconstruindo os datasets com as sentenças traduzidas...")
    translated_iter = iter(translated_sentences)

    for config in CONFIGS:
        dataset_original = datasets_dict[config]

        def replace_sentences(example):
            example['context'] = next(translated_iter)
            num_target_sentences = len(example['sentences']['sentence'])
            translated_target_sentences = [next(translated_iter) for _ in range(num_target_sentences)]
            example['sentences']['sentence'] = translated_target_sentences
            return example

        translated_dataset = dataset_original.map(replace_sentences)
        
        output_path = f"stereoset_{config}_{DATASET_SPLIT}_pt_nllb_final.json"
        print(f"Salvando o dataset '{config}' traduzido em: {output_path}")
        translated_dataset.to_json(output_path, force_ascii=False) # Removido indent=2 para salvar em JSON Lines

    print("\nSucesso! Processo concluído.")

if __name__ == "__main__":
    traduzir_dataset_completo()
