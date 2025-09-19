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


python src/run_evaluation.py \
    --model-name-or-path neuralmind/bert-base-portuguese-cased \
    --input-file /caminho/para/seu/arquivo_intrasentence_traduzido.json \
    --output-file resultados_intrasentence_bertimbau.json \
    --task intrasentence

import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import logging

# Desativa logs de informação da biblioteca 'transformers' para um output mais limpo
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- CONFIGURAÇÕES ---
# Mude para 'neuralmind/bert-large-portuguese-cased' se quiser usar o modelo grande
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased' 
GOLD_FILE = 'stereoset_pt_gold.json'
OUTPUT_FILE = 'predictions_bertimbau.json'
# ---------------------

def calculate_pll_score(text, model, tokenizer, device):
    """
    Calcula a Pseudo-Log-Likelihood (PLL) para uma dada sentença.
    Scores mais altos (menos negativos) indicam maior probabilidade.
    """
    # Tokeniza a sentença, adicionando tokens especiais [CLS] e [SEP]
    tokenized_input = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Ignora os tokens [CLS] e [SEP] no cálculo do score
    tokens_to_score = tokenized_input[0][1:-1]
    
    total_log_prob = 0.0

    # Itera sobre cada token da sentença (exceto [CLS] e [SEP])
    for i in range(1, len(tokenized_input[0]) - 1):
        
        # Cria uma cópia dos IDs para mascarar o token da iteração atual
        masked_input = tokenized_input.clone()
        
        # Guarda o ID do token original que será mascarado
        original_token_id = masked_input[0, i].item()
        
        # Mascara o token na posição 'i'
        masked_input[0, i] = tokenizer.mask_token_id

        # Realiza a predição com o modelo sem calcular gradientes para otimização
        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits
        
        # Pega os logits (saída bruta) apenas para a posição do token mascarado
        masked_token_logits = logits[0, i, :]
        
        # Aplica log_softmax para converter logits em log-probabilidades
        log_probs = torch.nn.functional.log_softmax(masked_token_logits, dim=0)
        
        # Pega a log-probabilidade específica do token original e soma ao total
        token_log_prob = log_probs[original_token_id].item()
        total_log_prob += token_log_prob
        
    return total_log_prob


def generate_predictions():
    """
    Função principal que carrega o modelo, os dados, calcula os scores
    e salva o arquivo de predições.
    """
    # Verifica se a GPU está disponível e define o dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Usando dispositivo: {device.upper()}")

    # Carrega o modelo e o tokenizador pré-treinados
    print(f"💾 Carregando modelo '{MODEL_NAME}'... (Isso pode levar um momento)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval() # Coloca o modelo em modo de avaliação
    print("✅ Modelo carregado com sucesso!")

    # Carrega o arquivo gold unificado
    try:
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{GOLD_FILE}' não encontrado. Verifique o nome e o caminho do arquivo.")
        return

    predictions = {"intrasentence": [], "intersentence": []}
    total_sentences = sum(len(ex['sentences']) for task in gold_data.values() for ex in task)
    
    print(f"📊 Processando {total_sentences} sentenças...")

    # Usa tqdm para criar uma barra de progresso
    with tqdm(total=total_sentences, unit="sentença") as pbar:
        # Itera sobre as duas tarefas (intrasentence e intersentence)
        for task_type in gold_data:
            for example in gold_data[task_type]:
                for sentence in example['sentences']:
                    sentence_id = sentence['id']
                    sentence_text = sentence['sentence']
                    
                    # Calcula o score PLL para a sentença
                    score = calculate_pll_score(sentence_text, model, tokenizer, device)
                    
                    # Adiciona o resultado à lista correta
                    predictions[task_type].append({"id": sentence_id, "score": score})
                    pbar.update(1)

    # Salva o arquivo de predições
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Arquivo de predições foi salvo com sucesso em '{OUTPUT_FILE}'!")


if __name__ == "__main__":
    generate_predictions()
