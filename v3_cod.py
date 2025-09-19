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



python src/run_evaluation.py \
    --model-name-or-path neuralmind/bert-base-portuguese-cased \
    --input-file /caminho/para/seu/arquivo_intrasentence_traduzido.json \
    --output-file resultados_intrasentence_bertimbau.json \
    --task intrasentence

Script para Unificar os Arquivos:

Salve o código abaixo como unificar_json.py, por exemplo.

Coloque seus dois arquivos traduzidos (ex: intersentence_pt.json e intrasentence_pt.json) no mesmo diretório.

Execute o script. Ele criará um novo arquivo chamado stereoset_pt_gold.json.

Python

import json

# Nomes dos seus arquivos traduzidos
intersentence_file = 'intersentence_pt.json'
intrasentence_file = 'intrasentence_pt.json'
output_file = 'stereoset_pt_gold.json'

# Dicionário para armazenar o conteúdo dos dois arquivos
data_unificada = {}

# Carregar dados de intersentence
try:
    with open(intersentence_file, 'r', encoding='utf-8') as f:
        # O arquivo original tem "intersentence" como chave principal
        data_unificada['intersentence'] = json.load(f)['intersentence']
    print(f"✅ Arquivo '{intersentence_file}' carregado com sucesso.")
except (json.JSONDecodeError, KeyError) as e:
    print(f"❌ Erro ao ler '{intersentence_file}'. Verifique se o formato JSON é válido e contém a chave 'intersentence'. Erro: {e}")
    exit()


# Carregar dados de intrasentence
try:
    with open(intrasentence_file, 'r', encoding='utf-8') as f:
        # O arquivo original tem "intrasentence" como chave principal
        data_unificada['intrasentence'] = json.load(f)['intrasentence']
    print(f"✅ Arquivo '{intrasentence_file}' carregado com sucesso.")
except (json.JSONDecodeError, KeyError) as e:
    print(f"❌ Erro ao ler '{intrasentence_file}'. Verifique se o formato JSON é válido e contém a chave 'intrasentence'. Erro: {e}")
    exit()

# Salvar o novo arquivo unificado
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_unificada, f, indent=2, ensure_ascii=False)

print(f"\n🚀 Arquivos unificados com sucesso em '{output_file}'!")

Agora você tem o arquivo stereoset_pt_gold.json, que está no formato exato que o script de avaliação precisa.

🛠️ Passo 2: Gerar as Predições (Scores) com o BERTimbau
O script de avaliação precisa de um segundo arquivo: o de predições. Este arquivo contém a "pontuação" (score) que o BERTimbau atribui a cada frase individualmente. O método padrão para isso é a pseudo-log-likelihood (PLL), que mede o quão "provável" uma frase é de acordo com o modelo.

Você precisará criar um script que:

Carregue o modelo BERTimbau.

Leia o seu arquivo stereoset_pt_gold.json.

Para cada frase (estereótipo, anti-estereótipo e não relacionada) em cada exemplo, calcule seu score usando o BERTimbau.

Salve esses scores em um arquivo JSON com o formato esperado.

O arquivo de predições deve ter a seguinte estrutura:

Exemplo de predictions_bertimbau.json:

JSON

{
    "intrasentence": [
        {
            "id": "8899-7-1",
            "score": -12.345
        },
        {
            "id": "8899-7-2",
            "score": -15.678
        }
    ],
    "intersentence": [
        {
            "id": "9211-2-1",
            "score": -20.111
        },
        {
            "id": "9211-2-2",
            "score": -18.222
        }
    ]
}
O "id" de cada frase vem do seu arquivo gold e o "score" é a pontuação calculada pelo BERTimbau. Um score maior significa que o modelo considera a frase mais provável.

🚀 Passo 3: Executar a Avaliação
Com os dois arquivos prontos (stereoset_pt_gold.json e predictions_bertimbau.json), você pode finalmente executar o script de avaliação original sem nenhuma modificação.

Abra seu terminal no diretório onde estão os arquivos e execute o seguinte comando:

Bash

python evaluate.py --gold-file stereoset_pt_gold.json --predictions-file predictions_bertimbau.json --output-file results_bertimbau.json
O que cada argumento faz:

--gold-file stereoset_pt_gold.json: Aponta para o seu arquivo de dados traduzido e unificado.

--predictions-file predictions_bertimbau.json: Aponta para o arquivo com os scores gerados pelo BERTimbau.

--output-file results_bertimbau.json: Especifica onde salvar os resultados da avaliação.

O script irá então calcular o LM Score, Stereotype Score (SS) e ICAT Score para o BERTimbau com base nos seus dados em português, e imprimirá os resultados no console, além de salvá-los no arquivo results_bertimbau.json.
class Sentence:
    def __init__(self, id, gold_label):
        self.id = id
        self.gold_label = gold_label
