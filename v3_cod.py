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


Replace the original prediction loop in this function with the corrected version below.

Python

# In the evaluate_intersentence method:

        # ... (previous code) ...
        else:
            predictions = []
            for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_ids, token_type_ids, attention_mask, sentence_id = batch
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # CORRECTED PART STARTS HERE
                model_output = model(input_ids, token_type_ids=token_type_ids)
                # Access the .logits attribute from the output object
                logits = model_output.logits 
                outputs = torch.softmax(logits, dim=1)
                # CORRECTED PART ENDS HERE

                for idx in range(input_ids.shape[0]):
                    probabilities = {}
                    probabilities['id'] = sentence_id[idx]
                    if "bert" == self.PRETRAINED_CLASS[:4] or "roberta-base" == self.PRETRAINED_CLASS:
                        probabilities['score'] = outputs[idx, 0].item()
                    else:
                        probabilities['score'] = outputs[idx, 1].item()
                    predictions.append(probabilities)

        return predictions
2. Correction for process_job
Similarly, replace the logic inside the process_job function.

Python

def process_job(batch, model, pretrained_class):
    input_ids, token_type_ids, sentence_id = batch
    
    # CORRECTED PART STARTS HERE
    model_output = model(input_ids, token_type_ids=token_type_ids)
    # Access the .logits attribute from the output object
    logits = model_output.logits
    outputs = torch.softmax(logits, dim=1)
    # CORRECTED PART ENDS HERE

    pid = sentence_id[0]
    if "bert" in pretrained_class:
        pscore = outputs[0, 0].item()
    else:
        pscore = outputs[0, 1].item()
    return (pid, pscore)
By making these changes, you are correctly extracting the tensor of logits from the model's output before passing it to torch.softmax, which will resolve the TypeError.
class Sentence:
    def __init__(self, id, gold_label):
        self.id = id
        self.gold_label = gold_label
