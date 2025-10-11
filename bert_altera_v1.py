Para Avaliar o BERTimbau:
Execute o seguinte comando. Note que especificamos o nome do modelo em --pretrained-class e garantimos que o tokenizador e os modelos de avaliação sejam os corretos para a arquitetura BERT.

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
