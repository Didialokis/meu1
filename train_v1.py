import sagemaker
from sagemaker.huggingface import HuggingFace
import os

# Configuração da Sessão e Role do SageMaker
try:
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    sagemaker_default_bucket = sagemaker_session.default_bucket()
    print(f"Sessão SageMaker e Role configurados. Bucket padrão: {sagemaker_default_bucket}")
except Exception as e:
    print(f"AVISO: Não foi possível obter role/sessão SageMaker automaticamente: {e}. Usando placeholders.")
    role = "arn:aws:iam::SEU_ACCOUNT_ID:role/SEU_SAGEMAKER_EXECUTION_ROLE" # SUBSTITUA se rodar localmente
    sagemaker_session = sagemaker.Session()
    sagemaker_default_bucket = "seu-bucket-sagemaker-padrao" # SUBSTITUA se rodar localmente
print(f"Role ARN: {role}")

# Hiperparâmetros para o script de treinamento MLM (train_pipeline.py)
# Foco em streaming direto do Hub para Aroeira
hyperparameters = {
    # Configs Gerais e de Dados
    'max_len': 128,
    'aroeira_subset_size': 10000, # Controla quantos exemplos pegar do stream do Hub
    'vocab_size': 30000,
    'min_frequency_tokenizer': 2,
    'trust_remote_code': "True", # Para datasets.load_dataset

    # Configs Pré-treinamento MLM
    'epochs_pretrain': 1,
    'batch_size_pretrain': 8,
    'lr_pretrain': 5e-5,

    # Arquitetura Modelo BERT
    'model_hidden_size': 256,
    'model_num_layers': 2,
    'model_num_attention_heads': 4,
    'model_dropout_prob': 0.1,
    
    # Nomes de Arquivos de Saída (o script de treino adicionará o output_dir antes)
    'tokenizer_vocab_filename': "aroeira_mlm_tokenizer-vocab.json",
    'tokenizer_merges_filename': "aroeira_mlm_tokenizer-merges.txt",
    'pretrained_bertlm_save_filename': "aroeira_bertlm_pretrained.pth",
    'temp_tokenizer_train_file': "temp_aroeira_for_tokenizer.txt",
    
    # Controle de Fluxo (foco apenas em MLM)
    'do_dataprep_tokenizer': "True",
    'do_pretrain': "True",
    'do_finetune_nli': "False", 
    'do_finetune_ner': "False",
    
    'logging_steps': 50,
    'evaluation_strategy': "epoch", 
    'save_strategy': "epoch", 
    'fp16': "False",
    'load_best_model_at_end': "False",
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'weight_decay': 0.01,
    # sagemaker_input_data_dir e input_data_filename foram removidos pois Aroeira virá do Hub
}
print(f"Hiperparâmetros para o job: {hyperparameters}")

# Configuração do HuggingFace Estimator
SOURCE_DIRECTORY = './scripts/' 
ENTRY_POINT_SCRIPT = 'train_pipeline.py' 

huggingface_estimator = HuggingFace(
    entry_point=ENTRY_POINT_SCRIPT,
    source_dir=SOURCE_DIRECTORY,
    role=role,
    transformers_version='4.36.0', # Ou '4.36'
    pytorch_version='2.1.0',     # Ou '2.1'
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge', 
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    dependencies=[os.path.join(SOURCE_DIRECTORY, 'requirements.txt')],
    output_path=f"s3://{sagemaker_default_bucket}/aroeira_mlm_hub_stream_output/job_output/",
    model_dir=f"s3://{sagemaker_default_bucket}/aroeira_mlm_hub_stream_output/model_artifacts/",
)

print(f"Iniciando job de treinamento SageMaker para: {os.path.join(SOURCE_DIRECTORY, ENTRY_POINT_SCRIPT)}")
print("O script de treino baixará/fará stream dos datasets (Aroeira, etc.) diretamente do Hugging Face Hub.")

# Para esta configuração, não precisamos passar 'inputs' para .fit() para o Aroeira,
# pois o script de treino o carregará do Hub.
huggingface_estimator.fit(wait=True)

print("Job do SageMaker (MLM Aroeira via Hub Streaming) submetido/concluído.")
