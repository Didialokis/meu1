import sagemaker
from sagemaker.huggingface import HuggingFace
import os # Adicionado para os.path.join, se necessário (embora não usado no snippet final simplificado)

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
sagemaker_default_bucket = sagemaker_session.default_bucket()

hyperparameters = {
    'max_len': 128,
    'aroeira_subset_size': 10000,
    'vocab_size': 30000,
    'min_frequency_tokenizer': 2,
    'trust_remote_code': "True",

    'epochs_pretrain': 1,
    'batch_size_pretrain': 8, # Será usado se per_device_train_batch_size não for passado ou priorizado
    'lr_pretrain': 5e-5,

    'model_hidden_size': 256,
    'model_num_layers': 2,
    'model_num_attention_heads': 4,
    'model_dropout_prob': 0.1,
    
    'output_dir': os.environ.get('SM_MODEL_DIR', './pipeline_outputs'), # Script de treino usa SM_MODEL_DIR

    'tokenizer_vocab_filename': "aroeira_tokenizer-vocab.json",
    'tokenizer_merges_filename': "aroeira_tokenizer-merges.txt",
    'pretrained_bertlm_save_filename': "aroeira_bertlm_pretrained.pth",
    'temp_tokenizer_train_file': "temp_aroeira_for_tokenizer.txt",
    
    'finetune_epochs': 1,
    'finetune_batch_size': 8, # Será usado se per_device_train_batch_size não for passado ou priorizado
    'finetune_lr': 3e-5,
    
    'assin_nli_model_save_filename': "aroeira_assin_nli_model.pth",
    'assin_subset_train': 200,
    'assin_subset_val': 50,

    'harem_model_save_filename': "aroeira_harem_ner_model.pth",
    'harem_subset_size': 100,
    'pad_token_label_id_ner': -100,

    'do_dataprep_tokenizer': "True",
    'do_pretrain': "True",
    'do_finetune_nli': "False", 
    'do_finetune_ner': "False",
    
    # Args genéricos que seu script parse_args pode aceitar (opcional)
    'logging_steps': 50,
    'evaluation_strategy': "epoch",
    'save_strategy': "epoch",
    'fp16': "False" # Defina como "True" se for usar e a instância suportar
}

# Assume que train_pipeline.py e requirements.txt estão em './scripts/'
SOURCE_DIRECTORY = './scripts/' 
ENTRY_POINT_SCRIPT = 'train_pipeline.py' 

huggingface_estimator = HuggingFace(
    entry_point=ENTRY_POINT_SCRIPT,
    source_dir=SOURCE_DIRECTORY,
    role=role,
    transformers_version='4.36',
    pytorch_version='2.1',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # Escolha uma instância GPU apropriada
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    dependencies=[os.path.join(SOURCE_DIRECTORY, 'requirements.txt')] 
    # Opcional: output_path para logs e outros artefatos no S3
    # output_path=f"s3://{sagemaker_default_bucket}/aroeira_pipeline_job_outputs/",
    # Opcional: model_dir no S3 (para onde /opt/ml/model será copiado)
    # model_dir=f"s3://{sagemaker_default_bucket}/aroeira_pipeline_model_artifacts/",
)

print(f"Iniciando job de treinamento SageMaker para o script: {SOURCE_DIRECTORY}{ENTRY_POINT_SCRIPT}")
# O script train_pipeline.py baixa seus próprios datasets (Aroeira, ASSIN, HAREM)
# Se você tivesse um dataset principal no S3, você o passaria aqui:
# inputs = {'nome_do_canal': f"s3://{sagemaker_default_bucket}/caminho/para/dados/"}
# huggingface_estimator.fit(inputs, wait=True)
huggingface_estimator.fit(wait=True)

print("Job do SageMaker submetido/concluído.")
