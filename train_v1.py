import sagemaker
from sagemaker.huggingface import HuggingFace
# from sagemaker import get_execution_role # sagemaker.get_execution_role() é o usual

# --- Bloco de Inicialização SageMaker ---
print("Configurando sessão e role do SageMaker...")
try:
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    sagemaker_default_bucket = sagemaker_session.default_bucket()
except Exception as e:
    # Fallback para quando não está em um ambiente SageMaker típico (ex: rodando localmente para teste de submissão)
    # Você pode precisar configurar suas credenciais AWS e um role ARN manualmente neste caso.
    print(f"AVISO: Não foi possível obter role/sessão automaticamente ({e}). Usando placeholders.")
    print("Certifique-se de que suas credenciais AWS e o role ARN estão corretos se estiver fora do SageMaker.")
    # Substitua pelo seu role ARN se estiver testando localmente e o get_execution_role() falhar
    role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_EXECUTION_ROLE" 
    sagemaker_session = sagemaker.Session() # Pode precisar de mais configuração para rodar localmente com AWS real
    sagemaker_default_bucket = "your-sagemaker-default-bucket-name" # Substitua se necessário

print(f"Role ARN: {role}")
print(f"Bucket S3 Padrão: {sagemaker_default_bucket}")

# --- Hiperparâmetros para o script de treinamento BERT (train_pipeline.py) ---
# Estes serão passados como argumentos de linha de comando para o seu entry_point script.
# As chaves devem corresponder aos nomes dos argumentos definidos no argparse do seu script de treino.
hyperparameters = {
    # Configs Gerais e de Dados
    'max_len': 128,
    'aroeira_subset_size': 10000,       # Para um teste mais rápido no SageMaker. Omita ou use um valor alto para dataset completo.
    'vocab_size': 30000,
    'min_frequency_tokenizer': 2,
    'trust_remote_code': "True",        # SageMaker passa booleanos como strings "True" ou "False"

    # Configs Pré-treinamento MLM
    'epochs_pretrain': 1,               # Mantenha baixo para testes iniciais
    'batch_size_pretrain': 8,           # Nome do arg no script: per_device_train_batch_size_pretrain se usou dest
                                        # Ou use o arg geral per_device_train_batch_size se o script o priorizar
    'lr_pretrain': 5e-5,                # Nome do arg no script: learning_rate_pretrain

    # Arquitetura Modelo BERT
    'model_hidden_size': 256,
    'model_num_layers': 2,
    'model_num_attention_heads': 4,
    'model_dropout_prob': 0.1,

    # Configs Fine-tuning
    'finetune_epochs': 1,               # Mantenha baixo para testes
    'finetune_batch_size': 8,
    'finetune_lr': 3e-5,
    
    'assin_subset_train': 200,
    'assin_subset_val': 50,
    'harem_subset_size': 100,
    'pad_token_label_id_ner': -100,

    # Controle de Fluxo (o que executar)
    'do_dataprep_tokenizer': "True",
    'do_pretrain': "True",
    'do_finetune_nli': "False",         # Exemplo: desabilitar NLI para teste
    'do_finetune_ner': "False",         # Exemplo: desabilitar NER para teste

    # Nomes de arquivos base (o script de treino adicionará o output_dir antes)
    # O output_dir principal é gerenciado pelo SageMaker (SM_MODEL_DIR)
    'tokenizer_vocab_filename': "aroeira_mlm_tokenizer-vocab.json",
    'tokenizer_merges_filename': "aroeira_mlm_tokenizer-merges.txt",
    'pretrained_bertlm_save_filename': "aroeira_bertlm_pretrained.pth",
    'temp_tokenizer_train_file': "temp_aroeira_for_tokenizer.txt",
    'assin_nli_model_save_filename': "aroeira_assin_nli_model.pth",
    'harem_model_save_filename': "aroeira_harem_ner_model.pth",

    # Outros argumentos que seu `parse_args` no script de treino possa esperar
    # (ex: logging_steps, save_steps, evaluation_strategy, etc.)
    'logging_steps': 50,
    'evaluation_strategy': "epoch", # Ou "steps" se preferir e configurar eval_steps
    'save_strategy': "epoch",     # Ou "steps"
    # 'fp16': "True", # Habilite se a instância e o script suportarem
}
print(f"Hiperparâmetros definidos: {hyperparameters}")

# --- Configuração do HuggingFace Estimator ---
# Assume que seu script de treino (`train_pipeline.py`) e `requirements.txt` 
# estão em um subdiretório chamado 'scripts' relativo a este script de setup.
# Se estiverem no mesmo diretório, use source_dir='./'.
SOURCE_DIRECTORY = './scripts/' # Modifique se necessário
ENTRY_POINT_SCRIPT = 'train_pipeline.py' # O nome do seu script principal

print(f"Configurando HuggingFace Estimator com entry_point='{ENTRY_POINT_SCRIPT}' em source_dir='{SOURCE_DIRECTORY}'")
huggingface_estimator = HuggingFace(
    entry_point=ENTRY_POINT_SCRIPT,
    source_dir=SOURCE_DIRECTORY,
    role=role,
    transformers_version='4.36', # Versão do Transformers
    pytorch_version='2.1',     # Versão do PyTorch
    py_version='py310',        # Versão do Python
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # Ex: Instância GPU. Escolha conforme necessidade/orçamento.
                                    # ml.g5.2xlarge é mais potente. ml.m5.large para CPU (lento).
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    dependencies=[os.path.join(SOURCE_DIRECTORY, 'requirements.txt')], # Caminho para requirements.txt
    # O diretório de saída principal no S3. O SageMaker mapeia /opt/ml/model para cá.
    output_path=f"s3://{sagemaker_default_bucket}/aroeira_bert_pipeline_output/",
    # Se quiser checkpoints no S3 (útil para treinos longos):
    # checkpoint_s3_uri=f"s3://{sagemaker_default_bucket}/aroeira_bert_pipeline_checkpoints/",
    metric_definitions=[ # Opcional: para métricas no console do SageMaker
        {'Name': 'train:loss', 'Regex': r"Epoch \d+ \[Train\].*Avg Loss: (\d\.\d+)"},
        {'Name': 'eval:loss', 'Regex': r"Epoch \d+ \[Val\].*Avg Loss: (\d\.\d+)"},
        {'Name': 'eval:accuracy', 'Regex': r"Epoch \d+ \[Val\].*accuracy: (\d\.\d+)"},
        {'Name': 'eval:f1_w', 'Regex': r"Epoch \d+ \[Val\].*f1_w: (\d\.\d+)"},
        {'Name': 'eval:f1_ner', 'Regex': r"Epoch \d+ \[Val\].*f1_ner: (\d\.\d+)"},
    ]
)

# --- Caminho de Dados S3 (Opcional) ---
# O script atual baixa os datasets (Aroeira, ASSIN, HAREM) do Hugging Face Hub.
# Se você tivesse um dataset principal já no S3 que quisesse usar, configuraria um input channel aqui.
# Exemplo:
# train_data_s3_path = f"s3://{sagemaker_default_bucket}/meu_dataset_aroeira_processado/"
# inputs = {'nome_do_canal': train_data_s3_path}
# E no seu script de treino, acessaria os dados de '/opt/ml/input/data/nome_do_canal/'

# --- Iniciar o Job de Treinamento ---
print("Iniciando job de treinamento do SageMaker...")
# Como o script baixa os dados, não precisamos passar 'inputs' para .fit() para Aroeira/ASSIN/HAREM.
try:
    huggingface_estimator.fit(wait=True) # wait=True para este script esperar a conclusão
    print("Job do SageMaker concluído.")
except Exception as e_fit:
    print(f"Erro ao submeter ou executar o job do SageMaker: {e_fit}")
    import traceback
    traceback.print_exc()
