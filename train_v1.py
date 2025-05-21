import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import os # Necessário para os.path.basename

# Inicializa a sessão e o role do SageMaker
sagemaker_session = sagemaker.Session() # Nome correto da classe
role = get_execution_role()

print(f"SageMaker Role ARN: {role}")
print(f"Sessão SageMaker: {sagemaker_session}")
sagemaker_default_bucket = sagemaker_session.default_bucket()
if sagemaker_default_bucket:
    print(f"Bucket S3 Padrão: {sagemaker_default_bucket}")

# Caminho S3 para o seu dataset Aroeira principal
# Este arquivo JSON será baixado para o container de treinamento.
train_data_s3_path = 's3://393325852832-processedtexts/moderated/final_dataset_autoral_rights.json'

# Hiperparâmetros que serão passados para o script de treinamento (bert_aroeira_v2.py)
# As chaves aqui devem corresponder aos nomes dos argumentos definidos no argparse do script de treino.
hyperparameters = {
    # Configs Gerais e de Dados
    'max_len': 128,
    # 'aroeira_subset_size' não é mais estritamente necessário aqui se 'train_data_s3_path'
    # já representa o conjunto de dados completo ou o subconjunto desejado para o Aroeira.
    # Se o script de treino ainda tiver lógica para subsetting interno, você pode passar.
    # Para este exemplo, assumimos que o JSON em S3 é o dataset a ser usado.
    'vocab_size': 30000,
    'min_frequency_tokenizer': 2,
    'trust_remote_code': "True", # Booleano como string para SageMaker

    # Caminho para os dados DENTRO do container SageMaker (do canal 'train')
    'sagemaker_input_data_dir': '/opt/ml/input/data/train/', # Corresponde ao canal 'train' no .fit()
    'input_data_filename': os.path.basename(train_data_s3_path), # Nome do arquivo dentro do dir acima

    # Configs Pré-treinamento MLM
    'epochs_pretrain': 1,        # Exemplo: mantenha baixo para testes iniciais no SageMaker
    'batch_size_pretrain': 8,    # Ajuste conforme o instance_type
    'lr_pretrain': 5e-5,

    # Arquitetura Modelo BERT
    'model_hidden_size': 256,
    'model_num_layers': 2,
    'model_num_attention_heads': 4,
    'model_dropout_prob': 0.1,
    
    # Nomes de Arquivos de Saída (o script de treino adicionará o output_dir antes)
    # O script de treino usará SM_MODEL_DIR (/opt/ml/model) como output_dir principal.
    'tokenizer_vocab_filename': "aroeira_mlm_tokenizer-vocab.json",
    'tokenizer_merges_filename': "aroeira_mlm_tokenizer-merges.txt",
    'pretrained_bertlm_save_filename': "aroeira_bertlm_pretrained.pth",
    'temp_tokenizer_train_file': "temp_aroeira_for_tokenizer.txt",
    
    # Controle de Fluxo (foco apenas em MLM para este exemplo simplificado)
    'do_dataprep_tokenizer': "True", # Executar preparação de dados e treino do tokenizador
    'do_pretrain': "True",           # Executar pré-treinamento MLM
    'do_finetune_nli': "False",      # Desabilitado
    'do_finetune_ner': "False",      # Desabilitado
    
    # Argumentos comuns do HF Trainer (para compatibilidade se seu parse_args os tiver)
    'logging_steps': 50,
    'evaluation_strategy': "epoch", 
    'save_strategy': "epoch", 
    'fp16': "False", # Defina como "True" se a instância e o script suportarem
    'load_best_model_at_end': "False",
    # Podem ser adicionados outros hiperparâmetros que seu script de treino espera
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'weight_decay': 0.01,
}
print(f"Hiperparâmetros configurados para o job: {hyperparameters}")

# Configurar o Hugging Face Estimator
# O nome do script de entrada deve ser o nome do seu arquivo .py principal
ENTRY_POINT_SCRIPT = 'bert_aroeira_v2.py' # Conforme sua imagem
SOURCE_DIRECTORY = './' # Assume que o script e requirements.txt estão no diretório atual

huggingface_estimator = HuggingFace(
    entry_point=ENTRY_POINT_SCRIPT,
    source_dir=SOURCE_DIRECTORY,
    role=role,
    transformers_version='4.36', # Mantendo as versões da sua imagem
    pytorch_version='2.1.0',     # Mantendo as versões da sua imagem
    py_version='py310',          # Mantendo as versões da sua imagem
    instance_count=1,
    instance_type='ml.g5.2xlarge', # Mantendo a instância da sua imagem
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    dependencies=['requirements.txt'] # Assume que está na raiz de source_dir
)

# Configurar o input de dados do S3 para o canal 'train'
# O nome do canal 'train' aqui resultará no caminho /opt/ml/input/data/train/ no container
s3_input_data = sagemaker.inputs.TrainingInput(
    s3_data=train_data_s3_path,
    distribution='FullyReplicated',
    content_type='application/json', # Assumindo que seu arquivo S3 é JSON
    s3_data_type='S3Prefix' 
)
inputs = {'train': s3_input_data} # Canal nomeado 'train'

print(f"Iniciando job de treinamento SageMaker para: {os.path.join(SOURCE_DIRECTORY, ENTRY_POINT_SCRIPT)}")
print(f"Dados de entrada S3: {train_data_s3_path} -> canal 'train'")
print(f"Dados estarão disponíveis em: {hyperparameters['sagemaker_input_data_dir']}{hyperparameters['input_data_filename']}")

huggingface_estimator.fit(inputs, wait=True) # wait=True para o script esperar a conclusão

print("Job do SageMaker submetido/concluído.")
