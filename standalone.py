# Célula 2: Configuração Central do Treinamento

config = SimpleNamespace(
    # --- Configurações de Dados ---
    s3_data_path="s3://seu-bucket-aqui/caminho/para/dados.jsonl", # <-- MUDE AQUI
    shard_size=100000,          # Tamanho de cada fragmento de dados a ser lido do S3
    num_training_shards=5,      # Número de fragmentos a processar no total
    
    # --- Configurações de Arquitetura do Modelo ---
    model_d_model=256,
    model_n_layers=3,
    model_heads=4,
    model_dropout_prob=0.1,
    
    # --- Configurações de Treinamento ---
    epochs_pretrain=1,          # Épocas para treinar em CADA shard (1 é recomendado)
    batch_size_pretrain=16,     # Ajuste de acordo com a memória da sua GPU
    lr_pretrain_adam=1e-4,
    warmup_steps=2500,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.01,

    # --- Configurações de Tokenizer e Sequência ---
    max_len=128,
    vocab_size=30000,
    min_frequency_tokenizer=2,
    
    # --- Configurações de Paths Locais ---
    output_dir="./notebook_outputs", # Onde os modelos e checkpoints serão salvos
    log_filename="notebook_pretrain.log",
    temp_tokenizer_train_file="temp_notebook_for_tokenizer.txt",
    pretrained_bert_save_filename="best_model_notebook.pth",

    # --- Outras Configurações ---
    device=DEVICE,
    trust_remote_code=True,
    log_level='INFO',
    logging_steps=50,
)

# Cria os caminhos derivados
config.model_intermediate_size = config.model_d_model * 4
output_dir_path = Path(config.output_dir)
output_dir_path.mkdir(parents=True, exist_ok=True)
config.pretrained_bert_save_filename = str(output_dir_path / config.pretrained_bert_save_filename)
config.temp_tokenizer_train_file = str(output_dir_path / config.temp_tokenizer_train_file)
config.log_file_path = str(output_dir_path / config.log_filename)
