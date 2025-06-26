def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(...)
    # ... outros argumentos ...
    
    # --- MODIFICAÇÃO 1: REMOVER ARGUMENTOS ANTIGOS DE DADOS ---
    # Remova ou comente as linhas abaixo, pois serão substituídas
    # parser.add_argument("--aroeira_subset_size", type=int, default=10000)
    # parser.add_argument("--sagemaker_input_data_dir", type=str, default=None)
    # parser.add_argument("--input_data_filename", type=str, default=None)

    # --- MODIFICAÇÃO 2: ADICIONAR NOVOS ARGUMENTOS PARA STREAMING DO S3 ---
    parser.add_argument("--s3_data_path", type=str, required=True,
                        help="Caminho S3 completo para o arquivo de dados JSONL.")
    parser.add_argument("--streaming_subset_size", type=int, default=50000,
                        help="Número de exemplos para carregar do S3 para a memória.")

    # ... resto da função
    return known_args

//////////////////////////////////////////  Modificar a Função setup_data_and_train_tokenizer para Usar os Novos Argumentos


def setup_data_and_train_tokenizer(args, logger):
    logger.info("--- Fase: Preparação de Dados (Streaming do S3) e Tokenizador ---")
    _all_aroeira_sents_list = []
    text_col = "text"
    temp_file_for_tokenizer = Path(args.temp_tokenizer_train_file)

    # --- LÓGICA DE CARREGAMENTO DE DADOS TOTALMENTE SUBSTITUÍDA ---
    if not args.s3_data_path or not args.s3_data_path.startswith("s3://"):
        raise ValueError("Um caminho S3 válido deve ser fornecido via --s3_data_path.")

    logger.info(f"Iniciando streaming de dados do S3: {args.s3_data_path}")
    logger.info(f"Carregando um subconjunto de {args.streaming_subset_size} exemplos para a memória...")

    # 1. Inicia o streaming do S3 sem carregar tudo
    streamed_ds = datasets.load_dataset(
        "json",
        data_files=args.s3_data_path,
        split="train",
        streaming=True, # A MÁGICA ACONTECE AQUI!
        trust_remote_code=args.trust_remote_code
    )

    # 2. Itera sobre o stream e carrega apenas o subconjunto desejado na memória
    subset_iterator = streamed_ds.take(args.streaming_subset_size)
    
    try:
        for ex in tqdm(subset_iterator, desc="Processando stream do S3 para memória", total=args.streaming_subset_size):
            sent = ex.get(text_col)
            if isinstance(sent, str) and sent.strip():
                _all_aroeira_sents_list.append(sent.strip())
    except Exception as e:
        logger.error(f"Erro durante o streaming do dataset do S3: {e}")
        logger.error("Verifique se o arquivo no S3 é um JSONL válido e se as permissões estão corretas.")
        raise e

    if not _all_aroeira_sents_list:
        raise ValueError("Nenhuma sentença foi carregada do subconjunto do S3.")

    logger.info(f"Total de sentenças carregadas para pré-treino e tokenizador: {len(_all_aroeira_sents_list)}")
    
    # O resto da função (treino do tokenizador, etc.) permanece o mesmo,
    # pois agora opera na lista _all_aroeira_sents_list que está na memória.
    # ... (código de salvar arquivo temporário e treinar o tokenizador) ...

    # Retorna a lista em memória para o resto do pipeline
    return loaded_tokenizer, pad_id_val, _all_aroeira_sents_list

/////////////////////////////////////////////////// train.py

Parte 2: Modificar o Script Lançador (train.py)
Agora, no seu script que inicia o job do SageMaker, você precisa passar esses novos hiperparâmetros.

Python

# Em seu arquivo train.py

# ... (configuração de sessão e role) ...

# 2. Defina a variável para o caminho S3
# Esta será a variável "df" que você mencionou
s3_data_location = f's3://{sagemaker_default_bucket}/datasets/aroeira/final_dataset_autoral_rights.json'

# 3. Hiperparâmetros
hyperparameters = {
    # ... outros hiperparâmetros ...
    'max_len': 128,
    'vocab_size': 30000,
    'epochs_pretrain': 1,
    'batch_size_pretrain': 8,

    # --- MODIFICAÇÃO 3: PASSAR OS NOVOS PARÂMETROS ---
    # Remova os parâmetros antigos de dados
    # 'sagemaker_input_data_dir': ...,
    # 'input_data_filename': ...,
    # 'aroeira_subset_size': 10000,
    
    # Adicione os novos parâmetros de streaming
    's3_data_path': s3_data_location,
    'streaming_subset_size': 100000,  # <-- CONTROLE A MEMÓRIA AQUI! Aumente ou diminua conforme necessário.
    
    # ... outros hiperparâmetros ...
    'do_dataprep_tokenizer': "True",
    'do_pretrain': "True",
    'logging_steps': 50,
}

# 4. Configurar o HuggingFace Estimator (sem grandes alterações)
huggingface_estimator = HuggingFace(
    entry_point='train_article_style_pretraining.py',
    source_dir='./',
    role=role,
    # ... (configurações de instância, versão, etc.)
    hyperparameters=hyperparameters,
    # ...
)

# 5. REMOVER O CANAL DE INPUT (se não for mais necessário)
# Como o script agora lê diretamente do S3 via hiperparâmetro,
# você não precisa mais do canal de input 'inputs'.

print(f"Iniciando job de treinamento SageMaker.")
print(f"Lendo dados diretamente do S3: {hyperparameters['s3_data_path']}")
print(f"Tamanho do subconjunto em memória: {hyperparameters['streaming_subset_size']}")

# Iniciar o job de treinamento sem o canal de input
huggingface_estimator.fit(wait=True)

print("Job do SageMaker submetido/concluído.")
