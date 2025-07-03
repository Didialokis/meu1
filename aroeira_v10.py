Código de Treinamento Corrigido
Esta versão assume que o problema é o shard_size ser maior que os arquivos individuais. A nova lógica processará os arquivos em lotes.

1. parse_args() - Novo argumento files_per_shard

Substituímos shard_size e num_shards por files_per_shard.

Python

def parse_args():
    parser = argparse.ArgumentParser(...)
    
    # --- MODIFICAÇÃO: Argumentos para controlar o processamento de arquivos ---
    parser.add_argument("--s3_data_path", type=str, required=True, 
                        help="Caminho S3 ou LOCAL para o DIRETÓRIO contendo os arquivos batch_*.jsonl.")
    parser.add_argument("--files_per_shard", type=int, default=10, 
                        help="Número de arquivos .jsonl (batches) para processar em cada shard de treinamento.")

    # Remover argumentos antigos de sharding
    # parser.add_argument("--shard_size", ...)
    # parser.add_argument("--num_shards", ...)
    # ... resto dos argumentos ...
    return args
2. run_pretraining_on_shards - A Nova Lógica Principal

Esta função agora busca os nomes dos arquivos e os processa em grupos.

Python

import s3fs # Adicione esta importação no topo do seu arquivo

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    """
    Orquestra o treinamento processando os arquivos batch_*.jsonl em grupos.
    """
    logger.info("--- INICIANDO PROCESSO DE TREINAMENTO EM SHARDS DE ARQUIVOS ---")

    # 1. Obter a lista de todos os arquivos de dados do S3
    logger.info("Buscando a lista de arquivos de dados no S3...")
    base_data_path = args.s3_data_path.rstrip('/')
    glob_data_path = f"{base_data_path}/batch_*.jsonl"
    
    try:
        s3 = s3fs.S3FileSystem()
        all_files = sorted(s3.glob(glob_data_path)) # Usamos sorted para garantir a ordem
    except Exception as e:
        logger.error(f"Não foi possível listar os arquivos no S3 com o padrão '{glob_data_path}'. Erro: {e}")
        return

    if not all_files:
        logger.error(f"Nenhum arquivo de dados encontrado em '{glob_data_path}'. Encerrando.")
        return

    logger.info(f"Encontrados {len(all_files)} arquivos de dados para processar.")

    # 2. Criar os shards (grupos de arquivos)
    num_files_per_shard = args.files_per_shard
    file_shards = [all_files[i:i + num_files_per_shard] for i in range(0, len(all_files), num_files_per_shard)]
    
    logger.info(f"Dados divididos em {len(file_shards)} shards de treinamento, cada um com até {num_files_per_shard} arquivos.")

    # 3. Loop principal sobre os shards de arquivos
    for shard_num, file_list_for_shard in enumerate(file_shards):
        logger.info(f"--- Processando Shard {shard_num + 1}/{len(file_shards)} ---")
        logger.info(f"Arquivos neste shard: {file_list_for_shard[:3]}... (e mais {len(file_list_for_shard)-3} se houver)")

        # Carrega todos os arquivos deste shard em um único stream
        # Adiciona 's3://' ao início de cada caminho de arquivo se não estiver presente
        full_path_files = [f"s3://{f}" if not f.startswith('s3://') else f for f in file_list_for_shard]
        
        shard_ds = datasets.load_dataset("json", data_files=full_path_files, split="train")
        # Como carregamos um número gerenciável de arquivos, podemos carregá-los na memória
        # Se os shards de arquivos ainda forem muito grandes, a lógica de streaming pode ser re-adicionada aqui
        sentences_list = [ex['text'] for ex in shard_ds if ex.get('text')]

        logger.info(f"Shard {shard_num + 1} carregado na memória com {len(sentences_list)} sentenças.")
        if not sentences_list:
            logger.warning("Shard vazio. Pulando para o próximo.")
            continue
        
        # O resto da lógica (criar Dataset, DataLoader, Trainer, e treinar) permanece o mesmo.
        # ... (código para criar Dataset, DataLoader, Modelo e Trainer) ...
        # ... (chamada para trainer.train()) ...
        
    logger.info("--- PROCESSO DE TREINAMENTO EM SHARDS DE ARQUIVOS CONCLUÍDO ---")

3. setup_and_train_tokenizer - Pequeno Ajuste

Esta função também precisa ser ajustada para usar a nova lógica para carregar os primeiros arquivos para treinar o tokenizador.

Python

def setup_and_train_tokenizer(args, logger):
    # ... (semelhante a run_pretraining_on_shards, busca a lista de arquivos)
    base_data_path = args.s3_data_path.rstrip('/')
    glob_data_path = f"{base_data_path}/batch_*.jsonl"
    s3 = s3fs.S3FileSystem()
    all_files = sorted(s3.glob(glob_data_path))

    if not all_files:
        raise RuntimeError(f"Nenhum arquivo encontrado em {glob_data_path} para treinar o tokenizador.")

    # Usa os primeiros N arquivos para treinar o tokenizador
    files_for_tokenizer = all_files[:args.files_per_shard]
    logger.info(f"Usando os primeiros {len(files_for_tokenizer)} arquivos para treinar o tokenizador.")
    
    full_path_files = [f"s3://{f}" if not f.startswith('s3://') else f for f in files_for_tokenizer]
    tokenizer_ds = datasets.load_dataset("json", data_files=full_path_files, split="train")
    sentences_for_tokenizer = [ex['text'] for ex in tokenizer_ds if ex.get('text')]

    # ... (resto da lógica de treino do tokenizador com a lista `sentences_for_tokenizer`)
    return tokenizer, tokenizer.pad_token_id
