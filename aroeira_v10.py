Excelente pedido. Ter um histórico de checkpoints por época é uma prática muito valiosa, pois permite que você volte para um "estado dourado" do modelo, e não apenas para o último ponto de salvamento.

A solução ideal é implementar um sistema de dois níveis:

Checkpoint de Resumo (latest_checkpoint.pth): Continuará a ser salvo ao final de cada shard. Sua única função é a recuperação de desastres, garantindo que você nunca perca mais do que o progresso de um shard se o processo for interrompido. Ele será constantemente sobrescrito.

Snapshot de Época (epoch_01_checkpoint.pth, etc.): Um novo arquivo, nomeado com o número da época, que será salvo apenas no final de cada época global completa. Estes arquivos nunca serão sobrescritos, criando o seu histórico de versões.

A boa notícia é que podemos implementar isso com poucas alterações, principalmente na função save_checkpoint e na forma como a chamamos.

Código Completo das Funções Modificadas
Apenas as funções parse_args, save_checkpoint e run_pretraining_on_shards precisam de alterações. A função load_checkpoint não precisa mudar, pois ela sempre deve resumir a partir do latest_checkpoint.pth.

1. parse_args() - Adicionando um Argumento para Controlar o Novo Comportamento

Vamos adicionar uma flag para que você possa ligar ou desligar o salvamento dos snapshots de época.

Python

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Pré-treino BERT com Shards e Checkpoints Versionados por Época.")
    
    # ... (todos os argumentos anteriores) ...
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Diretório para salvar checkpoints.")

    # --- MODIFICAÇÃO: Argumento para controlar os snapshots de época ---
    parser.add_argument("--save_epoch_checkpoints", action='store_true',
                        help="Se especificado, salva um checkpoint separado no final de cada época global.")

    # ... (resto dos argumentos) ...
    
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args
Nota: action='store_true' cria uma flag booleana. Se você incluir --save_epoch_checkpoints no seu comando, o valor será True. Se não incluir, será False.

2. save_checkpoint() - Modificada para Salvar Snapshots de Época

A função agora terá um parâmetro extra, save_epoch_snapshot, para decidir se deve ou não criar o arquivo de versão da época.

Python

def save_checkpoint(args, global_epoch, shard_num, model, optimizer, scheduler, best_val_loss, save_epoch_snapshot=False):
    """
    Salva um checkpoint. Sempre salva 'latest_checkpoint.pth'.
    Opcionalmente, salva um snapshot versionado da época.
    """
    checkpoint_dir_str = args.checkpoint_dir
    is_s3 = checkpoint_dir_str.startswith("s3://")
    s3 = s3fs.S3FileSystem() if is_s3 else None
    
    state = {
        'global_epoch': global_epoch,
        'shard_num': shard_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'rng_state': random.getstate(),
    }

    # 1. Salva o estado em um buffer na memória
    buffer = io.BytesIO()
    torch.save(state, buffer)

    # 2. Sempre salva/sobrescreve o 'latest_checkpoint.pth' para resumo
    buffer.seek(0) # Rebobina o buffer para o início
    if is_s3:
        latest_path = f"{checkpoint_dir_str.rstrip('/')}/latest_checkpoint.pth"
        with s3.open(latest_path, 'wb') as f:
            f.write(buffer.read())
    else:
        latest_path = Path(checkpoint_dir_str) / "latest_checkpoint.pth"
        with open(latest_path, 'wb') as f:
            f.write(buffer.read())
    logging.info(f"Checkpoint de resumo salvo em: {latest_path}")

    # --- MODIFICAÇÃO: Lógica para salvar o snapshot da época ---
    # 3. Se solicitado, salva um novo arquivo de checkpoint para a época
    if save_epoch_snapshot:
        buffer.seek(0) # Rebobina o buffer novamente
        epoch_filename = f"epoch_{global_epoch + 1:02d}_checkpoint.pth"
        
        if is_s3:
            epoch_path = f"{checkpoint_dir_str.rstrip('/')}/{epoch_filename}"
            with s3.open(epoch_path, 'wb') as f:
                f.write(buffer.read())
        else:
            epoch_path = Path(checkpoint_dir_str) / epoch_filename
            with open(epoch_path, 'wb') as f:
                f.write(buffer.read())
        logging.info(f"*** Snapshot da Época {global_epoch + 1} salvo em: {epoch_path} ***")

    # A lógica para salvar o melhor modelo continua a mesma...
    if best_val_loss < getattr(save_checkpoint, "global_best_val_loss", float('inf')):
        # ... (código para salvar best_model.pth)
3. run_pretraining_on_shards() - Orquestrando Quando Salvar

Esta função agora decide quando passar o sinalizador save_epoch_snapshot=True para a função save_checkpoint.

Python

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    logger.info("--- Fase: Pré-Treinamento em Épocas Globais e Shards ---")
    
    # ... (código para buscar a lista de arquivos `all_files_master_list`) ...
    
    # Instancia modelo e otimizadores fora do loop
    model = ArticleBERTLMWithHeads(...)
    optimizer = Adam(...)
    scheduler = ScheduledOptim(...)
    
    start_epoch, start_shard = load_checkpoint(args, model, optimizer, scheduler)
    
    # Loop de ÉPOCA GLOBAL
    for epoch_num in range(start_epoch, args.num_global_epochs):
        logger.info(f"--- INICIANDO ÉPOCA GLOBAL {epoch_num + 1}/{args.num_global_epochs} ---")
        
        current_files = list(all_files_master_list)
        if start_shard == 0:
            random.shuffle(current_files)
            logger.info("Ordem dos arquivos de dados foi embaralhada para esta época.")
        
        file_shards = [current_files[i:i + args.files_per_shard_training] for i in range(0, len(current_files), args.files_per_shard_training)]
        
        # Loop INTERNO sobre os shards
        for shard_num in range(start_shard, len(file_shards)):
            # ... (código para carregar o shard e criar DataLoaders) ...
            
            trainer = PretrainingTrainer(...)
            best_loss_in_shard = trainer.train(num_epochs=args.epochs_per_shard)
            
            # --- MODIFICAÇÃO: Lógica de chamada do save_checkpoint ---
            # Verifica se este é o último shard da época atual
            is_last_shard_of_epoch = (shard_num == len(file_shards) - 1)
            
            # Decide se o snapshot da época deve ser salvo
            should_save_epoch_snapshot = is_last_shard_of_epoch and args.save_epoch_checkpoints

            # Salva o checkpoint, passando a flag para o snapshot
            save_checkpoint(
                args,
                global_epoch=epoch_num,
                shard_num=shard_num,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_loss_in_shard,
                save_epoch_snapshot=should_save_epoch_snapshot
            )
            # --------------------------------------------------------

        # Reseta o start_shard para 0 para a próxima época global
        start_shard = 0

    logger.info(f"--- {args.num_global_epochs} ÉPOCAS GLOBAIS CONCLUÍDAS ---")
Como Funciona na Prática
Agora, ao rodar seu script com o novo argumento:

Bash

python train_bert_sharded.py \
    --s3_data_path "s3://seu-bucket/dados/" \
    --num_global_epochs 3 \
    --files_per_shard_training 10 \
    --save_epoch_checkpoints
O conteúdo do seu diretório de checkpoints (--checkpoint_dir) será:

Durante o treinamento: latest_checkpoint.pth será constantemente atualizado a cada 10 arquivos processados.

Ao final da Época 1: Um novo arquivo epoch_01_checkpoint.pth será criado e não será mais tocado. O latest_checkpoint.pth continuará a ser atualizado.

Ao final da Época 2: Um novo arquivo epoch_02_checkpoint.pth será criado.

Ao final da Época 3: Um novo arquivo epoch_03_checkpoint.pth será criado.


Se o processo parar na Época 2, Shard 5, o latest_checkpoint.pth terá o estado exato daquele ponto. Ao reiniciar, o load_checkpoint lerá este arquivo e o treinamento continuará do Shard 6 da Época 2, preservando seu histórico intacto dos arquivos epoch_01_checkpoint.pth.

//////////////////////////////////////////////////////////////
///
///
////////////////////////////////////////////////////////////////////////
# --- CORREÇÃO: Função de salvar checkpoint usando Boto3 ---
# --- CORREÇÃO FINAL: Função de salvar checkpoint usando Boto3 ---
def save_checkpoint(args, global_epoch, shard_num, model, optimizer, scheduler, best_val_loss):
    """
    Salva um checkpoint completo, usando Boto3 para caminhos S3.
    """
    checkpoint_dir_str = args.checkpoint_dir
    is_s3 = checkpoint_dir_str.startswith("s3://")
    
    state = {
        'global_epoch': global_epoch,
        'shard_num': shard_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'rng_state': random.getstate(),
    }

    # Salva o estado em um buffer na memória
    buffer = io.BytesIO()
    torch.save(state, buffer)
    # Importante: rebobine o buffer para o início antes de fazer o upload
    buffer.seek(0)

    if is_s3:
        s3 = boto3.client('s3')
        parsed_url = urlparse(checkpoint_dir_str)
        bucket = parsed_url.netloc
        dir_key = parsed_url.path.lstrip('/')
        checkpoint_key = f"{dir_key.rstrip('/')}/latest_checkpoint.pth"
        
        logging.info(f"Salvando checkpoint no S3: s3://{bucket}/{checkpoint_key}")
        try:
            s3.upload_fileobj(Fileobj=buffer, Bucket=bucket, Key=checkpoint_key)
        except ClientError as e:
            logging.error(f"Falha ao salvar checkpoint no S3: {e}")
            return # Sai da função se não conseguir salvar
    else: # Caminho local
        path_obj = Path(checkpoint_dir_str)
        path_obj.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path_obj / "latest_checkpoint.pth"
        logging.info(f"Salvando checkpoint localmente: {checkpoint_path}")
        with open(checkpoint_path, 'wb') as f:
            f.write(buffer.read())

    # Lógica para salvar o melhor modelo (também ciente do S3)
    if best_val_loss < getattr(save_checkpoint, "global_best_val_loss", float('inf')):
        save_checkpoint.global_best_val_loss = best_val_loss
        
        # Salva o state_dict do modelo em um buffer separado
        model_buffer = io.BytesIO()
        torch.save(model.state_dict(), model_buffer)
        model_buffer.seek(0)
        
        output_dir_str = args.output_dir
        if output_dir_str.startswith("s3://"):
            best_model_parsed_url = urlparse(output_dir_str)
            best_model_bucket = best_model_parsed_url.netloc
            best_model_key = f"{best_model_parsed_url.path.lstrip('/')}/best_model.pth"
            logging.info(f"*** Nova melhor validação global. Salvando modelo em s3://{best_model_bucket}/{best_model_key} ***")
            s3.upload_fileobj(Fileobj=model_buffer, Bucket=best_model_bucket, Key=best_model_key)
        else:
            best_model_path = Path(output_dir_str) / "best_model.pth"
            logging.info(f"*** Nova melhor validação global. Salvando modelo em {best_model_path} ***")
            Path(output_dir_str).mkdir(parents=True, exist_ok=True)
            with open(best_model_path, 'wb') as f:
                f.write(model_buffer.read())


# --- CORREÇÃO FINAL: Função de carregar checkpoint usando Boto3 ---
def load_checkpoint(args, model, optimizer, scheduler):
    """
    Carrega o último checkpoint, usando Boto3 para caminhos S3.
    """
    checkpoint_dir_str = args.checkpoint_dir
    is_s3 = checkpoint_dir_str.startswith("s3://")
    start_epoch = 0
    start_shard = 0
    
    checkpoint = None
    if is_s3:
        s3 = boto3.client('s3')
        parsed_url = urlparse(checkpoint_dir_str)
        bucket = parsed_url.netloc
        dir_key = parsed_url.path.lstrip('/')
        checkpoint_key = f"{dir_key.rstrip('/')}/latest_checkpoint.pth"
        
        logging.info(f"Verificando existência do checkpoint no S3: s3://{bucket}/{checkpoint_key}")
        try:
            # Baixa o objeto para um buffer em memória
            buffer = io.BytesIO()
            s3.download_fileobj(Bucket=bucket, Key=checkpoint_key, Fileobj=buffer)
            # Rebobina o buffer para o início para que o torch.load possa lê-lo
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location=args.device)
            logging.info("Checkpoint encontrado e carregado do S3.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info("Nenhum checkpoint encontrado no S3. Iniciando do zero.")
            else:
                logging.error(f"Erro inesperado ao acessar checkpoint no S3: {e}")
            return start_epoch, start_shard
    else: # Caminho local
        checkpoint_path = Path(checkpoint_dir_str) / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            logging.info(f"Carregando checkpoint localmente de: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
        else:
            logging.info("Nenhum checkpoint encontrado localmente. Iniciando do zero.")
            return start_epoch, start_shard

    # Aplica o estado aos objetos
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # Usa um atributo estático para manter o controle da melhor perda entre as execuções
    save_checkpoint.global_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    if 'rng_state' in checkpoint:
        random.setstate(checkpoint['rng_state'])
    
    start_epoch = checkpoint.get('global_epoch', 0)
    start_shard = checkpoint.get('shard_num', -1) + 1
    
    logging.info(f"Checkpoint aplicado. Resumindo da Época Global {start_epoch + 1}, Shard {start_shard + 1}.")
    return start_epoch, start_shard

# Não se esqueça de inicializar o atributo estático fora da função
save_checkpoint.global_best_val_loss = float('inf')

/////////////////////////////////////////////////////////////////////////////////////
def main():
    ARGS = parse_args()
    
    # --- MODIFICAÇÃO: Lógica do Modo de Teste Rápido ---
    if ARGS.quick_test:
        # Sobrescreve os argumentos para um teste rápido
        ARGS.num_global_epochs = 1
        ARGS.files_per_shard_training = 1
        ARGS.files_per_shard_tokenizer = 1
        ARGS.num_shards_limit = 2 # Processa apenas 2 shards
        ARGS.batch_size_pretrain = 4
        ARGS.max_len = 32
        ARGS.model_d_model = 64
        ARGS.model_n_layers = 1
        ARGS.model_heads = 2
        ARGS.output_dir = "./bert_quick_test_outputs"
        ARGS.checkpoint_dir = "./checkpoints_quick_test"
        # Limpa checkpoints antigos de teste para garantir um início limpo
        if os.path.exists(os.path.join(ARGS.checkpoint_dir, "latest_checkpoint.pth")):
             os.remove(os.path.join(ARGS.checkpoint_dir, "latest_checkpoint.pth"))
    # --------------------------------------------------------
    
    Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
    Path(ARGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(ARGS.output_dir) / f'training_log_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    setup_logging(ARGS.log_level, str(log_file))
    logger = logging.getLogger(__name__)

    if ARGS.quick_test:
        logger.warning("="*50)
        logger.warning("MODO DE TESTE RÁPIDO ATIVADO. AS CONFIGURAÇÕES FORAM REDUZIDAS.")
        logger.warning("="*50)

    # ... o resto da função main continua como antes ...
    logger.info(f"Dispositivo selecionado: {ARGS.device}")
    # ...






















///////////////////////////////////////////////////////////////
import torch
import torch.nn as nn
# ... (outras importações inalteradas)
import s3fs # <-- Adicionar esta importação
import shutil # <-- Adicionar para gerenciar pastas temporárias

# --- Funções e Classes do Modelo (Inalteradas) ---
# Todas as classes do modelo, de ArticleStyleBERTDataset a PretrainingTrainer
# permanecem como na última versão. Para manter a resposta limpa,
# o foco será nas funções que foram alteradas.
# ... (Cole aqui as classes ArticleStyleBERTDataset, ArticleBERT, PretrainingTrainer, etc.)

# --- CORREÇÃO: Funções de Checkpoint com suporte direto a S3 ---

def save_checkpoint(args, shard_num, model, optimizer, scheduler, best_val_loss):
    """Salva um checkpoint completo, funcionando em caminhos locais ou S3."""
    checkpoint_path_str = f"{args.checkpoint_dir.rstrip('/')}/latest_checkpoint.pth"
    best_model_path_str = f"{args.output_dir.rstrip('/')}/best_model.pth"
    logger = logging.getLogger(__name__)

    state = {
        'shard_num': shard_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }

    is_s3 = checkpoint_path_str.startswith("s3://")
    
    try:
        if is_s3:
            s3 = s3fs.S3FileSystem()
            logger.info(f"Salvando checkpoint diretamente no S3: {checkpoint_path_str}")
            with s3.open(checkpoint_path_str, 'wb') as f:
                torch.save(state, f)
        else:
            # Comportamento antigo para caminhos locais
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_path_str)

        logger.info(f"Checkpoint salvo. Shard {shard_num} concluído.")

        if best_val_loss < getattr(save_checkpoint, "global_best_val_loss", float('inf')):
            save_checkpoint.global_best_val_loss = best_val_loss
            logger.info(f"*** Nova melhor validação global encontrada ({best_val_loss:.4f}). Salvando melhor modelo... ***")
            if is_s3:
                with s3.open(best_model_path_str, 'wb') as f:
                    torch.save(model.state_dict(), f)
            else:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path_str)
            logger.info(f"Melhor modelo salvo em: {best_model_path_str}")

    except Exception as e:
        logger.error(f"Falha ao salvar checkpoint em '{checkpoint_path_str}': {e}")

# Inicializa o atributo estático
save_checkpoint.global_best_val_loss = float('inf')


def load_checkpoint(args, model, optimizer, scheduler):
    """Carrega o último checkpoint, funcionando com caminhos locais ou S3."""
    checkpoint_path_str = f"{args.checkpoint_dir.rstrip('/')}/latest_checkpoint.pth"
    start_shard = 0
    logger = logging.getLogger(__name__)
    is_s3 = checkpoint_path_str.startswith("s3://")
    
    try:
        # Verifica a existência do arquivo
        if is_s3:
            s3 = s3fs.S3FileSystem()
            if not s3.exists(checkpoint_path_str):
                logger.info("Nenhum checkpoint encontrado no S3. Iniciando do zero.")
                return start_shard
        else:
            if not Path(checkpoint_path_str).exists():
                logger.info("Nenhum checkpoint local encontrado. Iniciando do zero.")
                return start_shard

        logger.info(f"Carregando checkpoint de: {checkpoint_path_str}")
        
        # Carrega o arquivo
        if is_s3:
            with s3.open(checkpoint_path_str, 'rb') as f:
                checkpoint = torch.load(f, map_location=args.device)
        else:
            checkpoint = torch.load(checkpoint_path_str, map_location=args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        save_checkpoint.global_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_shard = checkpoint.get('shard_num', -1) + 1
        
        logger.info(f"Checkpoint carregado. Resumindo do Shard {start_shard}. Melhor Val Loss global: {save_checkpoint.global_best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Não foi possível carregar o checkpoint de '{checkpoint_path_str}'. Iniciando do zero. Erro: {e}")
        start_shard = 0

    return start_shard


# --- CORREÇÃO: Lógica do Tokenizer para lidar com outputs no S3 ---
def setup_and_train_tokenizer(args, logger):
    """
    Prepara e treina o tokenizador. Gerencia I/O local e S3 de forma robusta.
    """
    logger.info("--- Fase: Preparação do Tokenizador ---")

    # Define os caminhos. Se o output for S3, usamos um diretório temporário local.
    is_s3_output = args.output_dir.startswith("s3://")
    if is_s3_output:
        # Usa um diretório temporário que será limpo no final
        local_tokenizer_path = Path("./temp_tokenizer_assets")
        s3_tokenizer_path = f"{args.output_dir.rstrip('/')}/tokenizer_assets/"
        s3 = s3fs.S3FileSystem()
    else:
        # Se for local, trabalha diretamente no diretório final
        local_tokenizer_path = Path(args.output_dir) / "tokenizer_assets"
        s3, s3_tokenizer_path = None, None

    # Limpa o diretório temporário local, caso tenha sobrado de uma execução anterior
    if is_s3_output and local_tokenizer_path.exists():
        shutil.rmtree(local_tokenizer_path)
    local_tokenizer_path.mkdir(parents=True, exist_ok=True)
    
    # Verifica se o tokenizador já existe no destino final (local ou S3)
    final_destination_exists = s3.exists(s3_tokenizer_path) if is_s3_output else local_tokenizer_path.exists() and any(local_tokenizer_path.iterdir())

    if final_destination_exists:
        logger.info(f"Tokenizador já existe no destino final.")
        if is_s3_output:
            logger.info(f"Baixando de {s3_tokenizer_path} para {local_tokenizer_path}")
            s3.get(s3_tokenizer_path, str(local_tokenizer_path), recursive=True)
    else:
        logger.info("Tokenizador não encontrado no destino. Gerando agora...")
        
        # Carrega os dados para treinar o tokenizador
        base_data_path = args.s3_data_path.rstrip('/')
        glob_data_path = f"{base_data_path}/batch_*.jsonl" if "batch_*.jsonl" not in base_data_path else base_data_path
        s3_data_client = s3fs.S3FileSystem()
        all_files = sorted(s3_data_client.glob(glob_data_path))
        
        if not all_files:
            raise RuntimeError(f"Nenhum arquivo encontrado em {glob_data_path} para treinar o tokenizador.")
            
        files_for_tokenizer = all_files[:args.files_per_shard_tokenizer]
        logger.info(f"Usando os primeiros {len(files_for_tokenizer)} arquivos para treinar o tokenizador.")
        full_path_files = [f"s3://{f}" if not f.startswith('s3://') else f for f in files_for_tokenizer]
        
        tokenizer_ds = datasets.load_dataset("json", data_files=full_path_files, split="train")
        sentences_for_tokenizer = [ex['text'] for ex in tokenizer_ds if ex and ex.get('text')]
        
        # Escreve sentenças em um arquivo de texto temporário
        temp_text_file = local_tokenizer_path / "temp_corpus.txt"
        with open(temp_text_file, "w", encoding="utf-8") as f:
            for s_line in sentences_for_tokenizer: f.write(s_line + "\n")
            
        # Treina e salva os arquivos do tokenizador no diretório local
        wp_trainer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
        wp_trainer.train(
            files=[str(temp_text_file)],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency_tokenizer,
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        )
        wp_trainer.save_model(str(local_tokenizer_path))
        temp_text_file.unlink() # Deleta o arquivo de corpus que pode ser grande

        # Se o destino for S3, faz o upload dos arquivos recém-criados
        if is_s3_output:
            logger.info(f"Fazendo upload do novo tokenizador para {s3_tokenizer_path}")
            s3.put(str(local_tokenizer_path), s3_tokenizer_path, recursive=True)

    # Carrega o tokenizador a partir do caminho LOCAL (seja ele o final ou o temporário)
    logger.info(f"Carregando modelo do tokenizador do caminho local: {local_tokenizer_path}")
    tokenizer = BertTokenizer.from_pretrained(str(local_tokenizer_path))
    pad_id = tokenizer.pad_token_id

    # Se usamos um diretório temporário, limpa no final
    if is_s3_output:
        shutil.rmtree(local_tokenizer_path)

    logger.info("Tokenizador preparado com sucesso.")
    return tokenizer, pad_id
def main():
    ARGS = parse_args()
    
    # --- MODIFICAÇÃO: Só cria diretórios se os caminhos forem locais ---
    if not ARGS.output_dir.startswith("s3://"):
        Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
    if not ARGS.checkpoint_dir.startswith("s3://"):
        Path(ARGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # O log SEMPRE será salvo localmente para evitar problemas de I/O
    local_log_dir = Path("./training_logs")
    local_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = local_log_dir / f'training_log_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    setup_logging(ARGS.log_level, str(log_file))
    logger = logging.getLogger(__name__)

    logger.info(f"Dispositivo selecionado: {ARGS.device}")
    logger.info("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items():
        logger.info(f"{arg_name}: {value}")
    logger.info("---------------------------------")
    
    # 1. Prepara o tokenizador (cria e usa seu próprio stream temporário)
    # Esta etapa é robusta e lida com S3
    tokenizer, pad_id = setup_and_train_tokenizer(ARGS, logger)
    
    # 2. Inicia o loop de treinamento principal
    # Esta função agora contém a lógica de checkpoint correta
    run_pretraining_on_shards(ARGS, tokenizer, pad_id, logger)
    
    logger.info("--- Pipeline de Pré-treinamento Finalizado com Sucesso ---")

if __name__ == "__main__":
    # Garanta que todas as definições de classe e função estejam acima desta linha
    main()

//////////////////////////////////////////////////////////////////////////////////

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import datasets
import tokenizers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast as BertTokenizer
from tqdm.auto import tqdm
import random
from pathlib import Path
import sys
import math
import numpy as np
import argparse
import os
import logging
import datetime
import s3fs

# --- Função para Configurar Logging ---
def setup_logging(log_level_str, log_file_path_str):
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int): raise ValueError(f'Nível de log inválido: {log_level_str}')
    Path(log_file_path_str).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(name)s:%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_file_path_str), logging.StreamHandler(sys.stdout)])

# --- Definições de Classes do Modelo BERT (Inalteradas) ---
class ArticleStyleBERTDataset(Dataset):
    def __init__(self, corpus_sents_list, tokenizer_instance, seq_len_config):
        self.tokenizer, self.seq_len = tokenizer_instance, seq_len_config
        self.corpus_sents = [s for s in corpus_sents_list if s and s.strip()]
        self.corpus_len = len(self.corpus_sents)
        if self.corpus_len < 2: raise ValueError("Corpus precisa de pelo menos 2 sentenças.")
        self.cls_id, self.sep_id, self.pad_id, self.mask_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
    def __len__(self): return self.corpus_len
    def _get_sentence_pair_for_nsp(self, sent_a_idx):
        sent_a, is_next = self.corpus_sents[sent_a_idx], 0
        if random.random() < 0.5 and sent_a_idx + 1 < self.corpus_len:
            sent_b, is_next = self.corpus_sents[sent_a_idx + 1], 1
        else:
            rand_sent_b_idx = random.randrange(self.corpus_len)
            while self.corpus_len > 1 and rand_sent_b_idx == sent_a_idx: rand_sent_b_idx = random.randrange(self.corpus_len)
            sent_b = self.corpus_sents[rand_sent_b_idx]
        return sent_a, sent_b, is_next
    def _apply_mlm_to_tokens(self, token_ids_list):
        inputs, labels = list(token_ids_list), list(token_ids_list)
        for i, token_id in enumerate(inputs):
            if token_id in [self.cls_id, self.sep_id, self.pad_id]: labels[i] = self.pad_id; continue
            if random.random() < 0.15:
                action_prob = random.random()
                if action_prob < 0.8: inputs[i] = self.mask_id
                elif action_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
            else: labels[i] = self.pad_id
        return inputs, labels
    def __getitem__(self, idx):
        sent_a_str, sent_b_str, nsp_label = self._get_sentence_pair_for_nsp(idx)
        tokens_a_ids = self.tokenizer.encode(sent_a_str, add_special_tokens=False, truncation=True, max_length=self.seq_len - 3)
        tokens_b_ids = self.tokenizer.encode(sent_b_str, add_special_tokens=False, truncation=True, max_length=self.seq_len - len(tokens_a_ids) - 3)
        masked_tokens_a_ids, mlm_labels_a_ids = self._apply_mlm_to_tokens(tokens_a_ids)
        masked_tokens_b_ids, mlm_labels_b_ids = self._apply_mlm_to_tokens(tokens_b_ids)
        input_ids = [self.cls_id] + masked_tokens_a_ids + [self.sep_id] + masked_tokens_b_ids + [self.sep_id]
        mlm_labels = [self.pad_id] + mlm_labels_a_ids + [self.pad_id] + mlm_labels_b_ids + [self.pad_id]
        segment_ids = ([0] * (len(masked_tokens_a_ids) + 2)) + ([1] * (len(masked_tokens_b_ids) + 1))
        current_len = len(input_ids)
        if current_len > self.seq_len: input_ids, mlm_labels, segment_ids = input_ids[:self.seq_len], mlm_labels[:self.seq_len], segment_ids[:self.seq_len]
        padding_len = self.seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids.extend([self.pad_id] * padding_len); mlm_labels.extend([self.pad_id] * padding_len); segment_ids.extend([0] * padding_len)
        return {"bert_input": torch.tensor(input_ids), "bert_label": torch.tensor(mlm_labels), "segment_label": torch.tensor(segment_ids), "is_next": torch.tensor(nsp_label), "attention_mask": torch.tensor(attention_mask, dtype=torch.long)}
class ArticlePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len): super().__init__(); pe = torch.zeros(max_len, d_model).float(); pe.requires_grad = False; pos_col = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)); pe[:, 0::2] = torch.sin(pos_col * div_term); pe[:, 1::2] = torch.cos(pos_col * div_term); self.pe = pe.unsqueeze(0)
    def forward(self, x_ids): return self.pe[:, :x_ids.size(1)]
class ArticleBERTEmbedding(nn.Module):
    def __init__(self, vocab_sz, d_model, seq_len, dropout_rate, pad_idx): super().__init__(); self.tok = nn.Embedding(vocab_sz, d_model, padding_idx=pad_idx); self.seg = nn.Embedding(3, d_model, padding_idx=0); self.pos = ArticlePositionalEmbedding(d_model, seq_len); self.drop = nn.Dropout(p=dropout_rate)
    def forward(self, sequence_ids, segment_label_ids): return self.drop(self.tok(sequence_ids) + self.pos(sequence_ids) + self.seg(segment_label_ids))
class ArticleMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate): super().__init__(); assert d_model % num_heads == 0; self.d_k = d_model // num_heads; self.heads = num_heads; self.drop = nn.Dropout(dropout_rate); self.q_lin, self.k_lin, self.v_lin, self.out_lin = [nn.Linear(d_model, d_model) for _ in range(4)]
    def forward(self, q_in, k_in, v_in, mha_mask_for_scores): bs = q_in.size(0); q = self.q_lin(q_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); k = self.k_lin(k_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); v = self.v_lin(v_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k);
        if mha_mask_for_scores is not None: scores = scores.masked_fill(mha_mask_for_scores == 0, -1e9)
        weights = self.drop(F.softmax(scores, dim=-1)); context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bs, -1, self.heads * self.d_k); return self.out_lin(context)
class ArticleFeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_size, dropout_rate): super().__init__(); self.fc1 = nn.Linear(d_model, ff_hidden_size); self.fc2 = nn.Linear(ff_hidden_size, d_model); self.drop = nn.Dropout(dropout_rate); self.activ = nn.GELU()
    def forward(self, x): return self.fc2(self.drop(self.activ(self.fc1(x))))
class ArticleEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_size, dropout_rate): super().__init__(); self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.attn = ArticleMultiHeadedAttention(num_heads, d_model, dropout_rate); self.ff = ArticleFeedForward(d_model, ff_hidden_size, dropout_rate); self.drop = nn.Dropout(dropout_rate)
    def forward(self, embeds, mha_padding_mask): attended = self.attn(embeds, embeds, embeds, mha_padding_mask); x = self.norm1(embeds + self.drop(attended)); ff_out = self.ff(x); return self.norm2(x + self.drop(ff_out))
class ArticleBERT(nn.Module):
    def __init__(self, vocab_sz, d_model, n_layers, heads_config, seq_len_config, pad_idx_config, dropout_rate_config, ff_h_size_config): super().__init__(); self.d_model = d_model; self.emb = ArticleBERTEmbedding(vocab_sz, d_model, seq_len_config, dropout_rate_config, pad_idx_config); self.enc_blocks = nn.ModuleList([ArticleEncoderLayer(d_model, heads_config, ff_h_size_config, dropout_rate_config) for _ in range(n_layers)])
    def forward(self, input_ids, segment_ids, attention_mask): mha_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2); x = self.emb(input_ids, segment_ids);
        for block in self.enc_blocks: x = block(x, mha_padding_mask)
        return x
class ArticleNSPHead(nn.Module):
    def __init__(self, hidden_d_model): super().__init__(); self.linear = nn.Linear(hidden_d_model, 2); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out[:, 0]))
class ArticleMLMHead(nn.Module):
    def __init__(self, hidden_d_model, vocab_sz): super().__init__(); self.linear = nn.Linear(hidden_d_model, vocab_sz); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out))
class ArticleBERTLMWithHeads(nn.Module):
    def __init__(self, bert_model, vocab_size): super().__init__(); self.bert = bert_model; self.nsp_head = ArticleNSPHead(self.bert.d_model); self.mlm_head = ArticleMLMHead(self.bert.d_model, vocab_size)
    def forward(self, input_ids, segment_ids, attention_mask): bert_output = self.bert(input_ids, segment_ids, attention_mask); return self.nsp_head(bert_output), self.mlm_head(bert_output)
class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps): self._optimizer = optimizer; self.n_warmup_steps = n_warmup_steps; self.n_current_steps = 0; self.init_lr = np.power(d_model, -0.5)
    def step_and_update_lr(self): self._update_learning_rate(); self._optimizer.step()
    def zero_grad(self): self._optimizer.zero_grad()
    def _get_lr_scale(self):
        if self.n_current_steps == 0: return 0.0
        val1 = np.power(self.n_current_steps, -0.5)
        if self.n_warmup_steps > 0: val2 = np.power(self.n_warmup_steps, -1.5) * self.n_current_steps; return np.minimum(val1, val2)
        return val1
    def _update_learning_rate(self): self.n_current_steps += 1; lr = self.init_lr * self._get_lr_scale();
        for pg in self._optimizer.param_groups: pg['lr'] = lr
    def state_dict(self): return {'n_current_steps': self.n_current_steps}
    def load_state_dict(self, state_dict): self.n_current_steps = state_dict['n_current_steps']

# --- CORREÇÃO: Trainer SIMPLIFICADO. A lógica de checkpoint foi movida para fora. ---
class PretrainingTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer_schedule, device, pad_idx_mlm_loss, vocab_size, log_freq=100):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dev = device
        self.model = model.to(self.dev)
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.opt_schedule = optimizer_schedule
        self.crit_mlm = nn.NLLLoss(ignore_index=pad_idx_mlm_loss)
        self.crit_nsp = nn.NLLLoss()
        self.log_freq = log_freq
        self.vocab_size = vocab_size

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training)
        dl = self.train_dl if is_training else self.val_dl
        if not dl: return float('inf'), 0.0
        
        total_loss_ep, tot_nsp_ok, tot_nsp_el = 0.0, 0, 0
        mode = "Train" if is_training else "Val"
        desc = f"Epoch {epoch_num+1} [{mode}]"
        data_iter = tqdm(dl, desc=desc, file=sys.stdout)

        for i_batch, data in enumerate(data_iter):
            data = {k: v.to(self.dev) for k, v in data.items()}
            nsp_out, mlm_out = self.model(data["bert_input"], data["segment_label"], data["attention_mask"])
            loss_nsp = self.crit_nsp(nsp_out, data["is_next"])
            loss_mlm = self.crit_mlm(mlm_out.view(-1, self.vocab_size), data["bert_label"].view(-1))
            loss = loss_nsp + loss_mlm

            if is_training:
                self.opt_schedule.zero_grad()
                loss.backward()
                self.opt_schedule.step_and_update_lr()
            
            total_loss_ep += loss.item()
            nsp_preds = nsp_out.argmax(dim=-1)
            tot_nsp_ok += (nsp_preds == data["is_next"]).sum().item()
            tot_nsp_el += data["is_next"].nelement()

            if (i_batch + 1) % self.log_freq == 0:
                lr = self.opt_schedule._optimizer.param_groups[0]['lr']
                data_iter.set_postfix({"L":f"{total_loss_ep/(i_batch+1):.3f}", "NSP_Acc":f"{tot_nsp_ok/tot_nsp_el*100:.2f}%", "LR":f"{lr:.2e}"})
        
        avg_total_l = total_loss_ep / len(dl) if len(dl) > 0 else 0
        final_nsp_acc = tot_nsp_ok * 100.0 / tot_nsp_el if tot_nsp_el > 0 else 0
        self.logger.info(f"{desc} - AvgTotalL: {avg_total_l:.4f}, NSP Acc: {final_nsp_acc:.2f}%")
        return avg_total_l, final_nsp_acc

    def train(self, num_epochs):
        self.logger.info(f"Iniciando treinamento neste shard por {num_epochs} época(s).")
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_loss = float('inf')
            if self.val_dl:
                with torch.no_grad():
                    val_loss, _ = self._run_epoch(epoch, is_training=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        return best_val_loss

# --- CORREÇÃO: Funções de Checkpoint centralizadas e cientes de shards ---
def save_checkpoint(args, shard_num, model, optimizer, scheduler, best_val_loss):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"

    state = {
        'shard_num': shard_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint salvo. Shard {shard_num} concluído.")

    if best_val_loss < getattr(save_checkpoint, "global_best_val_loss", float('inf')):
        save_checkpoint.global_best_val_loss = best_val_loss
        best_model_path = Path(args.output_dir) / "best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"*** Nova melhor validação global encontrada ({best_val_loss:.4f}). Modelo salvo em {best_model_path} ***")

# Inicializa o atributo estático
save_checkpoint.global_best_val_loss = float('inf')

def load_checkpoint(args, model, optimizer, scheduler):
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pth"
    start_shard = 0
    if not checkpoint_path.exists():
        logging.info("Nenhum checkpoint encontrado. Iniciando do zero.")
        return start_shard

    logging.info(f"Carregando checkpoint de: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    save_checkpoint.global_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    start_shard = checkpoint.get('shard_num', -1) + 1
    
    logging.info(f"Checkpoint carregado. Resumindo do Shard {start_shard}. Melhor Val Loss global: {save_checkpoint.global_best_val_loss:.4f}")
    return start_shard

# --- Funções do Pipeline ---
def setup_and_train_tokenizer(args, logger):
    logger.info("--- Fase: Preparação do Tokenizador ---")
    base_data_path = args.s3_data_path.rstrip('/')
    glob_data_path = f"{base_data_path}/batch_*.jsonl" if "batch_*.jsonl" not in base_data_path else base_data_path
    
    logger.info(f"Usando os primeiros {args.files_per_shard_tokenizer} arquivos para treinar o tokenizador...")
    s3 = s3fs.S3FileSystem()
    all_files = sorted(s3.glob(glob_data_path))
    if not all_files: raise RuntimeError(f"Nenhum arquivo encontrado em {glob_data_path}")
        
    files_for_tokenizer = all_files[:args.files_per_shard_tokenizer]
    full_path_files = [f"s3://{f}" if not f.startswith('s3://') else f for f in files_for_tokenizer]
    
    tokenizer_ds = datasets.load_dataset("json", data_files=full_path_files, split="train")
    sentences_for_tokenizer = [ex['text'] for ex in tokenizer_ds if ex and ex.get('text')]
    
    temp_file = Path(args.output_dir) / "temp_for_tokenizer.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for s_line in sentences_for_tokenizer: f.write(s_line + "\n")
    
    TOKENIZER_ASSETS_DIR = Path(args.output_dir) / "tokenizer_assets"
    TOKENIZER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not (TOKENIZER_ASSETS_DIR / "vocab.txt").exists():
        logger.info("Treinando novo tokenizador...")
        wp_trainer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
        wp_trainer.train(files=[str(temp_file)], vocab_size=args.vocab_size, min_frequency=args.min_frequency_tokenizer, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
        wp_trainer.save_model(str(TOKENIZER_ASSETS_DIR))
    else: logger.info(f"Tokenizador já existe em '{TOKENIZER_ASSETS_DIR}'.")
    
    tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_ASSETS_DIR), local_files_only=True)
    logger.info("Tokenizador preparado com sucesso.")
    return tokenizer, tokenizer.pad_token_id

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    logger.info("--- Fase: Pré-Treinamento em Shards ---")
    
    base_data_path = args.s3_data_path.rstrip('/')
    glob_data_path = f"{base_data_path}/batch_*.jsonl" if "batch_*.jsonl" not in base_data_path else base_data_path
    s3 = s3fs.S3FileSystem(); all_files = sorted(s3.glob(glob_data_path))
    if not all_files: logger.error(f"Nenhum arquivo de dados encontrado em '{glob_data_path}'."); return
    
    num_files_per_shard = args.files_per_shard_training
    file_shards = [all_files[i:i + num_files_per_shard] for i in range(0, len(all_files), num_files_per_shard)]
    logger.info(f"Encontrados {len(all_files)} arquivos de dados, divididos em {len(file_shards)} shards de treinamento.")

    # Instancia o modelo e otimizadores FORA do loop para manter o estado
    model = ArticleBERTLMWithHeads(ArticleBERT(vocab_sz=tokenizer.vocab_size, d_model=args.model_d_model, n_layers=args.model_n_layers, heads_config=args.model_heads, seq_len_config=args.max_len, pad_idx_config=pad_id, dropout_rate_config=args.model_dropout_prob, ff_h_size_config=args.model_d_model * 4), tokenizer.vocab_size)
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = ScheduledOptim(optimizer, args.model_d_model, args.warmup_steps)
    
    # Carrega o estado do checkpoint, se existir. Retorna o shard de onde começar.
    start_shard = load_checkpoint(args, model, optimizer, scheduler)
    
    for shard_num in range(start_shard, len(file_shards)):
        file_list_for_shard = file_shards[shard_num]
        logger.info(f"--- Processando Shard {shard_num + 1}/{len(file_shards)} ---")
        
        full_path_files = [f"s3://{f}" if not f.startswith('s3://') else f for f in file_list_for_shard]
        shard_ds = datasets.load_dataset("json", data_files=full_path_files, split="train")
        sentences_list = [ex['text'] for ex in shard_ds if ex and ex.get('text')]
        if not sentences_list: logger.warning(f"Shard {shard_num + 1} vazio. Pulando."); continue
        
        val_split = int(len(sentences_list) * 0.1)
        train_sents, val_sents = sentences_list[val_split:], sentences_list[:val_split]
        train_dataset = ArticleStyleBERTDataset(train_sents, tokenizer, args.max_len)
        val_dataset = ArticleStyleBERTDataset(val_sents, tokenizer, args.max_len) if val_sents else None
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size_pretrain, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size_pretrain, shuffle=False) if val_dataset else None
        
        trainer = PretrainingTrainer(model, train_dl, val_dl, scheduler, args.device, pad_id, tokenizer.vocab_size, args.logging_steps)
        best_loss_in_shard = trainer.train(num_epochs=args.epochs_per_shard)
        
        # Salva o checkpoint no final de cada shard
        save_checkpoint(args, shard_num, model, optimizer, scheduler, best_loss_in_shard)

    logger.info("--- PROCESSO DE TREINAMENTO EM SHARDS DE ARQUIVOS CONCLUÍDO ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Script autônomo e robusto de Pré-treino BERT com Shards e Checkpoints.")
    parser.add_argument("--s3_data_path", type=str, required=True, help="Caminho para o DIRETÓRIO S3/local contendo os arquivos batch_*.jsonl.")
    parser.add_argument("--output_dir", type=str, default="./bert_outputs", help="Diretório para salvar outputs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Diretório para salvar checkpoints.")
    parser.add_argument("--files_per_shard_training", type=int, default=10, help="Número de arquivos .jsonl a processar em cada shard de treinamento.")
    parser.add_argument("--files_per_shard_tokenizer", type=int, default=5, help="Número de arquivos .jsonl a usar para treinar o tokenizador.")
    parser.add_argument("--epochs_per_shard", type=int, default=1, help="Número de épocas para treinar em CADA shard.")
    parser.add_argument("--batch_size_pretrain", type=int, default=32)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2)
    parser.add_argument("--model_d_model", type=int, default=768)
    parser.add_argument("--model_n_layers", type=int, default=12)
    parser.add_argument("--model_heads", type=int, default=12)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--logging_steps', type=int, default=100)
    args = parser.parse_args()
    if args.device is None: args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def main():
    ARGS = parse_args()
    Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
    Path(ARGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(ARGS.output_dir) / f'training_log_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    setup_logging(ARGS.log_level, str(log_file))
    logger = logging.getLogger(__name__)

    logger.info(f"Dispositivo selecionado: {ARGS.device}")
    logger.info("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items(): logger.info(f"{arg_name}: {value}")
    logger.info("---------------------------------")
    
    tokenizer, pad_id = setup_and_train_tokenizer(ARGS, logger)
    run_pretraining_on_shards(ARGS, tokenizer, pad_id, logger)
    
    logger.info("--- Pipeline de Pré-treinamento Finalizado ---")

if __name__ == "__main__":
    main()
