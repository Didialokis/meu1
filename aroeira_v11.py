def parse_args():
    parser = argparse.ArgumentParser(description="Script de Pré-treino BERT com Épocas Globais, Shards e Checkpoints.")
    
    # --- Configurações de Dados e Paths ---
    parser.add_argument("--s3_data_path", type=str, required=True, help="Caminho para o DIRETÓRIO S3/local contendo os arquivos batch_*.jsonl.")
    parser.add_argument("--output_dir", type=str, default="./bert_outputs", help="Diretório para salvar outputs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Diretório para salvar checkpoints.")
    
    # --- Configurações de Sharding e Épocas ---
    parser.add_argument("--files_per_shard_training", type=int, default=10, help="Número de arquivos .jsonl a processar em cada shard de treinamento.")
    parser.add_argument("--files_per_shard_tokenizer", type=int, default=5, help="Número de arquivos .jsonl a usar para treinar o tokenizador.")
    
    # --- MODIFICAÇÃO: Argumento para Épocas Globais ---
    parser.add_argument("--num_global_epochs", type=int, default=1, help="Número total de passagens (épocas) sobre o dataset completo.")
    
    # O argumento antigo agora controla as passadas DENTRO de um shard. 1 é o ideal.
    parser.add_argument("--epochs_per_shard", type=int, default=1, help="Número de épocas para treinar em CADA shard. Recomenda-se 1.")
    
    # ... O resto dos argumentos permanece o mesmo ...
    parser.add_argument("--batch_size_pretrain", type=int, default=32)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)
    # ... etc ...

    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

#Essas duas funções precisam ser atualizadas para salvar e carregar o progresso do loop externo /////////////////

# --- MODIFICAÇÃO: Checkpoint agora salva o estado completo do loop ---
def save_checkpoint(args, global_epoch, shard_num, model, optimizer, scheduler, best_val_loss):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"

    state = {
        'global_epoch': global_epoch,
        'shard_num': shard_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'rng_state': random.getstate(), # Salva o estado do embaralhamento de arquivos
    }
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint salvo. Época Global {global_epoch+1}, Shard {shard_num+1} concluído.")

    # A lógica para salvar o melhor modelo permanece a mesma
    if best_val_loss < getattr(save_checkpoint, "global_best_val_loss", float('inf')):
        save_checkpoint.global_best_val_loss = best_val_loss
        best_model_path = Path(args.output_dir) / "best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"*** Nova melhor validação global encontrada ({best_val_loss:.4f}). Modelo salvo em {best_model_path} ***")

# Inicializa o atributo estático
save_checkpoint.global_best_val_loss = float('inf')


def load_checkpoint(args, model, optimizer, scheduler):
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pth"
    start_epoch = 0
    start_shard = 0
    if not checkpoint_path.exists():
        logging.info("Nenhum checkpoint encontrado. Iniciando do zero.")
        return start_epoch, start_shard

    logging.info(f"Carregando checkpoint de: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    save_checkpoint.global_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Carrega o estado do embaralhamento para garantir que a ordem dos arquivos seja a mesma
    random.setstate(checkpoint['rng_state'])
    
    # Retorna a próxima época e o próximo shard a serem processados
    start_epoch = checkpoint.get('global_epoch', 0)
    start_shard = checkpoint.get('shard_num', -1) + 1
    
    logging.info(f"Checkpoint carregado. Resumindo da Época Global {start_epoch + 1}, Shard {start_shard + 1}.")
    return start_epoch, start_shard

#////////////////////////////////////////////////////////////////////

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    logger.info("--- Fase: Pré-Treinamento em Épocas Globais e Shards ---")
    
    # 1. Obter a lista de todos os arquivos de dados do S3
    logger.info("Buscando a lista completa de arquivos de dados...")
    base_data_path = args.s3_data_path.rstrip('/')
    glob_data_path = f"{base_data_path}/batch_*.jsonl" if "batch_*.jsonl" not in base_data_path else base_data_path
    s3 = s3fs.S3FileSystem(); all_files_master_list = sorted(s3.glob(glob_data_path))
    if not all_files_master_list: logger.error(f"Nenhum arquivo de dados encontrado em '{glob_data_path}'."); return
    
    logger.info(f"Encontrados {len(all_files_master_list)} arquivos de dados para processar.")

    # 2. Instanciar modelo e otimizadores FORA do loop para manter o estado entre épocas e shards
    model = ArticleBERTLMWithHeads(...) # (argumentos do modelo)
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = ScheduledOptim(optimizer, args.model_d_model, args.warmup_steps)
    
    # 3. Carregar o estado do último checkpoint, se existir
    start_epoch, start_shard = load_checkpoint(args, model, optimizer, scheduler)
    
    # --- MODIFICAÇÃO: Loop de ÉPOCA GLOBAL ---
    for epoch_num in range(start_epoch, args.num_global_epochs):
        logger.info(f"--- INICIANDO ÉPOCA GLOBAL {epoch_num + 1}/{args.num_global_epochs} ---")
        
        # 4. Embaralhar a ordem dos arquivos a cada nova época (exceto na primeira época de um resumo)
        current_files = list(all_files_master_list)
        if start_shard == 0: # Só embaralha se estivermos no início de uma época
            random.shuffle(current_files)
            logger.info("Ordem dos arquivos de dados foi embaralhada para esta época.")
        
        # 5. Criar os shards de arquivos a partir da lista (potencialmente embaralhada)
        num_files_per_shard = args.files_per_shard_training
        file_shards = [current_files[i:i + num_files_per_shard] for i in range(0, len(current_files), num_files_per_shard)]
        
        # 6. Loop INTERNO sobre os shards da época atual
        for shard_num in range(start_shard, len(file_shards)):
            file_list_for_shard = file_shards[shard_num]
            logger.info(f"--- Processando Shard {shard_num + 1}/{len(file_shards)} (Época Global {epoch_num + 1}) ---")
            
            # (O resto da lógica para carregar dados do shard e criar DataLoaders permanece a mesma)
            # ...
            
            # Instancia o Trainer (que é leve e sem estado)
            trainer = PretrainingTrainer(model, train_dl, val_dl, scheduler, args.device, pad_id, tokenizer.vocab_size, args.logging_steps)
            best_loss_in_shard = trainer.train(num_epochs=args.epochs_per_shard)
            
            # Salva o checkpoint no final de cada shard, agora com o estado da época global
            save_checkpoint(args, epoch_num, shard_num, model, optimizer, scheduler, best_loss_in_shard)

        # Reseta o start_shard para 0 para a próxima época global
        start_shard = 0

    logger.info(f"--- {args.num_global_epochs} ÉPOCAS GLOBAIS CONCLUÍDAS ---")
