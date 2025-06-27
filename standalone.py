


class PretrainingTrainer:
    def __init__(self, model: ArticleBERTLMWithHeads, train_dataloader, val_dataloader,
                 d_model_for_optim: int, lr: float, betas: tuple, weight_decay: float,
                 warmup_steps: int, device, model_save_path, pad_idx_mlm_loss: int, vocab_size: int,
                 log_freq=10, checkpoint_dir='/opt/ml/checkpoints'):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dev = device
        
        # Define se este é o processo principal em um ambiente distribuído
        self.is_main_process = not (SMD_DATAPARALLEL_AVAILABLE and smd.is_initialized()) or smd.get_rank() == 0

        self.model = model
        # O modelo já deve ser movido para o dispositivo ANTES de ser passado para o Trainer
        # self.model = model.to(self.dev) 
        
        self.train_dl, self.val_dl = train_dataloader, val_dataloader
        self.opt = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_schedule = ScheduledOptim(self.opt, d_model_for_optim, warmup_steps)
        self.crit_mlm = nn.NLLLoss(ignore_index=pad_idx_mlm_loss)
        self.crit_nsp = nn.NLLLoss()
        self.log_freq = log_freq
        
        # Caminho para salvar o "melhor" modelo (baseado na validação)
        self.save_path = Path(model_save_path)
        
        # Caminhos para salvar checkpoints para resumo do treinamento
        self.checkpoint_dir = Path(checkpoint_dir)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"

        self.best_val_loss = float('inf')
        self.vocab_size = vocab_size

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Salva um checkpoint completo do estado do treinamento. A variável 'epoch' é passada como argumento."""
        if not self.is_main_process:
            return

        # Para modelos distribuídos, precisamos acessar o modelo original via .module
        model_is_distributed = isinstance(self.model, smd.DistributedDataParallel)
        model_state = self.model.module.state_dict() if model_is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch, # A variável 'epoch' vem do parâmetro da função
            'model_state_dict': model_state,
            'optimizer_state_dict': self.opt_schedule._optimizer.state_dict(),
            'scheduler_state_dict': self.opt_schedule.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info(f"Checkpoint da Época {epoch} salvo em: {self.checkpoint_path}")

        if is_best:
            torch.save(model_state, self.save_path)
            self.logger.info(f"Salvo novo melhor modelo (baseado em Val Loss) em: {self.save_path}")

    def _load_checkpoint(self):
        """Carrega o último checkpoint se existir e retorna a época de início."""
        start_epoch = 0
        if not self.checkpoint_path.exists():
            self.logger.info("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
            return start_epoch

        self.logger.info(f"Carregando checkpoint de: {self.checkpoint_path}")
        # Garante que o checkpoint seja carregado no dispositivo correto (CPU ou GPU)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dev)

        model_is_distributed = isinstance(self.model, smd.DistributedDataParallel)
        model_to_load = self.model.module if model_is_distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.opt_schedule.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Garante que best_val_loss e epoch existam no checkpoint
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        # CORREÇÃO: A próxima época a ser executada é a salva + 1
        start_epoch = checkpoint.get('epoch', -1) + 1
        
        self.logger.info(f"Checkpoint carregado. Resumindo da época {start_epoch}.")
        self.logger.info(f"Passos do agendador já concluídos: {self.opt_schedule.n_current_steps}")
        return start_epoch

    def _run_epoch(self, epoch_num: int, is_training: bool):
        # (Código inalterado)
        self.model.train(is_training)
        dl = self.train_dl if is_training else self.val_dl
        if not dl: return None, 0.0, 0
        if is_training and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch_num)
        total_loss_ep, total_mlm_l_ep, total_nsp_l_ep, tot_nsp_ok, tot_nsp_el = 0.0, 0.0, 0.0, 0, 0
        mode = "Train" if is_training else "Val"
        desc = f"Epoch {epoch_num+1} [{mode}] (MLM+NSP)"
        data_iter = tqdm(dl, desc=desc, file=sys.stdout, disable=not self.is_main_process)
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
            total_mlm_l_ep += loss_mlm.item()
            total_nsp_l_ep += loss_nsp.item()
            nsp_preds = nsp_out.argmax(dim=-1); tot_nsp_ok += (nsp_preds == data["is_next"]).sum().item(); tot_nsp_el += data["is_next"].nelement()
            if self.is_main_process and (i_batch + 1) % self.log_freq == 0:
                data_iter.set_postfix({"L":f"{total_loss_ep/(i_batch+1):.3f}", "MLM_L":f"{total_mlm_l_ep/(i_batch+1):.3f}", "NSP_L":f"{total_nsp_l_ep/(i_batch+1):.3f}", "NSP_Acc":f"{tot_nsp_ok/tot_nsp_el*100:.2f}%", "LR":f"{self.opt_schedule._optimizer.param_groups[0]['lr']:.2e}"})
        avg_total_l = total_loss_ep / len(dl) if len(dl) > 0 else 0; avg_mlm_l = total_mlm_l_ep / len(dl) if len(dl) > 0 else 0
        avg_nsp_l = total_nsp_l_ep / len(dl) if len(dl) > 0 else 0; final_nsp_acc = tot_nsp_ok * 100.0 / tot_nsp_el if tot_nsp_el > 0 else 0
        if self.is_main_process: self.logger.info(f"{desc} - AvgTotalL: {avg_total_l:.4f}, AvgMLML: {avg_mlm_l:.4f}, AvgNSPL: {avg_nsp_l:.4f}, NSP Acc: {final_nsp_acc:.2f}%")
        return avg_total_l, final_nsp_acc, tot_nsp_el


    def train(self, num_epochs: int):
        # Carrega o checkpoint e define a partir de qual época começar
        start_epoch = self._load_checkpoint()

        if self.is_main_process:
            self.logger.info(f"Iniciando/Resumindo pré-treinamento de {start_epoch} até {num_epochs} épocas.")

        # O loop de treino principal
        for epoch in range(start_epoch, num_epochs):
            self._run_epoch(epoch, is_training=True)
            
            val_total_loss_epoch = None
            if self.val_dl:
                with torch.no_grad():
                    val_total_loss_epoch, _, _ = self._run_epoch(epoch, is_training=False)
            
            is_best = False
            # Apenas o processo principal avalia e decide se é o melhor modelo
            if self.is_main_process:
                if self.val_dl and val_total_loss_epoch is not None and val_total_loss_epoch < self.best_val_loss:
                    self.best_val_loss = val_total_loss_epoch
                    is_best = True
                    self.logger.info(f"Nova melhor Val Loss na Época {epoch}: {self.best_val_loss:.4f}.")
            
            # --- CORREÇÃO: A chamada para salvar o checkpoint é feita aqui, DENTRO do loop ---
            # A variável 'epoch' está definida pelo loop 'for' e é passada corretamente.
            self._save_checkpoint(epoch, is_best=is_best)
            
            if self.is_main_process: self.logger.info("-" * 30)
        
        if self.is_main_process: self.logger.info(f"Treinamento concluído após {num_epochs} épocas.")














//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


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
