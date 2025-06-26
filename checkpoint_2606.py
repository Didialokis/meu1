Ótima pergunta. Esta é uma capacidade essencial para qualquer treinamento longo e caro, conhecida como checkpointing. A ideia é salvar periodicamente o estado completo do seu treinamento para que, em caso de falha (seja por um erro no código, um problema na instância do SageMaker ou por você usar "instâncias spot" que podem ser interrompidas), você possa reiniciar o job e ele continuará exatamente de onde parou.

Para implementar isso, precisamos salvar mais do que apenas os pesos do modelo. Um checkpoint completo inclui:

O estado do modelo (model.state_dict())
O estado do otimizador (optimizer.state_dict()), que inclui informações como os momentos do Adam.
O estado do agendador de learning rate (no seu caso, o n_current_steps do ScheduledOptim).
A época atual e o passo (batch) atual.
A melhor pontuação de validação até o momento para garantir que a lógica de "melhor modelo" continue funcionando.
Vamos implementar isso em duas partes: primeiro no script de treino e depois no script lançador.

Parte 1: Modificações no Script de Treino (train_article_style_pretraining.py)
1. Modificar a Classe ScheduledOptim para Salvar e Carregar seu Estado

Precisamos que nosso agendador customizado possa ser salvo e carregado.

Python

class ScheduledOptim(): # Agendador de LR do artigo
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer; self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0; self.init_lr = np.power(d_model, -0.5)

    # ... (código existente da classe) ...

    # --- MODIFICAÇÃO 1: Adicionar métodos de estado ---
    def state_dict(self):
        """Salva o estado do agendador."""
        return {'n_current_steps': self.n_current_steps}

    def load_state_dict(self, state_dict):
        """Carrega o estado do agendador."""
        self.n_current_steps = state_dict['n_current_steps']
    # ----------------------------------------------------
2. Adicionar Lógica de Salvar e Carregar Checkpoints no PretrainingTrainer

Esta é a mudança principal. Vamos adicionar funções para gerenciar os checkpoints.

Python

class PretrainingTrainer:
    def __init__(self, ..., log_freq=10, checkpoint_dir='/opt/ml/checkpoints'): # <-- Adicionar checkpoint_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dev = device; self.model = model.to(self.dev)
        # ... (inicialização existente)
        self.crit_nsp = nn.NLLLoss()
        
        # --- MODIFICAÇÃO 2: Adicionar caminhos de checkpoint ---
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"
        # -----------------------------------------------------

        self.log_freq, self.save_path = log_freq, Path(model_save_path)
        self.best_val_loss = float('inf')
        self.vocab_size = vocab_size

    def _save_checkpoint(self, epoch, is_best=False):
        """Salva um checkpoint completo do estado do treinamento."""
        # Se estiver usando DataParallel, acesse o modelo com .module
        model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) or 'DistributedDataParallel' in str(type(self.model)) else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.opt_schedule._optimizer.state_dict(),
            'scheduler_state_dict': self.opt_schedule.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Salva o checkpoint mais recente
        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info(f"Checkpoint salvo em: {self.checkpoint_path} (Época {epoch})")

        # Se for o melhor modelo, salva também no arquivo de melhor modelo
        if is_best:
            torch.save(model_state, self.save_path)
            self.logger.info(f"Novo melhor modelo salvo em: {self.save_path}")

    def _load_checkpoint(self):
        """Carrega o último checkpoint se existir."""
        start_epoch = 0
        if not self.checkpoint_path.exists():
            self.logger.info("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
            return start_epoch

        self.logger.info(f"Carregando checkpoint de: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dev)

        # Carrega o estado do modelo
        model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) or 'DistributedDataParallel' in str(type(self.model)) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Carrega o estado do otimizador e do agendador
        self.opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.opt_schedule.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1  # Começa da próxima época
        
        self.logger.info(f"Checkpoint carregado. Resumindo da época {start_epoch}.")
        self.logger.info(f"Passos do agendador já concluídos: {self.opt_schedule.n_current_steps}")
        return start_epoch

    def train(self, num_epochs):
        """Método de treino modificado para usar checkpoints."""
        
        # --- MODIFICAÇÃO 3: Carregar checkpoint no início ---
        start_epoch = self._load_checkpoint()
        # ----------------------------------------------------

        self.logger.info(f"Iniciando pré-treinamento (MLM+NSP) de {start_epoch} até {num_epochs} épocas.")
        model_saved_this_run = False

        for epoch in range(start_epoch, num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_total_loss_epoch = None
            if self.val_dl:
                with torch.no_grad():
                    val_total_loss_epoch, _, _ = self._run_epoch(epoch, is_training=False)
            
            is_best = False
            if self.val_dl and val_total_loss_epoch is not None and val_total_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_total_loss_epoch
                is_best = True
                model_saved_this_run = True
                self.logger.info(f"Nova melhor Val Total Loss: {self.best_val_loss:.4f}.")
            
            # --- MODIFICAÇÃO 4: Salvar checkpoint a cada época ---
            self._save_checkpoint(epoch, is_best=is_best)
            # -----------------------------------------------------

            # Lógica para salvar na última época se não houver validação
            if not self.val_dl and epoch == num_epochs - 1:
                self.logger.info(f"MLM+NSP sem validação. Salvando modelo da última época ({epoch+1})")
                torch.save(self.model.state_dict(), self.save_path)
                model_saved_this_run = True
            
            self.logger.info("-" * 30)
        
        # ... (resto da função de log) ...
3. Passar o checkpoint_dir na Instanciação do Trainer

Em run_bert_pretraining_nsp_mlm, não precisamos mudar nada, pois o valor padrão /opt/ml/checkpoints é exatamente o que o SageMaker usa.

Parte 2: Modificações no Script Lançador (train.py)
Agora, precisamos dizer ao SageMaker para habilitar o checkpointing para o nosso job.

Python

# Em seu arquivo train.py

# ... (configuração de sessão, role, hiperparâmetros) ...

S3_OUTPUT_PREFIX = "leixdie_treinamento_bert/output"

# --- MODIFICAÇÃO 5: Definir o caminho S3 para os checkpoints ---
s3_checkpoint_path = f"s3://{sagemaker_default_bucket}/{S3_OUTPUT_PREFIX}/checkpoints/"
# ------------------------------------------------------------

huggingface_estimator = HuggingFace(
    entry_point='train_article_style_pretraining.py',
    source_dir='./',
    role=role,
    # ... (configurações de instância, versão, etc.)
    hyperparameters=hyperparameters,
    
    # --- MODIFICAÇÃO 6: Habilitar o checkpointing no Estimator ---
    checkpoint_s3_uri=s3_checkpoint_path,
    checkpoint_local_path='/opt/ml/checkpoints', # O mesmo caminho do script de treino
    # -----------------------------------------------------------

    # --- DICA BÔNUS: Use Instâncias Spot para Economizar até 90% ---
    # Agora que você tem checkpoints, pode usar instâncias spot com segurança!
    # O treinamento será retomado automaticamente se a instância for interrompida.
    use_spot_instances=True,
    max_wait=1800, # Tempo (em seg) para esperar por uma instância spot. 3600s = 1h
    max_run=3600, # Tempo máximo de execução (em seg). 86400s = 24h
    # -----------------------------------------------------------------

    output_path=f"s3://{sagemaker_default_bucket}/{S3_OUTPUT_PREFIX}/job_files/",
    model_dir=f"s3://{sagemaker_default_bucket}/{S3_OUTPUT_PREFIX}/model/"
)

print(f"Iniciando job de treinamento SageMaker com CHECKPOINTING habilitado.")
print(f"Checkpoints serão salvos em: {s3_checkpoint_path}")

# Iniciar o job de treinamento
huggingface_estimator.fit(wait=True) # ou com inputs, se estiver usando



# Como Funciona o Fluxo de Resumo
Primeira Execução:

Você lança o job com o train.py modificado.
O SageMaker configura o job, sabendo que os checkpoints devem ser salvos em s3_checkpoint_path.
Seu script train_article_style_pretraining.py começa a rodar. A função _load_checkpoint não encontra nenhum arquivo em /opt/ml/checkpoints e o treino começa do zero.
No final de cada época, _save_checkpoint salva latest_checkpoint.pth em /opt/ml/checkpoints.
O SageMaker automaticamente sincroniza o conteúdo de /opt/ml/checkpoints com o seu S3 (s3_checkpoint_path).
O Job Falha ou é Interrompido.

Execução para Resumir:

Você executa o mesmo script train.py novamente, sem alterar nada. O SageMaker o identificará como uma tentativa de resumir um job com o mesmo nome ou você pode cloná-lo.
Antes de iniciar seu container de treino, o SageMaker vê que há arquivos no s3_checkpoint_path e os copia para o diretório local /opt/ml/checkpoints da nova instância.
Seu script train_article_style_pretraining.py começa a rodar.
Agora, a função _load_checkpoint encontra o arquivo latest_checkpoint.pth, carrega todos os estados (modelo, otimizador, época) e o treinamento continua exatamente de onde parou.
Essa abordagem torna seu processo de treinamento robusto, seguro e muito mais econômico, especialmente se você aproveitar as instâncias spot.
