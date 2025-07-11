
Com certeza\! Esta é uma excelente pergunta e um passo crucial para entender e melhorar o desempenho do seu modelo. Vamos dividir a resposta em duas partes.

### Parte 1: Métricas de Desempenho Registradas Atualmente

No estado atual do código, o `PretrainingTrainer` registra as seguintes métricas ao final de cada "época" (que no nosso caso, é uma passagem sobre um único shard):

As métricas são calculadas tanto para o conjunto de **Treino** quanto para o de **Validação** (se houver dados de validação no shard). Elas são impressas no seu arquivo de log (`training_log.log`).

  * **`AvgTotalL` (Loss Total Média):** A métrica de perda principal, que é a soma da perda de MLM e NSP. **Este é o valor mais importante para acompanhar; você quer que ele diminua com o tempo.**
  * **`AvgMLML` (Loss de Masked Language Modeling Média):** Mede quão bem o modelo está aprendendo a prever as palavras mascaradas.
  * **`AvgNSPL` (Loss de Next Sentence Prediction Média):** Mede quão bem o modelo está aprendendo a relação entre duas sentenças.
  * **`NSP Acc` (Acurácia de Next Sentence Prediction):** A porcentagem de vezes que o modelo acertou se a segunda sentença era a continuação da primeira. **Você quer que esta métrica aumente.**
  * **`LR` (Learning Rate):** A taxa de aprendizado atual, que muda a cada passo de otimização.

Essas métricas são ótimas, mas estão "presas" em um arquivo de texto. Para visualizá-las, precisamos de um método mais estruturado.

-----

### Parte 2: Como Gerar Gráficos com o Progresso

Para gerar gráficos, o primeiro passo é salvar essas métricas em um formato fácil de ler, como um arquivo CSV. Depois, podemos usar um script simples com `pandas` e `matplotlib` para plotar os resultados.

A seguir estão as modificações necessárias no código e um script de plotagem.

#### Modificação 1: Salvar as Métricas em um Arquivo CSV

Vamos modificar a classe `PretrainingTrainer` para que, além de imprimir os logs, ela salve as métricas de cada shard em um arquivo `metrics.csv`.

**No seu arquivo `train_standalone_bert.py`, encontre e substitua a classe `PretrainingTrainer` por esta versão atualizada:**

```python
import csv # Adicione esta importação no topo do seu arquivo

# ... (outras classes do modelo) ...

class PretrainingTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer_schedule, device, pad_idx_mlm_loss, vocab_size, log_freq=100, metrics_log_path='metrics.csv'):
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
        
        # --- MODIFICAÇÃO: Configurar o arquivo CSV de métricas ---
        self.metrics_log_path = metrics_log_path
        self._setup_metrics_log()

    def _setup_metrics_log(self):
        """Cria o arquivo CSV de métricas com o cabeçalho, se não existir."""
        if not Path(self.metrics_log_path).exists():
            with open(self.metrics_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['global_epoch', 'shard_num', 'mode', 'avg_loss', 'avg_mlm_loss', 'avg_nsp_loss', 'nsp_accuracy'])

    def _log_metrics_to_csv(self, global_epoch, shard_num, mode, metrics):
        """Adiciona uma nova linha de métricas ao arquivo CSV."""
        with open(self.metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                global_epoch,
                shard_num,
                mode,
                metrics['loss'],
                metrics['mlm_loss'],
                metrics['nsp_loss'],
                metrics['nsp_acc']
            ])

    def _run_epoch(self, epoch_num, is_training):
        # ... (a lógica interna de _run_epoch permanece a mesma)
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return float('inf'), 0.0, None
        total_loss_ep, total_mlm_l_ep, total_nsp_l_ep, tot_nsp_ok, tot_nsp_el = 0.0, 0.0, 0.0, 0, 0
        mode = "Train" if is_training else "Val"; desc = f"Epoch {epoch_num+1} [{mode}]"
        data_iter = tqdm(dl, desc=desc, file=sys.stdout)
        for i_batch, data in enumerate(data_iter):
            # ... (cálculo de loss e backpropagation) ...
        
        # Coleta as métricas finais da época/shard
        avg_total_l = total_loss_ep / len(dl) if len(dl) > 0 else 0
        avg_mlm_l = total_mlm_l_ep / len(dl) if len(dl) > 0 else 0
        avg_nsp_l = total_nsp_l_ep / len(dl) if len(dl) > 0 else 0
        final_nsp_acc = tot_nsp_ok * 100.0 / tot_nsp_el if tot_nsp_el > 0 else 0
        
        self.logger.info(f"{desc} - AvgTotalL: {avg_total_l:.4f}, NSP Acc: {final_nsp_acc:.2f}%")
        
        metrics_dict = {
            'loss': avg_total_l,
            'mlm_loss': avg_mlm_l,
            'nsp_loss': avg_nsp_l,
            'nsp_acc': final_nsp_acc
        }
        return avg_total_l, final_nsp_acc, metrics_dict

    def train(self, num_epochs, global_epoch, shard_num): # <-- MODIFICAÇÃO: Recebe o estado global
        self.logger.info(f"Iniciando treinamento neste shard por {num_epochs} época(s).")
        best_val_loss_in_shard = float('inf')
        for epoch in range(num_epochs):
            _, _, train_metrics = self._run_epoch(epoch, is_training=True)
            self._log_metrics_to_csv(global_epoch, shard_num, 'train', train_metrics)
            
            val_loss = float('inf')
            if self.val_dl:
                with torch.no_grad():
                    val_loss, _, val_metrics = self._run_epoch(epoch, is_training=False)
                self._log_metrics_to_csv(global_epoch, shard_num, 'validation', val_metrics)

            if val_loss < best_val_loss_in_shard:
                best_val_loss_in_shard = val_loss
        return best_val_loss_in_shard
```

**Modificação 2: Atualize a Chamada do `Trainer`**

Na função `run_pretraining_on_shards`, você precisa passar os novos parâmetros para o `Trainer`.

```python
def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    # ... (código de busca de arquivos e instanciação de modelo/optimizer/scheduler) ...
    
    # Loop de ÉPOCA GLOBAL
    for epoch_num in range(start_epoch, args.num_global_epochs):
        # ... (código do loop de shards) ...
        for shard_num in range(start_shard, len(file_shards)):
            # ... (código para carregar shard e criar DataLoaders) ...

            # --- MODIFICAÇÃO: Passe o caminho do CSV de métricas ---
            metrics_csv_path = Path(args.output_dir) / "training_metrics.csv"
            trainer = PretrainingTrainer(
                model, train_dl, val_dl, scheduler, args.device, pad_id, 
                tokenizer.vocab_size, args.logging_steps,
                metrics_log_path=metrics_csv_path
            )
            
            # --- MODIFICAÇÃO: Passe o estado global para o método train ---
            best_loss_in_shard = trainer.train(
                num_epochs=args.epochs_per_shard,
                global_epoch=epoch_num,
                shard_num=shard_num
            )
            
            # ... (código de salvar checkpoint) ...
```

#### Modificação 3: Script para Gerar os Gráficos

Agora que você terá um arquivo `training_metrics.csv` sendo gerado no seu `--output_dir`, crie um novo arquivo Python chamado `plot_metrics.py` e cole o código abaixo.

**Arquivo `plot_metrics.py`:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_metrics(csv_path):
    """
    Lê o arquivo de métricas CSV e gera gráficos de Loss e Acurácia.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{csv_path}'.")
        return

    # Para o eixo X, podemos usar uma combinação de época e shard para ver o progresso contínuo
    df['global_step'] = df['global_epoch'] + df['shard_num'] / df['shard_num'].max() if df['shard_num'].max() > 0 else df['global_epoch']
    
    # Separa os dados de treino e validação
    train_df = df[df['mode'] == 'train']
    val_df = df[df['mode'] == 'validation']

    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: Loss Total (Treino vs Validação) ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=train_df, x='global_step', y='avg_loss', label='Loss de Treino')
    if not val_df.empty:
        sns.lineplot(data=val_df, x='global_step', y='avg_loss', label='Loss de Validação')
    plt.title('Loss Total vs. Progresso do Treinamento')
    plt.xlabel('Época Global / Progresso do Shard')
    plt.ylabel('Loss Média')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    print("Gráfico 'training_loss_plot.png' salvo.")

    # --- Gráfico 2: Acurácia NSP (Treino vs Validação) ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=train_df, x='global_step', y='nsp_accuracy', label='Acurácia NSP de Treino')
    if not val_df.empty:
        sns.lineplot(data=val_df, x='global_step', y='nsp_accuracy', label='Acurácia NSP de Validação')
    plt.title('Acurácia NSP vs. Progresso do Treinamento')
    plt.xlabel('Época Global / Progresso do Shard')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.savefig('nsp_accuracy_plot.png')
    print("Gráfico 'nsp_accuracy_plot.png' salvo.")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera gráficos a partir do log de métricas do treinamento.")
    parser.add_argument("csv_path", type=str, help="Caminho para o arquivo training_metrics.csv gerado pelo script de treino.")
    args = parser.parse_args()
    
    plot_metrics(args.csv_path)
```

### Como Usar o Novo Fluxo

1.  **Execute o Treinamento:** Rode seu script de treino `train_standalone_bert.py` normalmente. Conforme ele roda, um arquivo `training_metrics.csv` será criado e atualizado no seu diretório de output.
2.  **Instale as Dependências de Plotagem:**
    ```bash
    pip install pandas matplotlib seaborn
    ```
3.  **Execute o Script de Plotagem:** Após o treinamento (ou mesmo durante, para ver o progresso), execute o script de plotagem, passando o caminho para o seu arquivo de métricas.
    ```bash
    python plot_metrics.py ./bert_outputs/training_metrics.csv
    ```

Isso irá exibir os gráficos na sua tela e salvar os arquivos `training_loss_plot.png` e `nsp_accuracy_plot.png` no mesmo diretório, lhe dando uma visão clara e objetiva do desempenho do seu modelo ao longo do tempo.















//////////////////////////////////////////////////////////////
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
