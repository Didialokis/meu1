Parte 1: Modificações no Código train_article_style_pretraining.py
Você precisará fazer as seguintes alterações para tornar seu script compatível com o SMDDP.

1. Adicionar as Importações Necessárias
No topo do seu arquivo, adicione as importações da biblioteca smdistributed e do DistributedSampler do PyTorch.

Python

import torch
# ... (outras importações)
import os # Já presente
import logging # Já presente

# --- ADICIONAR IMPORTAÇÕES PARA SMDDP ---
try:
    import smdistributed.dataparallel.torch.torch_smddp as smd
    from torch.utils.data.distributed import DistributedSampler
    SMD_DATAPARALLEL_AVAILABLE = True
except ImportError:
    SMD_DATAPARALLEL_AVAILABLE = False
# ------------------------------------
Colocar dentro de um try...except é uma boa prática para que o código ainda possa rodar em um ambiente sem a biblioteca instalada.

2. Inicializar o Processo Distribuído no main()
O SMDDP precisa ser inicializado. A função main é o lugar perfeito para isso. Também obteremos o "rank" (ID do processo) e o "world size" (número total de processos).

Python

def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)

    # --- INICIALIZAR SMDDP ---
    if SMD_DATAPARALLEL_AVAILABLE and ARGS.smdistributed_enabled:
        smd.init_process_group()
        ARGS.world_size = smd.get_world_size()
        ARGS.rank = smd.get_rank()
        ARGS.local_rank = smd.get_local_rank()
        # Define o dispositivo para o rank local (GPU específica)
        ARGS.device = ARGS.local_rank
        torch.cuda.set_device(ARGS.local_rank)
    else:
        # Comportamento padrão sem treinamento distribuído
        ARGS.world_size = 1
        ARGS.rank = 0
        ARGS.local_rank = 0
    # -------------------------

    # Apenas o processo principal (rank 0) deve configurar o logging
    if ARGS.rank == 0:
        setup_logging(ARGS.log_level, ARGS.log_file_path)
    
    logger = logging.getLogger(__name__)
    
    # ... (resto da função main)
    # Apenas o processo principal (rank 0) deve logar as configurações
    if ARGS.rank == 0:
        logger.info(f"PyTorch: {torch.__version__}")
        # ... (resto dos logs)
3. Adicionar Argumentos no parse_args()
Adicione um argumento para habilitar/desabilitar a lógica de distribuição.

Python

def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(...)
    # ... (outros argumentos)

    # --- ADICIONAR ARGUMENTO PARA SMDDP ---
    parser.add_argument("--smdistributed_enabled", type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help="Habilita a lógica de treinamento distribuído SMDDP.")
    # ------------------------------------
    
    known_args, _ = parser.parse_known_args(...)
    # ...
    # A definição do device agora é tratada na função main
    if known_args.device is None:
        known_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ...
    return known_args
4. Usar DistributedSampler no run_bert_pretraining_nsp_mlm()
O DataLoader precisa saber como dividir os dados entre as GPUs. O DistributedSampler faz exatamente isso.

Python

def run_bert_pretraining_nsp_mlm(args, tokenizer_instance: BertTokenizer, pad_token_id_val, all_sentences_list, logger):
    # ... (divisão de treino/validação)
    
    train_ds_pt = ArticleStyleBERTDataset(tr_s_pt, tokenizer_instance, args.max_len)
    if len(train_ds_pt)==0 and len(tr_s_pt)>0: logger.error("Dataset de treino MLM+NSP vazio."); return

    # --- CRIAR SAMPLER DISTRIBUÍDO ---
    train_sampler = None
    shuffle_dl = True
    if SMD_DATAPARALLEL_AVAILABLE and args.smdistributed_enabled:
        train_sampler = DistributedSampler(
            train_ds_pt, num_replicas=args.world_size, rank=args.rank
        )
        shuffle_dl = False # O sampler já embaralha os dados
    # ------------------------------------
    
    # Use shuffle_dl e train_sampler aqui
    train_dl_pt = DataLoader(train_ds_pt, batch_size=args.batch_size_pretrain, shuffle=shuffle_dl, num_workers=0, sampler=train_sampler)
    
    val_dl_pt = None
    if v_s_pt and len(v_s_pt) > 0:
        val_ds_pt = ArticleStyleBERTDataset(v_s_pt, tokenizer_instance, args.max_len)
        # O sampler para validação não é estritamente necessário, mas é uma boa prática
        val_sampler = None
        if SMD_DATAPARALLEL_AVAILABLE and args.smdistributed_enabled:
            val_sampler = DistributedSampler(val_ds_pt, num_replicas=args.world_size, rank=args.rank)
        
        if len(val_ds_pt) > 0: 
            val_dl_pt = DataLoader(val_ds_pt, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0, sampler=val_sampler)

    # ... (criação do modelo)
    bertlm_nsp_model = ArticleBERTLMWithHeads(bert_article_backbone, tokenizer_instance.vocab_size)
    
    # --- MOVER MODELO PARA GPU E ENVOLVER COM SMDDP ---
    bertlm_nsp_model = bertlm_nsp_model.to(args.device)
    if SMD_DATAPARALLEL_AVAILABLE and args.smdistributed_enabled:
        bertlm_nsp_model = smd.DistributedDataParallel(bertlm_nsp_model)
    # --------------------------------------------------

    trainer_pt_instance = PretrainingTrainer(
        model=bertlm_nsp_model, # Passe o modelo já envolvido
        # ... (resto dos argumentos)
    )
    # ...
5. Modificar o PretrainingTrainer para Evitar Ações Duplicadas
Apenas o processo principal (rank == 0) deve salvar o modelo e exibir a barra de progresso tqdm.

Python

class PretrainingTrainer:
    def __init__(self, ...):
        # ...
        # Adicione o rank aos atributos da classe para fácil acesso
        self.rank = smd.get_rank() if SMD_DATAPARALLEL_AVAILABLE and smd.is_initialized() else 0
        
    def _run_epoch(self, epoch_num, is_training):
        # ...
        mode = "Train" if is_training else "Val"; desc = f"Epoch {epoch_num+1} [{mode}] (MLM+NSP)"
        
        # --- APENAS RANK 0 MOSTRA TQDM ---
        data_iter = dl
        if self.rank == 0:
            data_iter = tqdm(dl, total=len(dl) if hasattr(dl,'__len__') else None, desc=desc, file=sys.stdout)
        # -----------------------------------
        
        for i_batch, data in enumerate(data_iter):
            # ...
            if self.rank == 0 and (i_batch + 1) % self.log_freq == 0:
                data_iter.set_postfix(...) # Atualiza a barra de progresso
    
    def train(self, num_epochs):
        # ...
        for epoch in range(num_epochs):
            # ...
            # --- APENAS RANK 0 SALVA O MODELO ---
            if self.rank == 0:
                if self.val_dl and val_total_loss_epoch is not None and val_total_loss_epoch < self.best_val_loss:
                    self.best_val_loss = val_total_loss_epoch
                    self.logger.info(f"Nova melhor Val Total Loss: {self.best_val_loss:.4f}. Salvando modelo: {self.save_path}")
                    # Ao salvar, acesse o modelo original dentro do wrapper com .module
                    model_to_save = self.model.module if isinstance(self.model, smd.DistributedDataParallel) else self.model
                    torch.save(model_to_save.state_dict(), self.save_path); model_saved_this_run = True
                elif not self.val_dl and epoch == num_epochs - 1:
                    # ...
                    model_to_save = self.model.module if isinstance(self.model, smd.DistributedDataParallel) else self.model
                    torch.save(model_to_save.state_dict(), self.save_path); model_saved_this_run = True
            # ------------------------------------
Importante: Ao salvar o state_dict de um modelo envolvido por DistributedDataParallel, você precisa acessar o modelo original através de self.model.module.

Parte 2: Como Usar a Configuração ao Lançar o Job
Agora que seu código está pronto, você pode lançar um job de treinamento no SageMaker. A configuração que você mencionou vai no parâmetro distribution do estimador do SageMaker.

Aqui está um exemplo de como seria um script de lançamento usando o SageMaker Python SDK:

Python

import sagemaker
from sagemaker.pytorch import PyTorch

# Inicializa a sessão do SageMaker
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# --- AQUI ESTÁ A CONFIGURAÇÃO DE DISTRIBUIÇÃO ---
distribution_config = {"smdistributed": {"dataparallel": {"enabled": True}}}
# ------------------------------------------------

# Defina os hiperparâmetros para o seu script
hyperparameters = {
    "epochs_pretrain": 10,
    "batch_size_pretrain": 64, # Você pode aumentar o batch size total (64 * número de GPUs)
    "lr_pretrain_adam": 1e-4,
    "max_len": 128,
    "model_d_model": 768,
    "model_n_layers": 6,
    "model_heads": 12,
    # ... outros argumentos do seu script
    "smdistributed_enabled": True # Habilita a lógica no script
}

# Crie o estimador PyTorch
estimator = PyTorch(
    entry_point="train_article_style_pretraining.py", # Seu script
    source_dir="./",                                  # Diretório onde o script está
    role=role,
    instance_count=2,                                 # Ex: 2 máquinas
    instance_type="ml.p3.8xlarge",                    # Ex: tipo de instância com 4 GPUs cada
    framework_version="1.13",                         # Use uma versão compatível com SMDDP
    py_version="py39",
    hyperparameters=hyperparameters,
    distribution=distribution_config                  # Passe a configuração aqui
)

# Inicie o job de treinamento
estimator.fit({"training": "s3://seu-bucket/caminho/para/os/dados"})
Resumo
Modifique seu script para importar smdistributed, inicializá-lo, usar o DistributedSampler, envolver o modelo com smd.DistributedDataParallel, e garantir que apenas o processo principal (rank 0) salve o modelo e os logs.
Use a configuração { "smdistributed": { "dataparallel": { "enabled": True } } } no parâmetro distribution do seu sagemaker.pytorch.PyTorch Estimator ao lançar o job.
Não se esqueça de ativar a lógica no script passando o hiperparâmetro --smdistributed_enabled True.
