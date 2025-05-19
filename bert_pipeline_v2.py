# --- Bloco 1: Imports Essenciais ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets
import tokenizers # Para tokenizers.__version__
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm # Para barras de progresso
import random
from pathlib import Path
import sys
import math # Para math.isnan e outras operações se necessário
import argparse
import os

print(f"PyTorch: {torch.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")

# --- Definições de Classes (Globais para o script) ---

class BERTMLMDataset(Dataset):
    """Dataset para Masked Language Modeling."""
    def __init__(self, sentences, tokenizer_instance, max_len_config, pad_token_id_config):
        self.sentences = sentences
        self.tokenizer = tokenizer_instance
        self.max_len = max_len_config
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.mask_id = self.tokenizer.token_to_id("<mask>")
        self.cls_id = self.tokenizer.token_to_id("<s>")
        self.sep_id = self.tokenizer.token_to_id("</s>")
        self.pad_id = pad_token_id_config
    def __len__(self): return len(self.sentences)
    def _mask_tokens(self, ids_orig):
        inputs, labels = list(ids_orig), list(ids_orig)
        for i, token_id in enumerate(inputs):
            if token_id in [self.cls_id, self.sep_id, self.pad_id]: labels[i] = self.pad_id; continue
            if random.random() < 0.15: # 15% dos tokens
                act_prob = random.random()
                if act_prob < 0.8: inputs[i] = self.mask_id
                elif act_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
                # 10% -> Manter Original
            else: labels[i] = self.pad_id # Não prever tokens não mascarados
        return torch.tensor(inputs), torch.tensor(labels)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.sentences[idx]) # Tokenizer já faz padding/truncation
        masked_ids, mlm_labels = self._mask_tokens(enc.ids)
        return {"input_ids": masked_ids, "attention_mask": torch.tensor(enc.attention_mask, dtype=torch.long), 
                "segment_ids": torch.zeros_like(masked_ids, dtype=torch.long), "labels": mlm_labels}

class BERTEmbedding(nn.Module):
    """Camada de Embedding do BERT (Token, Posição, Segmento)."""
    def __init__(self, vocab_size, hidden_size, max_len, dropout_prob, pad_token_id):
        super().__init__(); self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.seg_emb = nn.Embedding(2, hidden_size); self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12); self.drop = nn.Dropout(dropout_prob)
        self.register_buffer("pos_ids", torch.arange(max_len).expand((1, -1)))
    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1); tok_e, seg_e = self.tok_emb(input_ids), self.seg_emb(segment_ids)
        pos_e = self.pos_emb(self.pos_ids[:, :seq_len])
        return self.drop(self.norm(tok_e + pos_e + seg_e))

class BERTBaseModel(nn.Module):
    """Backbone BERT (pilha de Encoders Transformer)."""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_len_config, pad_token_id, dropout_prob):
        super().__init__()
        self.emb = BERTEmbedding(vocab_size, hidden_size, max_len_config, dropout_prob, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size, 
                                             dropout=dropout_prob, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, input_ids, attention_mask, segment_ids):
        x = self.emb(input_ids, segment_ids)
        return self.enc(x, src_key_padding_mask=(attention_mask == 0))

class BERTLM(nn.Module):
    """Modelo BERT com cabeça para Masked Language Modeling."""
    def __init__(self, bert_base: BERTBaseModel, vocab_size, hidden_size):
        super().__init__(); self.bert_base = bert_base
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
    def forward(self, input_ids, attention_mask, segment_ids):
        return self.mlm_head(self.bert_base(input_ids, attention_mask, segment_ids))

class SimplifiedTrainer: # Renomeado de GenericTrainer para clareza
    """Trainer simplificado, focado em MLM para este script."""
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 model_save_path, vocab_size_for_loss, log_freq=20):
        self.model, self.train_dl, self.val_dl = model.to(device), train_dataloader, val_dataloader
        self.opt, self.crit, self.dev = optimizer, criterion, device
        self.save_path = model_save_path
        self.best_val_loss = float('inf') # Para MLM, menor loss é melhor
        self.vocab_size = vocab_size_for_loss # Necessário para reshape do loss de MLM
        self.log_freq = log_freq

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None # Se não houver val_dataloader
        total_loss = 0.0
        desc = f"Epoch {epoch_num+1} [{'Train' if is_training else 'Val'}] (MLM)"
        progress_bar = tqdm(dl, total=len(dl) if hasattr(dl, '__len__') else None, desc=desc, file=sys.stdout)

        for i_batch, batch in enumerate(progress_bar):
            input_ids=batch["input_ids"].to(self.dev); attention_mask=batch["attention_mask"].to(self.dev)
            segment_ids=batch.get("segment_ids",torch.zeros_like(input_ids)).to(self.dev); 
            labels=batch["labels"].to(self.dev) # Espera "labels" do dataset MLM
            
            if is_training: self.opt.zero_grad()
            with torch.set_grad_enabled(is_training):
                logits = self.model(input_ids, attention_mask, segment_ids)
                loss = self.crit(logits.view(-1, self.vocab_size), labels.view(-1)) # Loss para MLM
            
            if is_training: loss.backward(); self.opt.step()
            total_loss += loss.item()
            if (i_batch + 1) % self.log_freq == 0: progress_bar.set_postfix({"loss": f"{total_loss / (i_batch + 1):.4f}"})
        
        avg_loss = total_loss / len(dl) if len(dl) > 0 else 0.0
        print(f"{desc} - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs):
        print(f"Treinando (MLM) por {num_epochs} épocas. Observando 'loss' de validação.")
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_loss = self._run_epoch(epoch, is_training=False) # _run_epoch retorna só o loss agora
            
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"Nova melhor Val Loss (MLM): {self.best_val_loss:.4f}. Salvando modelo: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path)
            elif epoch == num_epochs - 1 and not self.val_dl: # Sem validação, salva na última época
                print(f"(MLM) Sem validação. Salvando modelo da última época: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path)
            print("-" * 30)
        
        best_val_display = "N/A"
        if isinstance(self.best_val_loss,(int,float)):
            if math.isinf(self.best_val_loss):best_val_display="N/A (inf)"
            elif math.isnan(self.best_val_loss):best_val_display="N/A (NaN)"
            else:best_val_display=f"{self.best_val_loss:.4f}"
        print(f"Treinamento (MLM) concluído. Melhor Val Loss: {best_val_display}")
        if not self.val_dl and num_epochs > 0 and Path(self.save_path).exists(): print(f"Modelo salvo em: {self.save_path}")

# --- Funções para cada Fase do Pipeline ---
def setup_data_and_train_tokenizer(args):
    """Carrega dados Aroeira, treina/carrega tokenizador."""
    print("\n--- Fase: Preparação de Dados Aroeira e Tokenizador ---")
    _all_aroeira_sentences = [] # Variável local para esta função
    try:
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        text_col = "text"; collected_ex = []
        src_iter = streamed_ds; desc_tqdm = "Extraindo sentenças Aroeira"; total_tqdm = None
        if args.aroeira_subset_size is not None:
            iterator = iter(streamed_ds); print(f"Coletando {args.aroeira_subset_size} exemplos Aroeira...")
            try:
                for _ in range(args.aroeira_subset_size): collected_ex.append(next(iterator))
            except StopIteration: print(f"Alerta: Stream Aroeira esgotado. Coletados {len(collected_ex)}.")
            src_iter = collected_ex; desc_tqdm += " (subconjunto)"; total_tqdm = len(collected_ex)
        else: desc_tqdm += " (stream completo)"; print("AVISO: Processando stream completo do Aroeira.")
        
        for ex in tqdm(src_iter, total=total_tqdm, desc=desc_tqdm, file=sys.stdout):
            sent = ex.get(text_col)
            if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences.append(sent.strip())
        if not _all_aroeira_sentences: raise ValueError("Nenhuma sentença Aroeira foi extraída.")
        print(f"Total de sentenças Aroeira extraídas: {len(_all_aroeira_sentences)}")

        temp_file = Path(args.temp_tokenizer_train_file)
        if temp_file.exists(): temp_file.unlink()
        with open(temp_file, "w", encoding="utf-8") as f:
            for s_line in _all_aroeira_sentences: f.write(s_line + "\n")
        train_files_list = [str(temp_file)]
        
        vocab_f = Path(args.tokenizer_vocab_filename); merges_f = Path(args.tokenizer_merges_filename)
        if not vocab_f.exists() or not merges_f.exists():
            if not train_files_list or not Path(train_files_list[0]).exists():
                raise FileNotFoundError(f"Arquivo treino tokenizador '{train_files_list[0] if train_files_list else 'N/A'}' não encontrado.")
            base_vocab_name = Path(args.tokenizer_vocab_filename).name
            prefix = base_vocab_name.replace("-vocab.json", "")
            tok_model_bpe = ByteLevelBPETokenizer(lowercase=True)
            print(f"Treinando tokenizador com {train_files_list}...")
            tok_model_bpe.train(files=train_files_list, vocab_size=args.vocab_size, 
                               min_frequency=args.min_frequency_tokenizer, 
                               special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
            tok_model_bpe.save_model(str(Path(args.output_dir)), prefix=prefix)
        else: print("Tokenizador já existe. Carregando...")
        
        loaded_tokenizer = ByteLevelBPETokenizer(vocab=args.tokenizer_vocab_filename, merges=args.tokenizer_merges_filename, lowercase=True)
        loaded_tokenizer._tokenizer.post_processor = BertProcessing(("</s>", loaded_tokenizer.token_to_id("</s>")), ("<s>", loaded_tokenizer.token_to_id("<s>")))
        loaded_tokenizer.enable_truncation(max_length=args.max_len)
        loaded_tokenizer.enable_padding(pad_id=loaded_tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=args.max_len)
        pad_id_val = loaded_tokenizer.token_to_id("<pad>")
        print(f"Vocabulário: {loaded_tokenizer.get_vocab_size()}, PAD ID: {pad_id_val}")
        return loaded_tokenizer, pad_id_val, _all_aroeira_sentences
    except Exception as e_tok: print(f"Erro na preparação de dados/tokenizador: {e_tok}"); import traceback; traceback.print_exc(); raise

def run_mlm_pretrain(args, tokenizer_obj, pad_token_id_val, aroeira_sents_for_pretrain):
    """Configura e executa o pré-treinamento MLM."""
    print("\n--- Fase: Pré-Treinamento MLM ---")
    val_s_mlm_r = 0.1; num_v_mlm = int(len(aroeira_sents_for_pretrain) * val_s_mlm_r)
    if num_v_mlm < 1 and len(aroeira_sents_for_pretrain) > 1: num_v_mlm = 1
    tr_s_mlm = aroeira_sents_for_pretrain[num_v_mlm:]; v_s_mlm = aroeira_sents_for_pretrain[:num_v_mlm]
    if not tr_s_mlm: tr_s_mlm = aroeira_sents_for_pretrain; v_s_mlm = []

    print(f"Sentenças de Treino (MLM): {len(tr_s_mlm)}, Sentenças de Validação (MLM): {len(v_s_mlm)}")
    train_ds_mlm = BERTMLMDataset(tr_s_mlm, tokenizer_obj, args.max_len, pad_token_id_val)
    train_dl_mlm = DataLoader(train_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    val_dl_mlm = None
    if v_s_mlm and len(v_s_mlm) > 0:
        val_ds_mlm = BERTMLMDataset(v_s_mlm, tokenizer_obj, args.max_len, pad_token_id_val)
        if len(val_ds_mlm) > 0: val_dl_mlm = DataLoader(val_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)

    bert_base = BERTBaseModel(
        tokenizer_obj.get_vocab_size(), args.model_hidden_size, args.model_num_layers, 
        args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
        pad_token_id_val, dropout_prob=args.model_dropout_prob
    )
    bertlm_model = BERTLM(bert_base, tokenizer_obj.get_vocab_size(), args.model_hidden_size)
    
    # Usar hiperparâmetros de AdamW do args se disponíveis, senão defaults razoáveis
    opt_betas = (args.adam_beta1 if hasattr(args, 'adam_beta1') else 0.9, 
                 args.adam_beta2 if hasattr(args, 'adam_beta2') else 0.999)
    opt_eps = args.adam_epsilon if hasattr(args, 'adam_epsilon') else 1e-8
    opt_weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.01
    
    opt_mlm = torch.optim.AdamW(bertlm_model.parameters(), lr=args.lr_pretrain, 
                                betas=opt_betas, eps=opt_eps, weight_decay=opt_weight_decay)
    crit_mlm = nn.CrossEntropyLoss(ignore_index=pad_token_id_val)
    
    trainer_mlm_instance = SimplifiedTrainer( # Usando SimplifiedTrainer
        bertlm_model, train_dl_mlm, val_dl_mlm, opt_mlm, crit_mlm, args.device, 
        args.pretrained_bertlm_save_filename, tokenizer_obj.get_vocab_size(), log_freq=args.logging_steps
    )
    print("Pré-treinamento MLM configurado. Iniciando...")
    trainer_mlm_instance.train(num_epochs=args.epochs_pretrain)

# --- Função Principal e Parseador de Argumentos ---
def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino BERT no Aroeira.")
    # Configs Gerais e de Dados
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aroeira_subset_size", type=int, default=None, help="Exemplos Aroeira. None para completo.")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True)

    # Configs Pré-treinamento MLM
    parser.add_argument("--epochs_pretrain", type=int, default=1)
    parser.add_argument("--batch_size_pretrain", type=int, default=8)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)

    # Arquitetura Modelo BERT
    parser.add_argument("--model_hidden_size", type=int, default=256)
    parser.add_argument("--model_num_layers", type=int, default=2)
    parser.add_argument("--model_num_attention_heads", type=int, default=4)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    
    # Paths (nomes base, diretório de saída será adicionado)
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './bert_mlm_outputs'))
    parser.add_argument("--tokenizer_vocab_filename", type=str, default="aroeira_mlm_tokenizer-vocab.json")
    parser.add_argument("--tokenizer_merges_filename", type=str, default="aroeira_mlm_tokenizer-merges.txt")
    parser.add_argument("--pretrained_bertlm_save_filename", type=str, default="aroeira_bertlm_pretrained.pth")
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_tokenizer.txt")
        
    # Controle de Fluxo
    parser.add_argument("--do_dataprep_tokenizer", action='store_true', help="Executar preparação de dados e treino do tokenizador.")
    parser.add_argument("--do_pretrain", action='store_true', help="Executar pré-treinamento MLM.")
    
    # Argumentos comuns do HuggingFace Trainer (para compatibilidade com dict de hiperparâmetros do SageMaker)
    # Muitos não serão usados diretamente por este script simplificado, mas evitam erro de "unrecognized".
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument('--per_device_train_batch_size', type=int, help="Batch size geral, se não sobrescrito por _pretrain")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, help="Learning rate geral, se não sobrescrito por lr_pretrain")
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_strategy', type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--fp16", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--load_best_model_at_end", type=lambda x: (str(x).lower() == 'true'), default=False)
    # AdamW specific (Trainer uses these)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)


    if custom_args_list: args = parser.parse_args(custom_args_list)
    else: args = parser.parse_args()
    
    if args.device is None: args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_intermediate_size = args.model_hidden_size * 4 # Derivado
            
    output_dir_path = Path(args.output_dir)
    if not str(args.output_dir).startswith("/opt/ml/"):
         output_dir_path.mkdir(parents=True, exist_ok=True)
    
    args.tokenizer_vocab_filename = str(output_dir_path / Path(args.tokenizer_vocab_filename).name)
    args.tokenizer_merges_filename = str(output_dir_path / Path(args.tokenizer_merges_filename).name)
    args.pretrained_bertlm_save_filename = str(output_dir_path / Path(args.pretrained_bertlm_save_filename).name)
    args.temp_tokenizer_train_file = str(output_dir_path / Path(args.temp_tokenizer_train_file).name)
    
    # Priorizar batch_sizes específicos de fase se definidos, senão usar o global
    if args.per_device_train_batch_size is not None: # Se o global foi passado
        args.batch_size_pretrain = args.per_device_train_batch_size
    if args.learning_rate is not None: # Se o global foi passado
        args.lr_pretrain = args.learning_rate
    return args

# --- Função Principal ---
def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)
    
    print("--- Configurações Utilizadas ---")
    for arg_name_main, value_main in vars(ARGS).items(): print(f"{arg_name_main}: {value_main}")
    print("----------------------------------")

    current_tokenizer_obj = None; current_pad_id = None; _aroeira_sents_for_pt = []
    
    # Fase 1: Preparação de Dados Aroeira e Treinamento do Tokenizador
    # Executada se qualquer etapa de treino for solicitada, pois o tokenizador é sempre necessário.
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain:
        current_tokenizer_obj, current_pad_id, _aroeira_sents_for_pt = setup_data_and_train_tokenizer(ARGS)
    else:
        print("Nenhuma ação de preparação de dados ou treinamento foi solicitada. Encerrando.")
        return # Encerra se não houver nada a fazer.

    # Fase 2: Pré-Treinamento MLM
    if ARGS.do_pretrain:
        if not _aroeira_sents_for_pt or not current_tokenizer_obj:
            print("ERRO: Dados Aroeira ou tokenizador não preparados para pré-treinamento. Execute com --do_dataprep_tokenizer.")
        else:
            run_mlm_pretrain(ARGS, current_tokenizer_obj, current_pad_id, _aroeira_sents_for_pt)
    
    print("\n--- Pipeline Simplificado (MLM) Finalizado ---")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()

# --- Como Executar no Jupyter Notebook (Exemplo em uma célula separada) ---
# notebook_args = [
#     "--max_len", "64",
#     "--aroeira_subset_size", "1000", 
#     "--epochs_pretrain", "1",
#     "--batch_size_pretrain", "4", 
#     "--model_hidden_size", "128", 
#     "--model_num_layers", "2",
#     "--model_num_attention_heads", "2",
#     "--output_dir", "./notebook_outputs_mlm_only", 
#     "--do_dataprep_tokenizer", # Para garantir que o tokenizador seja criado/carregado
#     "--do_pretrain"
# ]
# main(notebook_mode_args_list=notebook_args) 
# print("Execução de teste do notebook (MLM-only) concluída.")
