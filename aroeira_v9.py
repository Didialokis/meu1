def setup_and_train_tokenizer(args, logger):
    """
    Prepara e treina o tokenizador. AGORA CRIA SEU PRÓPRIO STREAM para ser independente.
    """
    logger.info("--- Fase: Preparação do Tokenizador ---")
    
    # 1. Cria um stream temporário apenas para o tokenizador
    logger.info(f"Criando stream temporário para tokenizador a partir de: {args.s3_data_path}")
    streamed_ds = datasets.load_dataset("json", data_files=args.s3_data_path, split="train", streaming=True)
    
    # 2. Usa o primeiro shard como amostra
    # A função load_data_shard ainda é útil como auxiliar aqui
    sentences_for_tokenizer = load_data_shard(streamed_ds, args, logger, shard_num=0, stage_name="Tokenizer")

    if not sentences_for_tokenizer:
        raise RuntimeError("Não foi possível carregar o primeiro shard para treinar o tokenizador. Verifique o caminho dos dados.")

    # O resto da função é idêntico
    temp_file = Path(args.output_dir) / "temp_for_tokenizer.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for s_line in sentences_for_tokenizer: f.write(s_line + "\n")
    
    TOKENIZER_ASSETS_DIR = Path(args.output_dir) / "tokenizer_assets"
    TOKENIZER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not (TOKENIZER_ASSETS_DIR / "vocab.txt").exists():
        logger.info("Treinando novo tokenizador com base no primeiro shard...")
        wp_trainer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
        wp_trainer.train(files=[str(temp_file)], vocab_size=args.vocab_size, min_frequency=args.min_frequency_tokenizer, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
        wp_trainer.save_model(str(TOKENIZER_ASSETS_DIR))
    else:
        logger.info(f"Tokenizador já existe em '{TOKENIZER_ASSETS_DIR}'. Carregando...")

    tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_ASSETS_DIR), local_files_only=True)
    logger.info("Tokenizador preparado com sucesso.")
    return tokenizer, tokenizer.pad_token_id











3. run_pretraining_on_shards (agora cria seu próprio stream)

Esta é a mudança crucial. A função de treinamento principal agora inicializa seu próprio stream, garantindo que ele comece do zero.

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    """Orquestra o processo de treinamento, iterando sobre múltiplos shards de dados."""
    logger.info("--- INICIANDO PROCESSO DE TREINAMENTO EM SHARDS ---")
    
    # --- MODIFICAÇÃO CRUCIAL: Inicializa um novo stream para a fase de treinamento ---
    logger.info(f"Criando stream principal de treinamento a partir de: {args.s3_data_path}")
    try:
        streamed_ds_training = datasets.load_dataset("json", data_files=args.s3_data_path, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Falha ao inicializar o stream de dados para o treinamento: {e}")
        return
    # ----------------------------------------------------------------------------

    shard_num = 0
    while True:
        if args.num_shards != -1 and shard_num >= args.num_shards:
            logger.info(f"Número definido de shards ({args.num_shards}) processado. Finalizando.")
            break
        
        # Passa o stream de TREINAMENTO para a função de carregar o shard
        sentences_list = load_data_shard(streamed_ds_training, args, logger, shard_num)
        
        if not sentences_list:
            logger.info("Shard vazio encontrado. O dataset foi processado por completo. Finalizando o treinamento.")
            break

        # O resto do loop permanece inalterado...
        val_split = int(len(sentences_list) * 0.1); train_sents, val_sents = sentences_list[val_split:], sentences_list[:val_split]
        # ... (código para criar Dataset, DataLoader, Modelo e Trainer) ...
        # ... (chamada para trainer.train()) ...
        
        shard_num += 1

    logger.info("--- PROCESSO DE TREINAMENTO EM SHARDS CONCLUÍDO ---")
/////////////////////////////////////////////////////////////////////////////////////////////////////

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

# --- Função para Configurar Logging ---
def setup_logging(log_level_str, log_file_path_str):
    """Configura o sistema de logging para salvar em arquivo e exibir no console."""
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Nível de log inválido: {log_level_str}')
    Path(log_file_path_str).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path_str), logging.StreamHandler(sys.stdout)]
    )

# --- Definições de Classes do Modelo BERT ---
class ArticleStyleBERTDataset(Dataset):
    def __init__(self, corpus_sents_list, tokenizer_instance: BertTokenizer, seq_len_config: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = tokenizer_instance
        self.seq_len = seq_len_config
        self.corpus_sents = [s for s in corpus_sents_list if s and s.strip()]
        self.corpus_len = len(self.corpus_sents)
        if self.corpus_len < 2: raise ValueError("Corpus precisa de pelo menos 2 sentenças.")
        self.cls_id, self.sep_id, self.pad_id, self.mask_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
    def __len__(self): return self.corpus_len
    def _get_sentence_pair_for_nsp(self, sent_a_idx):
        sent_a = self.corpus_sents[sent_a_idx]; is_next = 0
        if random.random() < 0.5 and sent_a_idx + 1 < self.corpus_len:
            sent_b = self.corpus_sents[sent_a_idx + 1]; is_next = 1
        else:
            rand_sent_b_idx = random.randrange(self.corpus_len)
            while self.corpus_len > 1 and rand_sent_b_idx == sent_a_idx: rand_sent_b_idx = random.randrange(self.corpus_len)
            sent_b = self.corpus_sents[rand_sent_b_idx]
        return sent_a, sent_b, is_next
    def _apply_mlm_to_tokens(self, token_ids_list: list):
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
    def __init__(self, d_model, max_len):
        super().__init__(); pe = torch.zeros(max_len, d_model).float(); pe.requires_grad = False
        pos_col = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos_col * div_term); pe[:, 1::2] = torch.cos(pos_col * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, x_ids): return self.pe[:, :x_ids.size(1)]

class ArticleBERTEmbedding(nn.Module):
    def __init__(self, vocab_sz, d_model, seq_len, dropout_rate, pad_idx):
        super().__init__(); self.tok = nn.Embedding(vocab_sz, d_model, padding_idx=pad_idx); self.seg = nn.Embedding(3, d_model, padding_idx=0); self.pos = ArticlePositionalEmbedding(d_model, seq_len); self.drop = nn.Dropout(p=dropout_rate)
    def forward(self, sequence_ids, segment_label_ids): return self.drop(self.tok(sequence_ids) + self.pos(sequence_ids) + self.seg(segment_label_ids))

class ArticleMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate):
        super().__init__(); assert d_model % num_heads == 0; self.d_k = d_model // num_heads; self.heads = num_heads; self.drop = nn.Dropout(dropout_rate); self.q_lin, self.k_lin, self.v_lin, self.out_lin = [nn.Linear(d_model, d_model) for _ in range(4)]
    def forward(self, q_in, k_in, v_in, mha_mask_for_scores):
        bs = q_in.size(0); q = self.q_lin(q_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); k = self.k_lin(k_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); v = self.v_lin(v_in).view(bs, -1, self.heads, self.d_k).transpose(1, 2); scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mha_mask_for_scores is not None: scores = scores.masked_fill(mha_mask_for_scores == 0, -1e9)
        weights = self.drop(F.softmax(scores, dim=-1)); context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bs, -1, self.heads * self.d_k); return self.out_lin(context)

class ArticleFeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_size, dropout_rate): super().__init__(); self.fc1 = nn.Linear(d_model, ff_hidden_size); self.fc2 = nn.Linear(ff_hidden_size, d_model); self.drop = nn.Dropout(dropout_rate); self.activ = nn.GELU()
    def forward(self, x): return self.fc2(self.drop(self.activ(self.fc1(x))))

class ArticleEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_size, dropout_rate): super().__init__(); self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.attn = ArticleMultiHeadedAttention(num_heads, d_model, dropout_rate); self.ff = ArticleFeedForward(d_model, ff_hidden_size, dropout_rate); self.drop = nn.Dropout(dropout_rate)
    def forward(self, embeds, mha_padding_mask): attended = self.attn(embeds, embeds, embeds, mha_padding_mask); x = self.norm1(embeds + self.drop(attended)); ff_out = self.ff(x); return self.norm2(x + self.drop(ff_out))

class ArticleBERT(nn.Module):
    def __init__(self, vocab_sz, d_model, n_layers, heads_config, seq_len_config, pad_idx_config, dropout_rate_config, ff_h_size_config):
        super().__init__(); self.d_model = d_model; self.emb = ArticleBERTEmbedding(vocab_sz, d_model, seq_len_config, dropout_rate_config, pad_idx_config); self.enc_blocks = nn.ModuleList([ArticleEncoderLayer(d_model, heads_config, ff_h_size_config, dropout_rate_config) for _ in range(n_layers)])
    def forward(self, input_ids, segment_ids, attention_mask):
        mha_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2); x = self.emb(input_ids, segment_ids)
        for block in self.enc_blocks: x = block(x, mha_padding_mask)
        return x

class ArticleNSPHead(nn.Module):
    def __init__(self, hidden_d_model): super().__init__(); self.linear = nn.Linear(hidden_d_model, 2); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out[:, 0]))

class ArticleMLMHead(nn.Module):
    def __init__(self, hidden_d_model, vocab_sz): super().__init__(); self.linear = nn.Linear(hidden_d_model, vocab_sz); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out))

class ArticleBERTLMWithHeads(nn.Module):
    def __init__(self, bert_model: ArticleBERT, vocab_size: int): super().__init__(); self.bert = bert_model; self.nsp_head = ArticleNSPHead(self.bert.d_model); self.mlm_head = ArticleMLMHead(self.bert.d_model, vocab_size)
    def forward(self, input_ids, segment_ids, attention_mask): bert_output = self.bert(input_ids, segment_ids, attention_mask); return self.nsp_head(bert_output), self.mlm_head(bert_output)

class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer; self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0; self.init_lr = np.power(d_model, -0.5)
    def step_and_update_lr(self): self._update_learning_rate(); self._optimizer.step()
    def zero_grad(self): self._optimizer.zero_grad()
    def _get_lr_scale(self):
        if self.n_current_steps == 0: return 0.0
        val1 = np.power(self.n_current_steps, -0.5)
        if self.n_warmup_steps > 0:
            val2 = np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
            return np.minimum(val1, val2)
        return val1
    def _update_learning_rate(self):
        self.n_current_steps += 1; lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups: param_group['lr'] = lr
    def state_dict(self): return {'n_current_steps': self.n_current_steps}
    def load_state_dict(self, state_dict): self.n_current_steps = state_dict['n_current_steps']

class PretrainingTrainer:
    def __init__(self, model: ArticleBERTLMWithHeads, train_dataloader, val_dataloader,
                 d_model_for_optim: int, lr: float, betas: tuple, weight_decay: float,
                 warmup_steps: int, device, model_save_path, pad_idx_mlm_loss: int, vocab_size: int,
                 log_freq=10, checkpoint_dir='./checkpoints'):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dev = device; self.model = model.to(self.dev)
        self.train_dl, self.val_dl = train_dataloader, val_dataloader
        self.opt = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_schedule = ScheduledOptim(self.opt, d_model_for_optim, warmup_steps)
        self.crit_mlm = nn.NLLLoss(ignore_index=pad_idx_mlm_loss); self.crit_nsp = nn.NLLLoss()
        self.log_freq = log_freq; self.best_model_path = Path(model_save_path); self.vocab_size = vocab_size; self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"

    def _save_checkpoint(self, epoch, is_best):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.opt_schedule._optimizer.state_dict(), 'scheduler_state_dict': self.opt_schedule.state_dict(), 'best_val_loss': self.best_val_loss}
        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info(f"Checkpoint salvo em: {self.checkpoint_path} (Época {epoch})")
        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
            self.logger.info(f"Novo melhor modelo salvo em: {self.best_model_path}")

    def _load_checkpoint(self):
        start_epoch = 0
        if not self.checkpoint_path.exists():
            self.logger.info("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
            return start_epoch
        self.logger.info(f"Carregando checkpoint de: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dev)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.opt_schedule.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', -1) + 1
        self.logger.info(f"Checkpoint carregado. Resumindo da época {start_epoch} (passos do agendador: {self.opt_schedule.n_current_steps}).")
        return start_epoch

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None, 0.0, 0
        total_loss_ep, total_mlm_l_ep, total_nsp_l_ep = 0.0, 0.0, 0.0
        tot_nsp_ok, tot_nsp_el = 0, 0
        mode = "Train" if is_training else "Val"; desc = f"Epoch {epoch_num+1} [{mode}]"
        data_iter = tqdm(dl, desc=desc, file=sys.stdout)
        for i_batch, data in enumerate(data_iter):
            data = {k: v.to(self.dev) for k, v in data.items()}
            nsp_out, mlm_out = self.model(data["bert_input"], data["segment_label"], data["attention_mask"])
            loss_nsp = self.crit_nsp(nsp_out, data["is_next"]); loss_mlm = self.crit_mlm(mlm_out.view(-1, self.vocab_size), data["bert_label"].view(-1)); loss = loss_nsp + loss_mlm
            if is_training: self.opt_schedule.zero_grad(); loss.backward(); self.opt_schedule.step_and_update_lr()
            total_loss_ep += loss.item(); total_mlm_l_ep += loss_mlm.item(); total_nsp_l_ep += loss_nsp.item()
            nsp_preds = nsp_out.argmax(dim=-1); tot_nsp_ok += (nsp_preds == data["is_next"]).sum().item(); tot_nsp_el += data["is_next"].nelement()
            if (i_batch + 1) % self.log_freq == 0:
                data_iter.set_postfix({"L":f"{total_loss_ep/(i_batch+1):.3f}", "MLM_L":f"{total_mlm_l_ep/(i_batch+1):.3f}", "NSP_L":f"{total_nsp_l_ep/(i_batch+1):.3f}", "NSP_Acc":f"{tot_nsp_ok/tot_nsp_el*100:.2f}%", "LR":f"{self.opt_schedule._optimizer.param_groups[0]['lr']:.2e}"})
        avg_total_l = total_loss_ep/len(dl) if len(dl)>0 else 0; avg_mlm_l = total_mlm_l_ep/len(dl) if len(dl)>0 else 0; avg_nsp_l = total_nsp_l_ep/len(dl) if len(dl)>0 else 0; final_nsp_acc = tot_nsp_ok*100.0/tot_nsp_el if tot_nsp_el>0 else 0
        self.logger.info(f"{desc} - AvgTotalL: {avg_total_l:.4f}, AvgMLML: {avg_mlm_l:.4f}, AvgNSPL: {avg_nsp_l:.4f}, NSP Acc: {final_nsp_acc:.2f}%")
        return avg_total_l, final_nsp_acc, tot_nsp_el

    def train(self, num_epochs):
        start_epoch = self._load_checkpoint()
        self.logger.info(f"Iniciando/resumindo pré-treinamento de {start_epoch} até {num_epochs} épocas.")
        for epoch in range(start_epoch, num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_loss = float('inf')
            if self.val_dl:
                with torch.no_grad(): val_loss, _, _ = self._run_epoch(epoch, is_training=False)
            is_best = val_loss < self.best_val_loss
            if is_best: self.best_val_loss = val_loss; self.logger.info(f"Nova melhor Val Loss: {self.best_val_loss:.4f}.")
            self._save_checkpoint(epoch, is_best=is_best)
            self.logger.info("-" * 30)
        self.logger.info(f"Treinamento para este shard concluído. Melhor Val Loss: {self.best_val_loss:.4f}")

# --- Funções do Pipeline ---
def load_data_shard(args, logger, shard_num: int):
    """Carrega um fragmento específico (shard) do dataset para a memória."""
    logger.info(f"--- Carregando Shard de Dados Nº {shard_num + 1} ---")
    streamed_ds = datasets.load_dataset("json", data_files=args.s3_data_path, split="train", streaming=True)
    records_to_skip = shard_num * args.shard_size
    if records_to_skip > 0: logger.info(f"Pulando {records_to_skip} registros para chegar ao shard atual...")
    shard_iterator = streamed_ds.skip(records_to_skip).take(args.shard_size)
    shard_sents_list = []
    for ex in tqdm(shard_iterator, desc=f"Processando Shard {shard_num + 1}", total=args.shard_size):
        sent = ex.get("text")
        if isinstance(sent, str) and sent.strip(): shard_sents_list.append(sent.strip())
    if not shard_sents_list: logger.warning(f"Nenhuma sentença carregada para o shard {shard_num + 1}. Fim do dataset?")
    return shard_sents_list

def setup_and_train_tokenizer(args, logger):
    """Prepara os dados e treina o tokenizador usando o PRIMEIRO shard como amostra."""
    logger.info("--- Fase: Preparação do Tokenizador ---")
    sentences_for_tokenizer = load_data_shard(args, logger, shard_num=0)
    if not sentences_for_tokenizer: raise RuntimeError("Não foi possível carregar o primeiro shard para treinar o tokenizador.")
    temp_file = Path(args.output_dir) / "temp_for_tokenizer.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for s_line in sentences_for_tokenizer: f.write(s_line + "\n")
    TOKENIZER_ASSETS_DIR = Path(args.output_dir) / "tokenizer_assets"
    TOKENIZER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not (TOKENIZER_ASSETS_DIR / "vocab.txt").exists():
        logger.info("Treinando novo tokenizador com base no primeiro shard...")
        wp_trainer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
        wp_trainer.train(files=[str(temp_file)], vocab_size=args.vocab_size, min_frequency=args.min_frequency_tokenizer, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
        wp_trainer.save_model(str(TOKENIZER_ASSETS_DIR))
    else:
        logger.info(f"Tokenizador já existe em '{TOKENIZER_ASSETS_DIR}'. Carregando...")
    tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_ASSETS_DIR), local_files_only=True)
    logger.info("Tokenizador preparado com sucesso.")
    return tokenizer, tokenizer.pad_token_id

def run_pretraining_on_shards(args, tokenizer, pad_id, logger):
    """Orquestra o processo de treinamento, iterando sobre múltiplos shards de dados."""
    logger.info("--- INICIANDO PROCESSO DE TREINAMENTO EM SHARDS ---")
    shard_num = 0
    while True:
        if args.num_shards != -1 and shard_num >= args.num_shards:
            logger.info(f"Número definido de shards ({args.num_shards}) processado. Finalizando.")
            break
        
        sentences_list = load_data_shard(args, logger, shard_num)
        if not sentences_list:
            logger.info("Shard vazio encontrado. O dataset foi processado por completo. Finalizando o treinamento.")
            break

        val_split = int(len(sentences_list) * 0.1); train_sents, val_sents = sentences_list[val_split:], sentences_list[:val_split]
        train_dataset = ArticleStyleBERTDataset(train_sents, tokenizer, args.max_len)
        val_dataset = ArticleStyleBERTDataset(val_sents, tokenizer, args.max_len) if val_sents else None
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size_pretrain, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size_pretrain, shuffle=False) if val_dataset else None

        bert_backbone = ArticleBERT(vocab_sz=tokenizer.vocab_size, d_model=args.model_d_model, n_layers=args.model_n_layers, heads_config=args.model_heads, seq_len_config=args.max_len, pad_idx_config=pad_id, dropout_rate_config=args.model_dropout_prob, ff_h_size_config=args.model_d_model * 4)
        bertlm_model = ArticleBERTLMWithHeads(bert_backbone, tokenizer.vocab_size)
        
        trainer = PretrainingTrainer(model=bertlm_model, train_dataloader=train_dl, val_dataloader=val_dl, d_model_for_optim=args.model_d_model, lr=args.lr_pretrain, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=args.warmup_steps, device=args.device, model_save_path=Path(args.output_dir) / "best_model.pth", pad_idx_mlm_loss=pad_id, vocab_size=tokenizer.vocab_size, log_freq=args.logging_steps, checkpoint_dir=args.checkpoint_dir)
        
        trainer.train(num_epochs=args.epochs_per_shard)
        shard_num += 1
    logger.info("--- PROCESSO DE TREINAMENTO EM SHARDS CONCLUÍDO ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Script de Pré-treino BERT com Treinamento em Shards e Checkpointing.")
    parser.add_argument("--s3_data_path", type=str, required=True, help="Caminho S3 ou LOCAL para o arquivo de dados JSONL.")
    parser.add_argument("--output_dir", type=str, default="./bert_outputs", help="Diretório para salvar outputs (tokenizador, modelo final).")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Diretório para salvar os checkpoints do treinamento.")
    parser.add_argument("--shard_size", type=int, default=200000, help="Tamanho de cada fragmento de dados a ser carregado na memória.")
    parser.add_argument("--num_shards", type=int, default=-1, help="Número de shards a processar. Defina como -1 para processar o dataset inteiro.")
    parser.add_argument("--epochs_per_shard", type=int, default=1, help="Número de épocas para treinar em CADA shard. Recomenda-se 1.")
    parser.add_argument("--batch_size_pretrain", type=int, default=32)
    parser.add_argument("--lr_pretrain", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default=None, help="Forçar dispositivo ('cuda' ou 'cpu'). Padrão: auto-detectar.")
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





//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


python train_bert_sharded.py \
    --s3_data_path "caminho/para/seu/arquivo.jsonl" \
    --shard_size 250000 \
    --num_shards -1 \
    --output_dir "./meu_modelo_bert_final" \
    --checkpoint_dir "./meus_checkpoints_final" \
    --batch_size_pretrain 32 \
    --epochs_per_shard 1


//////////////////////////////
      teste


      python train_bert_sharded.py \
    --s3_data_path "caminho/para/seu/arquivo.jsonl" \
    --shard_size 50000 \
    --num_shards 5 \
    --output_dir "./teste_bert" \
    --checkpoint_dir "./teste_checkpoints" \
    --batch_size_pretrain 16 \
    --epochs_per_shard 2
/////////////////////////////////////////////


        def main():
    ARGS = parse_args()
    
    Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
    Path(ARGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(ARGS.output_dir) / f'training_log_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    setup_logging(ARGS.log_level, str(log_file))
    logger = logging.getLogger(__name__)

    logger.info(f"Dispositivo selecionado: {ARGS.device}")
    logger.info("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items():
        logger.info(f"{arg_name}: {value}")
    logger.info("---------------------------------")
    
    # --- MODIFICAÇÃO: Inicializa o stream de dados UMA ÚNICA VEZ ---
    logger.info(f"Inicializando stream de dados a partir de: {ARGS.s3_data_path}")
    try:
        streamed_ds = datasets.load_dataset("json", data_files=ARGS.s3_data_path, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Falha ao inicializar o stream de dados: {e}")
        logger.error("Verifique se o caminho do arquivo está correto e se as permissões de acesso (ex: s3fs) estão configuradas.")
        sys.exit(1) # Encerra o script se não conseguir acessar os dados
    # ------------------------------------------------------------

    # 1. Prepara o tokenizador usando o stream principal
    tokenizer, pad_id = setup_and_train_tokenizer(streamed_ds, ARGS, logger)
    
    # 2. Inicia o loop de treinamento sobre todos os shards, usando o mesmo stream
    run_pretraining_on_shards(streamed_ds, ARGS, tokenizer, pad_id, logger)
    
    logger.info("--- Pipeline de Pré-treinamento Finalizado ---")

if __name__ == "__main__":
    # O parse_args() e todas as outras classes (modelo, trainer) devem estar definidas antes desta linha
    main()
