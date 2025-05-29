# train_article_style_pretraining.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Importante para F.linear
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import datasets
import tokenizers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast as BertTokenizer # Usaremos BertTokenizerFast
from tqdm.auto import tqdm
import random
from pathlib import Path
import sys
import math
import numpy as np # Para ScheduledOptim
import itertools
import argparse
import os
import logging
import datetime

# --- Função para Configurar Logging ---
def setup_logging(log_level_str, log_file_path_str):
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Nível de log inválido: {log_level_str}')
    Path(log_file_path_str).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path_str), logging.StreamHandler(sys.stdout)]
    )

# --- Definições de Classes (Estilo Artigo) ---

class ArticleStyleBERTDataset(Dataset):
    """Dataset para pré-treinamento BERT com MLM e NSP, adaptado do artigo."""
    def __init__(self, corpus_sents_list, tokenizer_instance: BertTokenizer, seq_len_config: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = tokenizer_instance
        self.seq_len = seq_len_config
        self.corpus_sents = [s for s in corpus_sents_list if s and s.strip()]
        self.corpus_len = len(self.corpus_sents)

        if self.corpus_len < 2:
            raise ValueError("Corpus para pré-treinamento (MLM+NSP) precisa de pelo menos 2 sentenças.")

        # IDs de tokens especiais do BertTokenizerFast
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        self.logger.info(f"ArticleStyleBERTDataset inicializado com {self.corpus_len} sentenças. PAD ID: {self.pad_id}")

    def __len__(self): return self.corpus_len

    def _get_sentence_pair_for_nsp(self, sent_a_idx):
        sent_a = self.corpus_sents[sent_a_idx]
        is_next = 0
        if random.random() < 0.5 and sent_a_idx + 1 < self.corpus_len:
            sent_b = self.corpus_sents[sent_a_idx + 1]; is_next = 1
        else:
            rand_sent_b_idx = random.randrange(self.corpus_len)
            while self.corpus_len > 1 and (rand_sent_b_idx == sent_a_idx or rand_sent_b_idx == (sent_a_idx + 1) % self.corpus_len):
                rand_sent_b_idx = random.randrange(self.corpus_len)
            sent_b = self.corpus_sents[rand_sent_b_idx]
        return sent_a, sent_b, is_next

    def _apply_mlm_to_tokens(self, token_ids_list: list): # Renomeado de _random_word_masking_article
        inputs, labels = list(token_ids_list), list(token_ids_list)
        for i, token_id in enumerate(inputs):
            if token_id in [self.cls_id, self.sep_id, self.pad_id, self.mask_id]: 
                labels[i] = self.pad_id; continue
            if random.random() < 0.15:
                action_prob = random.random()
                if action_prob < 0.8: inputs[i] = self.mask_id
                elif action_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
            else: labels[i] = self.pad_id
        return inputs, labels # Retorna listas de IDs

    def __getitem__(self, idx):
        sent_a_str, sent_b_str, nsp_label = self._get_sentence_pair_for_nsp(idx)

        # Tokeniza palavras individuais para aplicar MLM (como no artigo, mas usando encode)
        # Isso é diferente do MLM no ByteLevelBPE que mascara sub-tokens de uma sentença já tokenizada.
        # Para simplificar e alinhar com BertTokenizer, vamos tokenizar as sentenças e depois mascarar.
        
        # Tokeniza as sentenças A e B (sem CLS/SEP aqui, serão adicionados depois)
        tokens_a_ids = self.tokenizer.encode(sent_a_str, add_special_tokens=False, truncation=True, max_length=self.seq_len -3) # Deixa espaço para CLS, SEP, SEP
        tokens_b_ids = self.tokenizer.encode(sent_b_str, add_special_tokens=False, truncation=True, max_length=self.seq_len - len(tokens_a_ids) -3)


        masked_tokens_a_ids, mlm_labels_a_ids = self._apply_mlm_to_tokens(tokens_a_ids)
        masked_tokens_b_ids, mlm_labels_b_ids = self._apply_mlm_to_tokens(tokens_b_ids)

        input_ids = [self.cls_id] + masked_tokens_a_ids + [self.sep_id] + masked_tokens_b_ids + [self.sep_id]
        mlm_labels = [self.pad_id] + mlm_labels_a_ids + [self.pad_id] + mlm_labels_b_ids + [self.pad_id]
        segment_ids = ([0] * (len(masked_tokens_a_ids) + 2)) + ([1] * (len(masked_tokens_b_ids) + 1)) # Segmentos 0 e 1

        # Truncamento e Padding final
        current_len = len(input_ids)
        if current_len > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]
            segment_ids = segment_ids[:self.seq_len]
        
        padding_len = self.seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len # Máscara de atenção
        
        input_ids.extend([self.pad_id] * padding_len)
        mlm_labels.extend([self.pad_id] * padding_len)
        segment_ids.extend([0] * padding_len) # Segmento 0 para padding

        return {"bert_input": torch.tensor(input_ids), "bert_label": torch.tensor(mlm_labels),
                "segment_label": torch.tensor(segment_ids), "is_next": torch.tensor(nsp_label),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)}


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
        super().__init__(); self.tok = nn.Embedding(vocab_sz, d_model, padding_idx=pad_idx)
        # Segmentos: 0 para padding, 1 para SentA (e CLS), 2 para SentB (e SEPs).
        # O artigo usa 3 embeddings de segmento com padding_idx=0.
        # Se passamos segment_ids 0 e 1 para tokens reais, precisamos de nn.Embedding(2,...)
        # Se os segment_ids do ArticleStyleBERTDataset são 0 e 1 para sentenças reais e
        # o padding nos segment_ids é 0, então o embedding de segmento 0 será usado para SentA E padding.
        # Para alinhar com o artigo (segmentos 1 e 2 para sentenças, 0 para padding):
        self.seg = nn.Embedding(3, d_model, padding_idx=0) # PAD=0, SentA=1, SentB=2
        self.pos = ArticlePositionalEmbedding(d_model, seq_len)
        self.drop = nn.Dropout(p=dropout_rate)
    def forward(self, sequence_ids, segment_label_ids): # segment_label_ids vêm do Dataset
        x = self.tok(sequence_ids) + self.pos(sequence_ids) + self.seg(segment_label_ids)
        return self.drop(x)

class ArticleMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate):
        super().__init__(); assert d_model % num_heads == 0; self.d_k = d_model // num_heads
        self.heads = num_heads; self.drop = nn.Dropout(dropout_rate)
        self.q_lin = nn.Linear(d_model,d_model); self.k_lin = nn.Linear(d_model,d_model); self.v_lin = nn.Linear(d_model,d_model)
        self.out_lin = nn.Linear(d_model, d_model)
    def forward(self, q_in, k_in, v_in, mha_mask_for_scores): # mha_mask (bs, 1, 1, seq_len) -> 0 para PAD
        bs = q_in.size(0)
        q = self.q_lin(q_in).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        k = self.k_lin(k_in).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        v = self.v_lin(v_in).view(bs,-1,self.heads,self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mha_mask_for_scores is not None: scores = scores.masked_fill(mha_mask_for_scores == 0, -1e9)
        weights = self.drop(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v).transpose(1,2).contiguous().view(bs,-1, self.heads*self.d_k)
        return self.out_lin(context)

class ArticleFeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_size, dropout_rate):
        super().__init__(); self.fc1 = nn.Linear(d_model, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, d_model); self.drop = nn.Dropout(dropout_rate)
        self.activ = nn.GELU()
    def forward(self, x): return self.fc2(self.drop(self.activ(self.fc1(x))))

class ArticleEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_size, dropout_rate):
        super().__init__(); self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.attn = ArticleMultiHeadedAttention(num_heads, d_model, dropout_rate)
        self.ff = ArticleFeedForward(d_model, ff_hidden_size, dropout_rate); self.drop = nn.Dropout(dropout_rate)
    def forward(self, embeds, mha_padding_mask): # mha_padding_mask (bs, 1, 1, seq_len)
        attended = self.attn(embeds, embeds, embeds, mha_padding_mask)
        x = self.norm1(embeds + self.drop(attended))
        ff_out = self.ff(x)
        return self.norm2(x + self.drop(ff_out))

class ArticleBERT(nn.Module): # BERT Backbone
    def __init__(self, vocab_sz, d_model, n_layers, heads_config, seq_len_config, pad_idx_config, dropout_rate_config, ff_h_size_config):
        super().__init__(); self.d_model = d_model
        self.emb = ArticleBERTEmbedding(vocab_sz, d_model, seq_len_config, dropout_rate_config, pad_idx_config)
        self.enc_blocks = nn.ModuleList(
            [ArticleEncoderLayer(d_model, heads_config, ff_h_size_config, dropout_rate_config) for _ in range(n_layers)])
        self.pad_idx = pad_idx_config
    def forward(self, input_ids, segment_ids, attention_mask): # Recebe attention_mask do dataloader
        # Usa a attention_mask fornecida (0 para PAD, 1 para token) para criar a máscara do MHA
        mha_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (bs, 1, 1, seq_len)
        x = self.emb(input_ids, segment_ids)
        for block in self.enc_blocks: x = block(x, mha_padding_mask)
        return x

class ArticleNSPHead(nn.Module):
    def __init__(self, hidden_d_model):
        super().__init__(); self.linear = nn.Linear(hidden_d_model, 2); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out[:, 0]))

class ArticleMLMHead(nn.Module):
    def __init__(self, hidden_d_model, vocab_sz):
        super().__init__(); self.linear = nn.Linear(hidden_d_model, vocab_sz); self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, bert_out): return self.log_softmax(self.linear(bert_out))

class ArticleBERTLMWithHeads(nn.Module): # Modelo completo com cabeças
    def __init__(self, bert_model: ArticleBERT, vocab_size: int):
        super().__init__(); self.bert = bert_model
        self.nsp_head = ArticleNSPHead(self.bert.d_model)
        self.mlm_head = ArticleMLMHead(self.bert.d_model, vocab_size)
    def forward(self, input_ids, segment_ids, attention_mask):
        bert_output = self.bert(input_ids, segment_ids, attention_mask)
        return self.nsp_head(bert_output), self.mlm_head(bert_output)

class ScheduledOptim(): # Agendador de LR do artigo
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer; self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0; self.init_lr = np.power(d_model, -0.5)
    def step_and_update_lr(self): self._update_learning_rate(); self._optimizer.step()
    def zero_grad(self): self._optimizer.zero_grad()
    def _get_lr_scale(self):
        if self.n_current_steps == 0: return 0.0
        val1 = np.power(self.n_current_steps, -0.5)
        if self.n_warmup_steps == 0: return val1
        val2 = np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        return np.minimum(val1, val2)
    def _update_learning_rate(self):
        self.n_current_steps += 1; lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups: param_group['lr'] = lr

class PretrainingTrainer: # Trainer para MLM+NSP
    def __init__(self, model: ArticleBERTLMWithHeads, train_dataloader, val_dataloader, 
                 d_model_for_optim: int, # Para ScheduledOptim
                 lr: float, betas: tuple, weight_decay: float, # Para Adam
                 warmup_steps: int, # Para ScheduledOptim
                 device, model_save_path, pad_idx_mlm_loss: int, vocab_size: int, log_freq=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dev = device; self.model = model.to(self.dev)
        self.train_dl, self.val_dl = train_dataloader, val_dataloader
        self.opt = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_schedule = ScheduledOptim(self.opt, d_model_for_optim, warmup_steps)
        self.crit_mlm = nn.NLLLoss(ignore_index=pad_idx_mlm_loss)
        self.crit_nsp = nn.NLLLoss()
        self.log_freq, self.save_path = log_freq, Path(model_save_path)
        self.best_val_loss = float('inf')
        self.vocab_size = vocab_size # Para o reshape do loss MLM

    def _run_epoch(self, epoch_num, is_training): # CORRIGIDO PARA RETORNAR 3 VALORES
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None, 0.0, 0 # loss, nsp_acc, nsp_total_elements
        total_loss_ep, total_mlm_l_ep, total_nsp_l_ep = 0.0, 0.0, 0.0
        tot_nsp_ok, tot_nsp_el = 0, 0
        mode = "Train" if is_training else "Val"; desc = f"Epoch {epoch_num+1} [{mode}] (MLM+NSP)"
        data_iter = tqdm(dl, total=len(dl) if hasattr(dl,'__len__') else None, desc=desc, file=sys.stdout)

        for i_batch, data in enumerate(data_iter):
            data = {k: v.to(self.dev) for k, v in data.items()}
            # ArticleBERT espera attention_mask do dataloader para criar mha_padding_mask
            nsp_out, mlm_out = self.model(data["bert_input"], data["segment_label"], data["attention_mask"])
            
            loss_nsp = self.crit_nsp(nsp_out, data["is_next"])
            loss_mlm = self.crit_mlm(mlm_out.view(-1, self.vocab_size), data["bert_label"].view(-1))
            loss = loss_nsp + loss_mlm

            if is_training: self.opt_schedule.zero_grad(); loss.backward(); self.opt_schedule.step_and_update_lr()
            
            total_loss_ep += loss.item(); total_mlm_l_ep += loss_mlm.item(); total_nsp_l_ep += loss_nsp.item()
            nsp_preds = nsp_out.argmax(dim=-1); tot_nsp_ok += (nsp_preds == data["is_next"]).sum().item(); tot_nsp_el += data["is_next"].nelement()

            if (i_batch + 1) % self.log_freq == 0:
                data_iter.set_postfix({"L":f"{total_loss_ep/(i_batch+1):.3f}", "MLM_L":f"{total_mlm_l_ep/(i_batch+1):.3f}", 
                                       "NSP_L":f"{total_nsp_l_ep/(i_batch+1):.3f}", "NSP_Acc":f"{tot_nsp_ok/tot_nsp_el*100:.2f}%", 
                                       "LR":f"{self.opt_schedule._optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_total_l = total_loss_ep/len(dl) if len(dl)>0 else 0; avg_mlm_l = total_mlm_l_ep/len(dl) if len(dl)>0 else 0
        avg_nsp_l = total_nsp_l_ep/len(dl) if len(dl)>0 else 0; final_nsp_acc = tot_nsp_ok*100.0/tot_nsp_el if tot_nsp_el>0 else 0
        self.logger.info(f"{desc} - AvgTotalL: {avg_total_l:.4f}, AvgMLML: {avg_mlm_l:.4f}, AvgNSPL: {avg_nsp_l:.4f}, NSP Acc: {final_nsp_acc:.2f}%")
        return avg_total_l, final_nsp_acc, tot_nsp_el # Retorna 3 valores

    def train(self, num_epochs): # Nome do método como no artigo
        self.logger.info(f"Iniciando pré-treinamento (MLM+NSP) por {num_epochs} épocas.")
        model_saved_this_run = False
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_total_loss_epoch = None
            if self.val_dl:
                with torch.no_grad():
                    val_total_loss_epoch, _, _ = self._run_epoch(epoch, is_training=False) # Desempacota 3 valores
            
            if self.val_dl and val_total_loss_epoch is not None and val_total_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_total_loss_epoch
                self.logger.info(f"Nova melhor Val Total Loss: {self.best_val_loss:.4f}. Salvando modelo: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
            elif not self.val_dl and epoch == num_epochs - 1:
                self.logger.info(f"MLM+NSP sem validação. Salvando modelo da última época ({epoch+1}): {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
            self.logger.info("-" * 30)
        
        best_val_display = "N/A"
        if isinstance(self.best_val_loss,(int,float)) and not math.isinf(self.best_val_loss):
            if math.isnan(self.best_val_loss): best_val_display="N/A (NaN)"
            else:best_val_display=f"{self.best_val_loss:.4f}"
        self.logger.info(f"Pré-treinamento (MLM+NSP) concluído. Melhor Val Total Loss: {best_val_display}")
        if model_saved_this_run: self.logger.info(f"Modelo salvo em: {self.save_path}")


# --- Funções para o Pipeline ---
def setup_data_and_train_tokenizer(args, logger):
    logger.info("--- Fase: Preparação de Dados Aroeira e Tokenizador (WordPiece) ---")
    _all_aroeira_sents_list = [] # Lista de sentenças para NSP e tokenizador
    text_col = "text"; temp_file_for_tokenizer = Path(args.temp_tokenizer_train_file)

    # Carregamento de dados Aroeira (S3 ou Hub) -> Resulta em _all_aroeira_sents_list
    if args.sagemaker_input_data_dir and args.input_data_filename:
        local_data_path = os.path.join(args.sagemaker_input_data_dir, args.input_data_filename)
        logger.info(f"Lendo dados Aroeira do SageMaker: {local_data_path}")
        if Path(local_data_path).exists():
            s3_ds = datasets.load_dataset("json", data_files=local_data_path, split="train", trust_remote_code=args.trust_remote_code)
            for ex in tqdm(s3_ds, desc="Extraindo sentenças (JSON S3)", file=sys.stdout):
                sent = ex.get(text_col);_all_aroeira_sents_list.append(sent.strip()) if isinstance(sent,str) and sent.strip() else None
        else: raise FileNotFoundError(f"Arquivo {local_data_path} (SageMaker) não encontrado.")
    elif args.aroeira_subset_size is not None:
        logger.info(f"Coletando {args.aroeira_subset_size} exemplos do Aroeira (Hub)...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        iterator = iter(streamed_ds); _collected_ex = []
        try:
            for _ in range(args.aroeira_subset_size): _collected_ex.append(next(iterator))
        except StopIteration: logger.warning(f"Stream Aroeira (Hub) esgotado. Coletados {len(_collected_ex)}.")
        for ex in tqdm(_collected_ex, desc="Extraindo sentenças (subset Hub)", file=sys.stdout):
            sent = ex.get(text_col); _all_aroeira_sents_list.append(sent.strip()) if isinstance(sent, str) and sent.strip() else None
    else: 
        logger.info("Processando stream completo do Aroeira (Hub) para lista de sentenças...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        for ex in tqdm(streamed_ds, desc="Coletando Aroeira (stream completo)", file=sys.stdout):
            sent = ex.get(text_col); _all_aroeira_sents_list.append(sent.strip()) if isinstance(sent, str) and sent.strip() else None
    if not _all_aroeira_sents_list: raise ValueError("Nenhuma sentença Aroeira extraída.")
    logger.info(f"Total de sentenças Aroeira para pré-treino e tokenizador: {len(_all_aroeira_sents_list)}")
    
    # Salva sentenças em arquivo temporário para treinar tokenizador
    if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
    with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
        for s_line in _all_aroeira_sents_list: f.write(s_line + "\n")
    tokenizer_train_file_str = str(temp_file_for_tokenizer)
        
    # Treinamento e Carregamento do Tokenizador WordPiece
    TOKENIZER_ASSETS_DIR = Path(args.output_dir) / "bert_wordpiece_tokenizer_assets"
    TOKENIZER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    vocab_file_target = TOKENIZER_ASSETS_DIR / "vocab.txt" 

    if not vocab_file_target.exists():
        if not Path(tokenizer_train_file_str).exists():
            raise FileNotFoundError(f"Arquivo treino '{tokenizer_train_file_str}' não encontrado.")
        wp_trainer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True)
        logger.info(f"Treinando BertWordPieceTokenizer com [{tokenizer_train_file_str}]...")
        wp_trainer.train(files=[tokenizer_train_file_str], vocab_size=args.vocab_size, 
                           min_frequency=args.min_frequency_tokenizer, 
                           special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
                           limit_alphabet=1000, wordpieces_prefix="##")
        wp_trainer.save_model(str(TOKENIZER_ASSETS_DIR), prefix=None) # Salva como vocab.txt no diretório
        if not vocab_file_target.exists(): raise FileNotFoundError(f"Falha ao salvar {vocab_file_target}")
        logger.info(f"Tokenizador BertWordPiece salvo em: {vocab_file_target}")
    else: logger.info(f"Tokenizador já existe em '{TOKENIZER_ASSETS_DIR}'. Carregando...")
    
    loaded_tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_ASSETS_DIR), local_files_only=True)
    pad_id_val = loaded_tokenizer.pad_token_id
    logger.info(f"Vocabulário (BertTokenizerFast): {loaded_tokenizer.vocab_size}, PAD ID: {pad_id_val}")
    return loaded_tokenizer, pad_id_val, _all_aroeira_sents_list


def run_bert_pretraining_nsp_mlm(args, tokenizer_instance: BertTokenizer, pad_token_id_val, all_sentences_list, logger):
    logger.info("--- Fase: Pré-Treinamento MLM + NSP (Estilo Artigo) ---")
    if not all_sentences_list or len(all_sentences_list) < 2:
        logger.error("Dados insuficientes para pré-treinamento MLM+NSP."); return

    val_s_pt_r = 0.1; num_v_pt = int(len(all_sentences_list) * val_s_pt_r)
    if num_v_pt < 1 and len(all_sentences_list) > 1: num_v_pt = 1
    tr_s_pt = all_sentences_list[num_v_pt:]; v_s_pt = all_sentences_list[:num_v_pt]
    if not tr_s_pt: tr_s_pt = all_sentences_list; v_s_pt = []
    logger.info(f"Sentenças Pré-Treino (MLM+NSP): Treino={len(tr_s_pt)}, Validação={len(v_s_pt)}")
    
    train_ds_pt = ArticleStyleBERTDataset(tr_s_pt, tokenizer_instance, args.max_len)
    if len(train_ds_pt)==0 and len(tr_s_pt)>0: logger.error("Dataset de treino MLM+NSP vazio."); return
    train_dl_pt = DataLoader(train_ds_pt, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    val_dl_pt = None
    if v_s_pt and len(v_s_pt) > 0:
        val_ds_pt = ArticleStyleBERTDataset(v_s_pt, tokenizer_instance, args.max_len)
        if len(val_ds_pt) > 0: val_dl_pt = DataLoader(val_ds_pt, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)

    bert_article_backbone = ArticleBERT(
        vocab_sz=tokenizer_instance.vocab_size, d_model=args.model_d_model, n_layers=args.model_n_layers, 
        heads_config=args.model_heads, seq_len_config=args.max_len, pad_idx_config=pad_token_id_val, 
        dropout_rate_config=args.model_dropout_prob, ff_h_size_config=args.model_intermediate_size)
    bertlm_nsp_model = ArticleBERTLMWithHeads(bert_article_backbone, tokenizer_instance.vocab_size)
    
    trainer_pt_instance = PretrainingTrainer(
        model=bertlm_nsp_model, train_dataloader=train_dl_pt, val_dataloader=val_dl_pt,
        d_model_for_optim=args.model_d_model, lr=args.lr_pretrain_adam, 
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, # Usando AdamW params
        warmup_steps=args.warmup_steps, device=args.device, 
        model_save_path=args.pretrained_bert_save_filename, # Renomeado
        pad_idx_mlm_loss=pad_token_id_val, vocab_size=tokenizer_instance.vocab_size, log_freq=args.logging_steps)
    logger.info("Pré-treinamento MLM+NSP (estilo artigo) configurado. Iniciando...")
    trainer_pt_instance.train(num_epochs=args.epochs_pretrain)

# --- Função Principal e Parseador de Argumentos ---
def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino BERT (Estilo Artigo) no Aroeira.")
    # Gerais
    parser.add_argument("--max_len", type=int, default=64) # Padrão do Artigo
    parser.add_argument("--aroeira_subset_size", type=int, default=10000) # Para Aroeira
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2) # Ajustado
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--sagemaker_input_data_dir", type=str, default=None)
    parser.add_argument("--input_data_filename", type=str, default=None)

    # Pré-treinamento MLM+NSP
    parser.add_argument("--epochs_pretrain", type=int, default=1) # Reduzido para teste
    parser.add_argument("--batch_size_pretrain", type=int, default=16) # Reduzido
    parser.add_argument("--lr_pretrain_adam", type=float, default=1e-4) # LR base para Adam + ScheduledOptim
    parser.add_argument("--warmup_steps", type=int, default=5000) # Reduzido

    # Arquitetura Modelo BERT (Estilo Artigo)
    parser.add_argument("--model_d_model", type=int, default=256) # hidden_size do artigo
    parser.add_argument("--model_n_layers", type=int, default=3) # n_layers
    parser.add_argument("--model_heads", type=int, default=4)    # heads
    parser.add_argument("--model_dropout_prob", type=float, default=0.1) # dropout_rate
    # model_intermediate_size (ff_h_size) será d_model * 4
    
    # Paths
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './bert_articlestyle_outputs'))
    parser.add_argument("--pretrained_bert_save_filename", type=str, default="aroeira_bert_articlestyle_pretrained.pth") # Novo nome
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_wp_tokenizer.txt") # Para WordPiece
    
    # Controle de Fluxo
    parser.add_argument("--do_dataprep_tokenizer", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--do_pretrain", type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_filename', type=str, default='pretrain_articlestyle.log')
    parser.add_argument('--logging_steps', type=int, default=10) # Frequência de log no TQDM e Trainer

    # Adam params para ScheduledOptim (pode ser útil se o SageMaker os passar)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.00) # Adam original não tem weight_decay, AdamW sim. Artigo usa Adam.

    known_args, _ = parser.parse_known_args(custom_args_list if custom_args_list else None)
    if known_args.device is None: known_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    known_args.model_intermediate_size = known_args.model_d_model * 4 # Derivado
    output_dir_path = Path(known_args.output_dir)
    if not str(known_args.output_dir).startswith("/opt/ml/"): output_dir_path.mkdir(parents=True, exist_ok=True)
    
    known_args.pretrained_bert_save_filename = str(output_dir_path / Path(known_args.pretrained_bert_save_filename).name)
    known_args.temp_tokenizer_train_file = str(output_dir_path / Path(known_args.temp_tokenizer_train_file).name)
    known_args.log_file_path = str(output_dir_path / Path(known_args.log_filename).name)
    return known_args

def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)
    setup_logging(ARGS.log_level, ARGS.log_file_path)
    logger = logging.getLogger(__name__)

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Datasets: {datasets.__version__}")
    logger.info(f"Tokenizers: {tokenizers.__version__}")
    logger.info("--- Configurações Utilizadas (Estilo Artigo) ---")
    for arg_name, value in vars(ARGS).items(): logger.info(f"{arg_name}: {value}")
    logger.info("-------------------------------------------------")

    current_tokenizer_obj, current_pad_id, data_source_for_pretrain = None, None, None
    
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain:
        current_tokenizer_obj, current_pad_id, data_source_for_pretrain = setup_data_and_train_tokenizer(ARGS, logger)
    else:
        logger.info("Nenhuma ação de preparação de dados ou pré-treinamento solicitada. Encerrando.")
        return

    if ARGS.do_pretrain:
        if not data_source_for_pretrain or not current_tokenizer_obj:
            logger.error("Fonte de dados Aroeira ou tokenizador não preparados para pré-treinamento.")
        else:
            # Usa a função de pré-treinamento MLM+NSP
            run_bert_pretraining_nsp_mlm(ARGS, current_tokenizer_obj, current_pad_id, data_source_for_pretrain, logger)
    
    logger.info("--- Pipeline de Pré-treinamento (Estilo Artigo) Finalizado ---")

if __name__ == "__main__":
    main()
