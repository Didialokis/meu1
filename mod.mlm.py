# train_pipeline_nsp.py (Novo nome sugerido)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets
import tokenizers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast as BertTokenizer
from tqdm.auto import tqdm
import random
from pathlib import Path
import sys
import math
import argparse
import os
import logging
import datetime

# --- Definições de Classes ---

class BERTPretrainingDataset(Dataset):
    """Dataset para pré-treinamento BERT com MLM e NSP."""
    def __init__(self, all_sentences_list, tokenizer_instance: BertTokenizer, max_len_config: int, pad_token_id_val: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = tokenizer_instance
        self.max_len = max_len_config
        self.corpus_sents = [s for s in all_sentences_list if s and s.strip()] # Garante que não há strings vazias
        self.corpus_len = len(self.corpus_sents)

        if self.corpus_len < 2:
            raise ValueError("Corpus precisa de pelo menos 2 sentenças para NSP.")

        self.vocab_size = self.tokenizer.vocab_size
        self.cls_token = self.tokenizer.cls_token
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_token = self.tokenizer.sep_token
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = pad_token_id_val
        self.mask_id = self.tokenizer.mask_token_id
        
        self.logger.info(f"BERTPretrainingDataset inicializado com {self.corpus_len} sentenças.")

    def __len__(self):
        return self.corpus_len # Cada sentença pode ser o início de um par

    def _get_sentence_pair_and_nsp_label(self, sent_a_idx):
        sent_a = self.corpus_sents[sent_a_idx]
        is_next = 0 # Default: Not Next
        
        if random.random() < 0.5 and sent_a_idx + 1 < self.corpus_len: # 50% chance de par positivo
            sent_b = self.corpus_sents[sent_a_idx + 1]
            is_next = 1
        else: # Par negativo
            rand_sent_b_idx = random.randrange(self.corpus_len)
            # Garante que a sentença aleatória não é a mesma que sent_a ou a próxima de sent_a
            while self.corpus_len > 1 and (rand_sent_b_idx == sent_a_idx or rand_sent_b_idx == (sent_a_idx + 1) % self.corpus_len):
                rand_sent_b_idx = random.randrange(self.corpus_len)
            sent_b = self.corpus_sents[rand_sent_b_idx]
        return sent_a, sent_b, is_next

    def _apply_mlm(self, token_ids_list: list):
        inputs, labels = list(token_ids_list), list(token_ids_list)
        for i, token_id in enumerate(inputs):
            # Não mascara CLS, SEP, PAD que serão adicionados depois ou já estão
            if token_id in [self.cls_id, self.sep_id, self.pad_id, self.mask_id]: 
                labels[i] = self.pad_id; continue
            if random.random() < 0.15: # 15% dos tokens da sentença original
                action_prob = random.random()
                if action_prob < 0.8: inputs[i] = self.mask_id
                elif action_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
                # 10% -> Manter Original
            else: labels[i] = self.pad_id # Não prever tokens não mascarados
        return inputs, labels

    def __getitem__(self, idx):
        sent_a_str, sent_b_str, nsp_label = self._get_sentence_pair_and_nsp_label(idx)

        # Tokeniza sentenças separadamente para aplicar MLM antes de adicionar CLS/SEP global
        # `add_special_tokens=False` para evitar CLS/SEP internos aqui
        tokens_a = self.tokenizer.encode(sent_a_str, add_special_tokens=False, truncation=False) # Não truncar ainda
        tokens_b = self.tokenizer.encode(sent_b_str, add_special_tokens=False, truncation=False)

        # Aplica MLM
        masked_tokens_a_ids, mlm_labels_a_ids = self._apply_mlm(tokens_a)
        masked_tokens_b_ids, mlm_labels_b_ids = self._apply_mlm(tokens_b)

        # Constrói a sequência final: [CLS] A [SEP] B [SEP]
        input_ids = [self.cls_id] + masked_tokens_a_ids + [self.sep_id] + masked_tokens_b_ids + [self.sep_id]
        
        # Labels para MLM: PAD para tokens não-MLM e para CLS/SEP
        mlm_labels = [self.pad_id] + mlm_labels_a_ids + [self.pad_id] + mlm_labels_b_ids + [self.pad_id]
        
        # Segment IDs: 0 para sentença A (incluindo [CLS] e primeiro [SEP]), 1 para sentença B (incluindo segundo [SEP])
        segment_ids = [0] * (len(masked_tokens_a_ids) + 2) + [1] * (len(masked_tokens_b_ids) + 1)

        # Truncamento para MAX_LEN (se necessário)
        input_ids = input_ids[:self.max_len]
        mlm_labels = mlm_labels[:self.max_len]
        segment_ids = segment_ids[:self.max_len]
        
        # Padding
        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids.extend([self.pad_id] * padding_len)
        mlm_labels.extend([self.pad_id] * padding_len)
        segment_ids.extend([0] * padding_len) # Padded tokens têm segmento 0

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long), # Renomeado para clareza
            "nsp_label": torch.tensor(nsp_label, dtype=torch.long)
        }

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, dropout_prob, pad_token_id):
        super().__init__(); self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.seg_emb = nn.Embedding(2, hidden_size); self.pos_emb = nn.Embedding(max_len, hidden_size) # Segmentos 0 e 1
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12); self.drop = nn.Dropout(dropout_prob)
        self.register_buffer("pos_ids", torch.arange(max_len).expand((1, -1)))
    def forward(self, input_ids, segment_ids): # segment_ids agora são 0 ou 1
        seq_len = input_ids.size(1); tok_e, seg_e = self.tok_emb(input_ids), self.seg_emb(segment_ids)
        pos_e = self.pos_emb(self.pos_ids[:, :seq_len])
        return self.drop(self.norm(tok_e + pos_e + seg_e))

class BERTBaseModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_len_config, pad_token_id, dropout_prob):
        super().__init__()
        self.emb = BERTEmbedding(vocab_size, hidden_size, max_len_config, dropout_prob, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size, 
                                             dropout=dropout_prob, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, input_ids, attention_mask, segment_ids):
        x = self.emb(input_ids, segment_ids)
        return self.enc(x, src_key_padding_mask=(attention_mask == 0))

class BERTForPretraining(nn.Module): # Novo modelo para MLM + NSP
    def __init__(self, bert_base: BERTBaseModel, vocab_size: int, hidden_size: int):
        super().__init__()
        self.bert_base = bert_base
        # Cabeça de MLM
        self.mlm_head_transform = nn.Linear(hidden_size, hidden_size) # Transformação adicional como no BERT original
        self.mlm_head_activation = nn.GELU()
        self.mlm_head_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlm_output_bias = nn.Parameter(torch.zeros(vocab_size)) # Bias para a camada de output MLM
        # A camada final de MLM usará os pesos da camada de embedding de token (weight tying)
        
        # Cabeça de NSP
        self.nsp_pooler = nn.Linear(hidden_size, hidden_size) # Pooler para o token [CLS]
        self.nsp_pooler_activation = nn.Tanh()
        self.nsp_classifier = nn.Linear(hidden_size, 2) # 2 classes: IsNext, NotNext

    def forward(self, input_ids, attention_mask, segment_ids):
        sequence_output = self.bert_base(input_ids, attention_mask, segment_ids) # (batch, seq_len, hidden_size)
        
        # Para MLM
        mlm_transformed_output = self.mlm_head_activation(self.mlm_head_transform(sequence_output))
        mlm_normalized_output = self.mlm_head_norm(mlm_transformed_output)
        # Weight tying: multiplica pela transposta dos embeddings de token
        # Os pesos da camada de embedding de token estão em self.bert_base.emb.tok_emb.weight
        mlm_logits = F.linear(mlm_normalized_output, self.bert_base.emb.tok_emb.weight, bias=self.mlm_output_bias)
        
        # Para NSP
        pooled_output = self.nsp_pooler_activation(self.nsp_pooler(sequence_output[:, 0])) # Usa o output do [CLS]
        nsp_logits = self.nsp_classifier(pooled_output)
        
        return mlm_logits, nsp_logits

class PretrainingTrainer: # Adaptado de SimplifiedTrainer para MLM+NSP
    def __init__(self, model: BERTForPretraining, train_dataloader, val_dataloader, optimizer, 
                 mlm_criterion, nsp_criterion, device, model_save_path, vocab_size, log_freq=20):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model, self.train_dl, self.val_dl = model.to(device), train_dataloader, val_dataloader
        self.opt, self.crit_mlm, self.crit_nsp, self.dev = optimizer, mlm_criterion, nsp_criterion, device
        self.save_path = Path(model_save_path); self.best_val_loss = float('inf')
        self.vocab_size = vocab_size; self.log_freq = log_freq

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None, 0.0, 0 # loss, nsp_acc, nsp_total_elements
        total_loss_epoch, total_mlm_loss_epoch, total_nsp_loss_epoch = 0.0, 0.0, 0.0
        total_nsp_correct, total_nsp_elements = 0, 0
        
        desc = f"Epoch {epoch_num+1} [{'Train' if is_training else 'Val'}] (MLM+NSP)"
        progress_bar = tqdm(dl, total=len(dl) if hasattr(dl, '__len__') else None, desc=desc, file=sys.stdout)

        for i_batch, batch in enumerate(progress_bar):
            input_ids=batch["input_ids"].to(self.dev); attention_mask=batch["attention_mask"].to(self.dev)
            segment_ids=batch["segment_ids"].to(self.dev) 
            mlm_labels=batch["mlm_labels"].to(self.dev)
            nsp_label=batch["nsp_label"].to(self.dev)
            
            if is_training: self.opt.zero_grad()
            with torch.set_grad_enabled(is_training):
                mlm_logits, nsp_logits = self.model(input_ids, attention_mask, segment_ids)
                
                loss_mlm = self.crit_mlm(mlm_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
                loss_nsp = self.crit_nsp(nsp_logits, nsp_label)
                loss = loss_mlm + loss_nsp # Perda combinada
            
            if is_training: loss.backward(); self.opt.step()
            
            total_loss_epoch += loss.item()
            total_mlm_loss_epoch += loss_mlm.item()
            total_nsp_loss_epoch += loss_nsp.item()

            nsp_preds = torch.argmax(nsp_logits, dim=1)
            total_nsp_correct += (nsp_preds == nsp_label).sum().item()
            total_nsp_elements += nsp_label.size(0)

            if (i_batch + 1) % self.log_freq == 0: 
                progress_bar.set_postfix({
                    "loss": f"{total_loss_epoch / (i_batch + 1):.4f}",
                    "mlm_l": f"{total_mlm_loss_epoch / (i_batch + 1):.4f}",
                    "nsp_l": f"{total_nsp_loss_epoch / (i_batch + 1):.4f}",
                    "nsp_acc": f"{total_nsp_correct / total_nsp_elements * 100:.2f}%" if total_nsp_elements > 0 else "0.00%"
                })
        
        avg_total_loss = total_loss_epoch / len(dl) if len(dl) > 0 else 0.0
        avg_mlm_loss = total_mlm_loss_epoch / len(dl) if len(dl) > 0 else 0.0
        avg_nsp_loss = total_nsp_loss_epoch / len(dl) if len(dl) > 0 else 0.0
        nsp_accuracy = total_nsp_correct / total_nsp_elements * 100 if total_nsp_elements > 0 else 0.0
        
        self.logger.info(f"{desc} - Avg Total Loss: {avg_total_loss:.4f}, MLM Loss: {avg_mlm_loss:.4f}, NSP Loss: {avg_nsp_loss:.4f}, NSP Acc: {nsp_accuracy:.2f}%")
        return avg_total_loss # Retorna o loss total para salvar o melhor modelo

    def train(self, num_epochs):
        self.logger.info(f"Treinando (MLM+NSP) por {num_epochs} épocas.")
        model_saved_this_run = False
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_total_loss_epoch = None
            if self.val_dl:
                 val_total_loss_epoch, _, _ = self._run_epoch(epoch, is_training=False) # Ignora nsp_acc e total_elements aqui

            if self.val_dl and val_total_loss_epoch is not None and val_total_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_total_loss_epoch
                self.logger.info(f"Melhor Val Total Loss: {self.best_val_loss:.4f}. Salvando: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
            print("-" * 30)
        
        if not self.val_dl and num_epochs > 0: # Salva última época se sem validação
            self.logger.info(f"(MLM+NSP) Sem validação. Salvando modelo da última época ({num_epochs}): {self.save_path}")
            torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
        
        best_val_display = "N/A"
        if isinstance(self.best_val_loss,(int,float)) and not math.isinf(self.best_val_loss):
            if math.isnan(self.best_val_loss): best_val_display="N/A (NaN)"
            else:best_val_display=f"{self.best_val_loss:.4f}"
        self.logger.info(f"Treinamento (MLM+NSP) concluído após {num_epochs} épocas. Melhor Val Total Loss: {best_val_display}")
        if model_saved_this_run: self.logger.info(f"Modelo salvo em: {self.save_path}")
        elif num_epochs == 0: self.logger.info("Nenhum treinamento realizado (0 épocas).")

# --- Funções para o Pipeline ---
def setup_data_and_train_tokenizer(args, logger): # Adicionado logger
    logger.info("--- Fase: Preparação de Dados Aroeira e Tokenizador (MLM+NSP) ---")
    # ... (Lógica de carregamento do Aroeira e escrita para temp_file_for_tokenizer mantida como antes) ...
    # ... (Lógica de treino do BertWordPieceTokenizer e carregamento com BertTokenizerFast mantida como antes) ...
    # A função deve retornar: loaded_tokenizer, pad_id_val, e a lista all_aroeira_individual_sentences
    # pois BERTPretrainingDataset precisa da lista de sentenças para NSP.
    
    # (Conteúdo da função setup_data_and_train_tokenizer da última resposta, usando logger)
    aroeira_data_source_for_pretrain = None # Será a lista de sentenças
    _all_aroeira_sentences_list_local = [] 
    text_col = "text"; temp_file_for_tokenizer = Path(args.temp_tokenizer_train_file)

    if args.sagemaker_input_data_dir and args.input_data_filename:
        local_data_path = os.path.join(args.sagemaker_input_data_dir, args.input_data_filename)
        logger.info(f"Lendo dados Aroeira do SageMaker: {local_data_path}")
        if Path(local_data_path).exists():
            # Para NSP, precisamos da lista de sentenças, então lemos o arquivo JSON aqui.
            loaded_s3_dataset = datasets.load_dataset("json", data_files=local_data_path, split="train", trust_remote_code=args.trust_remote_code)
            for ex in tqdm(loaded_s3_dataset, desc="Extraindo sentenças do JSON (S3)", file=sys.stdout):
                sent = ex.get(text_col)
                if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences_list_local.append(sent.strip())
            logger.info(f"Carregadas {len(_all_aroeira_sentences_list_local)} sentenças do arquivo S3 JSON.")
            if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
            with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
                for s_line in _all_aroeira_sentences_list_local: f.write(s_line + "\n")
            aroeira_data_source_for_pretrain = _all_aroeira_sentences_list_local
        else: raise FileNotFoundError(f"Arquivo {local_data_path} (SageMaker) não encontrado.")
    elif args.aroeira_subset_size is not None:
        logger.info(f"Coletando {args.aroeira_subset_size} exemplos do Aroeira (Hub)...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        iterator = iter(streamed_ds); _collected_examples = []
        try:
            for _ in range(args.aroeira_subset_size): _collected_examples.append(next(iterator))
        except StopIteration: logger.warning(f"Stream Aroeira (Hub) esgotado. Coletados {len(_collected_examples)}.")
        for ex in tqdm(_collected_examples, desc="Extraindo sentenças (subset Hub)", file=sys.stdout):
            sent = ex.get(text_col)
            if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences_list_local.append(sent.strip())
        if not _all_aroeira_sentences_list_local: raise ValueError("Nenhuma sentença Aroeira (subset Hub) extraída.")
        aroeira_data_source_for_pretrain = _all_aroeira_sentences_list_local
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for s_line in _all_aroeira_sentences_list_local: f.write(s_line + "\n")
    else: 
        logger.info("Processando stream completo do Aroeira (Hub) para arquivo e lista de sentenças...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for ex in tqdm(streamed_ds, desc="Escrevendo e coletando Aroeira (stream completo)", file=sys.stdout):
                sent = ex.get(text_col)
                if isinstance(sent, str) and sent.strip(): 
                    f.write(sent + "\n")
                    _all_aroeira_sentences_list_local.append(sent.strip())
        if not _all_aroeira_sentences_list_local: raise ValueError("Nenhuma sentença Aroeira (stream Hub) processada.")
        aroeira_data_source_for_pretrain = _all_aroeira_sentences_list_local # Para NSP
    
    logger.info(f"Total de sentenças Aroeira para pré-treino e tokenizador: {len(aroeira_data_source_for_pretrain)}")
    
    tokenizer_train_file_str = str(temp_file_for_tokenizer)
    TOKENIZER_SAVE_DIRECTORY = Path(args.output_dir) / "wordpiece_tokenizer_assets_nsp"
    TOKENIZER_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    vocab_file_in_save_dir = TOKENIZER_SAVE_DIRECTORY / "vocab.txt" 

    if not vocab_file_in_save_dir.exists():
        if not Path(tokenizer_train_file_str).exists():
            raise FileNotFoundError(f"Arquivo de treino para tokenizador '{tokenizer_train_file_str}' não encontrado.")
        wp_tokenizer_trainer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True)
        logger.info(f"Treinando tokenizador BertWordPiece com [{tokenizer_train_file_str}]...")
        wp_tokenizer_trainer.train(
            files=[tokenizer_train_file_str], vocab_size=args.vocab_size, 
            min_frequency=args.min_frequency_tokenizer, 
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
            limit_alphabet=1000, wordpieces_prefix="##")
        wp_tokenizer_trainer.save_model(str(TOKENIZER_SAVE_DIRECTORY), prefix=None) 
        logger.info(f"Tokenizador BertWordPiece treinado e salvo em: {vocab_file_in_save_dir}")
    else: logger.info(f"Tokenizador (vocab.txt) já existe em '{TOKENIZER_SAVE_DIRECTORY}'. Carregando...")
    
    logger.info(f"Tentando carregar tokenizador BertTokenizerFast do DIRETÓRIO: {TOKENIZER_SAVE_DIRECTORY}")
    loaded_tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_SAVE_DIRECTORY), 
        do_lower_case=True, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", 
        cls_token="[CLS]", mask_token="[MASK]", local_files_only=True)
    pad_id_val = loaded_tokenizer.pad_token_id
    logger.info(f"Vocabulário (BertTokenizerFast): {loaded_tokenizer.vocab_size}, PAD ID: {pad_id_val}")
    return loaded_tokenizer, pad_id_val, aroeira_data_source_for_pretrain


def run_bert_pretraining(args, tokenizer_obj: BertTokenizer, pad_token_id_val, all_sentences_for_nsp, logger): # Renomeado
    logger.info("--- Fase: Pré-Treinamento MLM + NSP ---")
    
    # BERTPretrainingDataset precisa da lista completa de sentenças para amostrar pares NSP
    if not all_sentences_for_nsp or len(all_sentences_for_nsp) < 2:
        logger.error("Dados insuficientes para pré-treinamento NSP (precisa de pelo menos 2 sentenças).")
        return

    val_s_pt_r = 0.1 # Proporção para validação do pré-treinamento
    num_v_pt = int(len(all_sentences_for_nsp) * val_s_pt_r)
    if num_v_pt < 1 and len(all_sentences_for_nsp) > 1: num_v_pt = 1
    
    tr_s_pt = all_sentences_for_nsp[num_v_pt:]
    v_s_pt = all_sentences_for_nsp[:num_v_pt]
    if not tr_s_pt: tr_s_pt = all_sentences_for_nsp; v_s_pt = []

    logger.info(f"Sentenças de Treino (MLM+NSP): {len(tr_s_pt)}, Validação: {len(v_s_pt)}")
    
    train_ds_pt = BERTPretrainingDataset(tr_s_pt, tokenizer_obj, args.max_len, pad_token_id_val)
    if len(train_ds_pt) == 0 and len(tr_s_pt) > 0 : 
        logger.error("Dataset de treino MLM+NSP vazio após processamento em BERTPretrainingDataset.")
        return # Ou raise
    train_dl_pt = DataLoader(train_ds_pt, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    
    val_dl_pt = None
    if v_s_pt and len(v_s_pt) > 0:
        val_ds_pt = BERTPretrainingDataset(v_s_pt, tokenizer_obj, args.max_len, pad_token_id_val)
        if len(val_ds_pt) > 0: val_dl_pt = DataLoader(val_ds_pt, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)
        elif len(v_s_pt) > 0 : logger.warning("Dataset de validação MLM+NSP resultou em 0 exemplos.")

    bert_base = BERTBaseModel(tokenizer_obj.vocab_size, args.model_hidden_size, args.model_num_layers, 
                             args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
                             pad_token_id_val, dropout_prob=args.model_dropout_prob)
    bert_pretrain_model = BERTForPretraining(bert_base, tokenizer_obj.vocab_size, args.model_hidden_size)
    
    opt_pt = torch.optim.AdamW(bert_pretrain_model.parameters(), lr=args.lr_pretrain, 
                                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.weight_decay)
    # Critérios de Perda
    crit_mlm = nn.CrossEntropyLoss(ignore_index=pad_token_id_val)
    crit_nsp = nn.CrossEntropyLoss() # NSP não tem ignore_index usualmente (são labels 0 ou 1)
    
    # O Trainer precisa ser adaptado para lidar com dois losses e duas saídas do modelo
    # Reutilizando PretrainingTrainer que foi definido para isso
    trainer_pt_instance = PretrainingTrainer( 
        bert_pretrain_model, train_dl_pt, val_dl_pt, opt_pt, 
        crit_mlm, crit_nsp, # Passa ambos os critérios
        args.device, 
        args.pretrained_bertlm_save_filename, # Salva o modelo MLM+NSP
        tokenizer_obj.vocab_size, # Para reshape do MLM loss
        log_freq=args.logging_steps
    )
    logger.info("Pré-treinamento MLM+NSP configurado. Iniciando...")
    trainer_pt_instance.train(num_epochs=args.epochs_pretrain)

# --- Função Principal e Parseador de Argumentos ---
def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino BERT no Aroeira (MLM+NSP).")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aroeira_subset_size", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--sagemaker_input_data_dir", type=str, default=None)
    parser.add_argument("--input_data_filename", type=str, default=None)
    parser.add_argument("--epochs_pretrain", type=int, default=1)
    parser.add_argument("--batch_size_pretrain", type=int, default=8)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)
    parser.add_argument("--model_hidden_size", type=int, default=256)
    parser.add_argument("--model_num_layers", type=int, default=2)
    parser.add_argument("--model_num_attention_heads", type=int, default=4)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './bert_mlm_nsp_outputs'))
    # Tokenizer_vocab_filename não é mais um arg, o caminho é construído internamente
    parser.add_argument("--pretrained_bertlm_save_filename", type=str, default="aroeira_bert_mlm_nsp_pretrained.pth")
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_tokenizer_nsp.txt")
    parser.add_argument("--do_dataprep_tokenizer", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--do_pretrain", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_filename', type=str, default='pretraining_pipeline.log')
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    known_args, _ = parser.parse_known_args(custom_args_list if custom_args_list else None)

    if known_args.device is None: known_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    known_args.model_intermediate_size = known_args.model_hidden_size * 4
    output_dir_path = Path(known_args.output_dir)
    if not str(known_args.output_dir).startswith("/opt/ml/"): output_dir_path.mkdir(parents=True, exist_ok=True)
    
    known_args.pretrained_bertlm_save_filename = str(output_dir_path / Path(known_args.pretrained_bertlm_save_filename).name)
    known_args.temp_tokenizer_train_file = str(output_dir_path / Path(known_args.temp_tokenizer_train_file).name)
    known_args.log_file_path = str(output_dir_path / Path(known_args.log_filename).name)
    return known_args

def setup_logging(log_level_str, log_file_path_str):
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int): raise ValueError(f'Invalid log level: {log_level_str}')
    Path(log_file_path_str).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path_str), logging.StreamHandler(sys.stdout)]
    )

def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)
    setup_logging(ARGS.log_level, ARGS.log_file_path)
    logger = logging.getLogger(__name__)

    logger.info(f"PyTorch: {torch.__version__}") # Movido para após setup do logging
    logger.info(f"Datasets: {datasets.__version__}")
    logger.info(f"Tokenizers: {tokenizers.__version__}")
    logger.info("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items(): logger.info(f"{arg_name}: {value}")
    logger.info("----------------------------------")

    current_tokenizer_obj, current_pad_id, data_source_for_pretrain = None, None, None # Renomeado data_source
    
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain:
        current_tokenizer_obj, current_pad_id, data_source_for_pretrain = setup_data_and_train_tokenizer(ARGS, logger)
    else:
        logger.info("Nenhuma ação de preparação de dados ou pré-treinamento solicitada. Encerrando.")
        return

    if ARGS.do_pretrain:
        if not data_source_for_pretrain or not current_tokenizer_obj:
            logger.error("Fonte de dados Aroeira ou tokenizador não preparados para pré-treinamento.")
        else:
            run_bert_pretraining(ARGS, current_tokenizer_obj, current_pad_id, data_source_for_pretrain, logger) # Renomeado
    
    logger.info("--- Pipeline de Pré-treinamento (MLM+NSP) Finalizado ---")

if __name__ == "__main__":
    main()
