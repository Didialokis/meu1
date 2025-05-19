# --- Imports Essenciais ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets
import tokenizers
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm
import random
from pathlib import Path
import sys
import math
import argparse # Para argumentos de linha de comando

# Métricas para benchmarks
from sklearn.metrics import accuracy_score, f1_score, classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

# --- Definições de Classes (Movidas para o topo para escopo global) ---

# Classes do Dataset (BERTMLMDataset, ASSINNLIDataset, HAREMNERDataset)
class BERTMLMDataset(Dataset):
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
            if random.random() < 0.15:
                act_prob = random.random()
                if act_prob < 0.8: inputs[i] = self.mask_id
                elif act_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
            else: labels[i] = self.pad_id
        return torch.tensor(inputs), torch.tensor(labels)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.sentences[idx])
        masked_ids, mlm_labels = self._mask_tokens(enc.ids)
        return {"input_ids": masked_ids, "attention_mask": torch.tensor(enc.attention_mask, dtype=torch.long), 
                "segment_ids": torch.zeros_like(masked_ids, dtype=torch.long), "labels": mlm_labels}

class ASSINNLIDataset(Dataset):
    def __init__(self, hf_ds, tokenizer_instance, max_len_config):
        self.tok, self.max_len = tokenizer_instance, max_len_config; self.data = []
        self.label_map = {"ENTAILMENT": 0, "NONE": 1, "PARAPHRASE": 2}; self.num_classes = 3
        proc_count, skip_count = 0,0
        feature_ej = hf_ds.features.get('entailment_judgment') if hasattr(hf_ds, 'features') else None
        
        print(f"Processando {len(hf_ds)} exemplos ASSIN NLI para o Dataset...")
        for ex in tqdm(hf_ds, desc="Processando ASSIN NLI", file=sys.stdout, mininterval=1.0, leave=False):
            p, h, j_val = ex.get("premise"), ex.get("hypothesis"), ex.get("entailment_judgment")
            j_str = None
            if isinstance(j_val, str): j_str = j_val
            elif isinstance(j_val, int) and feature_ej:
                try: j_str = feature_ej.int2str(j_val)
                except: pass
            if isinstance(p,str) and p.strip() and isinstance(h,str) and h.strip() and j_str in self.label_map:
                self.data.append({"p": p.strip(), "h": h.strip(), "lbl": self.label_map[j_str]}); proc_count+=1
            else: skip_count+=1
        print(f"ASSIN NLI Dataset: {proc_count} processados, {skip_count} ignorados.")
        if proc_count == 0 and len(hf_ds) > 0: print("AVISO: Nenhum dado ASSIN NLI processado.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]; enc = self.tok.encode(item["p"], item["h"])
        ids, seg_ids_list = enc.ids, [0]*len(enc.ids)
        try:
            sep_idx = ids.index(self.tok.token_to_id("</s>"),1)
            for i in range(sep_idx + 1, len(ids)): 
                if ids[i] != self.tok.token_to_id("<pad>"): seg_ids_list[i] = 1
                else: break
        except ValueError: pass
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(enc.attention_mask,dtype=torch.long),
                "segment_ids":torch.tensor(seg_ids_list,dtype=torch.long),"labels":torch.tensor(item["lbl"],dtype=torch.long)}

class HAREMNERDataset(Dataset):
    def __init__(self, hf_ds, tokenizer_instance, id2label_map_config, max_len_config, pad_label_id_config):
        self.tok = tokenizer_instance; self.id2lbl = id2label_map_config
        self.max_len = max_len_config; self.pad_lbl_id = pad_label_id_config
        self.encs, self.lbls_aligned = [], []
        proc_count, skip_count = 0,0
        print(f"Processando {len(hf_ds)} exemplos HAREM para o Dataset...")
        for i, ex in enumerate(tqdm(hf_ds, desc="Processando HAREM", file=sys.stdout, mininterval=1.0, leave=False)):
            orig_toks, orig_ner_ids = ex.get("tokens"), ex.get("ner_tags")
            if not orig_toks or orig_ner_ids is None or len(orig_toks)!=len(orig_ner_ids): skip_count+=1; continue
            enc = self.tok.encode(" ".join(orig_toks)); word_ids = enc.word_ids; aligned_lbls = []
            prev_word_idx = None; err_align = False
            for word_idx in word_ids:
                if word_idx is None: aligned_lbls.append(self.pad_lbl_id)
                elif word_idx != prev_word_idx:
                    if 0 <= word_idx < len(orig_ner_ids): aligned_lbls.append(orig_ner_ids[word_idx])
                    else: aligned_lbls.append(self.pad_lbl_id); err_align=True; break 
                else: aligned_lbls.append(self.pad_lbl_id)
                prev_word_idx = word_idx
            if err_align or len(enc.ids) != len(aligned_lbls): skip_count+=1; continue
            self.encs.append(enc); self.lbls_aligned.append(aligned_lbls); proc_count+=1
        print(f"HAREM Dataset: {proc_count} processados, {skip_count} ignorados.")
        if proc_count == 0 and len(hf_ds) > 0: print("AVISO SÉRIO: Nenhum dado HAREM processado.")
    def __len__(self): return len(self.encs)
    def __getitem__(self, idx):
        enc, lbls = self.encs[idx], self.lbls_aligned[idx]
        return {"input_ids":torch.tensor(enc.ids),"attention_mask":torch.tensor(enc.attention_mask,dtype=torch.long),
                "segment_ids":torch.zeros(len(enc.ids),dtype=torch.long),"labels":torch.tensor(lbls,dtype=torch.long)}

# Classes do Modelo BERT
class BERTEmbedding(nn.Module):
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
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_len_config, pad_token_id, dropout_prob=0.1):
        super().__init__()
        self.emb = BERTEmbedding(vocab_size, hidden_size, max_len_config, dropout_prob, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size, 
                                             dropout=dropout_prob, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, input_ids, attention_mask, segment_ids):
        x = self.emb(input_ids, segment_ids)
        return self.enc(x, src_key_padding_mask=(attention_mask == 0))

class BERTLM(nn.Module):
    def __init__(self, bert_base: BERTBaseModel, vocab_size, hidden_size):
        super().__init__(); self.bert_base = bert_base
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
    def forward(self, input_ids, attention_mask, segment_ids):
        return self.mlm_head(self.bert_base(input_ids, attention_mask, segment_ids))

class BERTForSequencePairClassification(nn.Module):
    def __init__(self, bert_base: BERTBaseModel, hidden_size, num_classes):
        super().__init__(); self.bert_base = bert_base; self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(hidden_size, num_classes)
    def forward(self, input_ids, attention_mask, segment_ids):
        return self.clf(self.drop(self.bert_base(input_ids, attention_mask, segment_ids)[:, 0, :]))

class BERTForTokenClassification(nn.Module):
    def __init__(self, bert_base: BERTBaseModel, hidden_size, num_labels):
        super().__init__(); self.bert_base = bert_base; self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, segment_ids):
        return self.clf(self.drop(self.bert_base(input_ids, attention_mask, segment_ids)))

# Classe Trainer
class GenericTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 model_save_path, task_type="mlm", vocab_size_for_loss=None, id2label=None, log_freq=20):
        self.model, self.train_dl, self.val_dl = model.to(device), train_dataloader, val_dataloader
        self.opt, self.crit, self.dev = optimizer, criterion, device
        self.save_path = model_save_path
        self.best_val_metric = float('inf') if task_type in ["mlm", "nli"] else float('-inf') # Para F1 de NER, maior é melhor
        self.task_type, self.vocab_size, self.id2label = task_type, vocab_size_for_loss, id2label
        self.log_freq = log_freq
        if task_type == "ner" and id2label is None: raise ValueError("id2label é necessário para task_type='ner'")

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None, {}
        total_loss, all_preds_epoch, all_labels_epoch = 0.0, [], []
        desc = f"Epoch {epoch_num+1} [{'Train' if is_training else 'Val'}] ({self.task_type})"
        # Garantir que len(dl) é usado no total do tqdm
        progress_bar = tqdm(dl, total=len(dl) if hasattr(dl, '__len__') else None, desc=desc, file=sys.stdout)


        for i_batch, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.dev); attention_mask = batch["attention_mask"].to(self.dev)
            segment_ids = batch.get("segment_ids", torch.zeros_like(input_ids)).to(self.dev)
            labels = batch["labels"].to(self.dev)

            if is_training: self.opt.zero_grad()
            with torch.set_grad_enabled(is_training):
                logits = self.model(input_ids, attention_mask, segment_ids)
                if self.task_type in ["mlm", "ner"]:
                    loss = self.crit(logits.view(-1, logits.size(-1)), labels.view(-1))
                elif self.task_type == "nli":
                    loss = self.crit(logits, labels)
                else: loss = torch.tensor(0.0, device=self.dev)
            
            if is_training: loss.backward(); self.opt.step()
            total_loss += loss.item()
            if (i_batch + 1) % self.log_freq == 0:
                progress_bar.set_postfix({"loss": f"{total_loss / (i_batch + 1):.4f}"})

            if not is_training: # Coletar predições para métricas de validação
                if self.task_type == "nli":
                    all_preds_epoch.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    all_labels_epoch.extend(labels.cpu().numpy())
                elif self.task_type == "ner":
                    preds_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                    labels_ids = labels.cpu().numpy()
                    for p_seq, l_seq, m_seq in zip(preds_ids, labels_ids, attention_mask.cpu().numpy()):
                        # Considera apenas tokens reais (mask == 1) e não-ignorados no loss
                        # self.crit.ignore_index é o PAD_TOKEN_LABEL_ID_NER (-100) para NER
                        all_preds_epoch.append([self.id2label[p] for p,l,m in zip(p_seq,l_seq,m_seq) if m==1 and l!=self.crit.ignore_index])
                        all_labels_epoch.append([self.id2label[l] for l,m in zip(l_seq,m_seq) if m==1 and l!=self.crit.ignore_index])
        
        # Evitar divisão por zero se dataloader estiver vazio
        avg_loss = total_loss / len(dl) if len(dl) > 0 else 0.0
        metrics = {"loss": avg_loss}

        if not is_training and all_labels_epoch:
            if self.task_type == "nli" and all_preds_epoch:
                metrics["accuracy"] = accuracy_score(all_labels_epoch, all_preds_epoch)
                metrics["f1_w"] = f1_score(all_labels_epoch, all_preds_epoch, average='weighted', zero_division=0)
            elif self.task_type == "ner":
                preds_filt = [p for p in all_preds_epoch if p]; labels_filt = [l for l in all_labels_epoch if l]
                if preds_filt and labels_filt:
                    try:
                        report = seqeval_classification_report(labels_filt, preds_filt, output_dict=True, zero_division=0, mode='strict', scheme=IOB2)
                        metrics["f1_ner"] = report['micro avg']['f1-score'] # Usando micro F1 para NER
                    except Exception as e_seqeval: print(f"Erro no cálculo de métricas NER com seqeval: {e_seqeval}")

        print_line = f"{desc} - Avg Loss: {avg_loss:.4f}"
        for k, v_metric in metrics.items():
            if k != "loss": print_line += f", {k}: {v_metric:.4f}"
        print(print_line)
        return avg_loss, metrics

    def train(self, num_epochs):
        metric_to_watch = "loss"; higher_is_better = False
        if self.task_type == "nli" and self.val_dl: metric_to_watch = "f1_w"; higher_is_better = True
        elif self.task_type == "ner" and self.val_dl: metric_to_watch = "f1_ner"; higher_is_better = True
        
        print(f"Treinando ({self.task_type}) por {num_epochs} épocas. Observando '{metric_to_watch}'.")
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            _, val_metrics = self._run_epoch(epoch, is_training=False)
            
            current_val = val_metrics.get(metric_to_watch) if val_metrics else None
            if current_val is not None:
                improved = (current_val < self.best_val_metric) if not higher_is_better else (current_val > self.best_val_metric)
                if improved:
                    self.best_val_metric = current_val
                    print(f"Melhor métrica ({metric_to_watch}): {self.best_val_metric:.4f}. Salvando modelo: {self.save_path}")
                    torch.save(self.model.state_dict(), self.save_path)
            elif epoch == num_epochs - 1 and not self.val_dl:
                print(f"({self.task_type}) Sem validação. Salvando modelo da última época: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path)
            print("-" * 30)
        
        best_val_display = "N/A"
        if isinstance(self.best_val_metric, (int, float)):
            if math.isinf(self.best_val_metric): best_val_display = "N/A (inf)"
            elif math.isnan(self.best_val_metric): best_val_display = "N/A (NaN)"
            else: best_val_display = f"{self.best_val_metric:.4f}"
        
        print(f"Treinamento ({self.task_type}) concluído. Melhor métrica ({metric_to_watch}): {best_val_display}")
        if not self.val_dl and num_epochs > 0 and Path(self.save_path).exists():
             print(f"Modelo final salvo em: {self.save_path}")

# --- Funções para cada Fase ---
def setup_environment_and_tokenizer(args):
    """Carrega dados Aroeira, treina/carrega tokenizador."""
    global tokenizer, PAD_TOKEN_ID, all_aroeira_sentences # Para serem acessíveis por outras funções
    
    all_aroeira_sentences = []
    try:
        streamed_aroeira_dataset = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        text_column_name = "text"
        
        aroeira_examples_collected = []
        source_iterable = streamed_aroeira_dataset
        desc_tqdm_aroeira = "Extraindo sentenças Aroeira"
        tqdm_total_aroeira = None

        if args.aroeira_subset_size is not None:
            stream_iterator = iter(streamed_aroeira_dataset)
            print(f"Coletando {args.aroeira_subset_size} exemplos do Aroeira stream...")
            try:
                for _ in range(args.aroeira_subset_size): aroeira_examples_collected.append(next(stream_iterator))
            except StopIteration: print(f"Alerta: Stream Aroeira esgotado. Coletados {len(aroeira_examples_collected)}.")
            source_iterable = aroeira_examples_collected
            desc_tqdm_aroeira += " (subconjunto)"
            tqdm_total_aroeira = len(aroeira_examples_collected)
        else:
            desc_tqdm_aroeira += " (stream completo)"
            print("AVISO: Processando stream completo do Aroeira.")

        for example in tqdm(source_iterable, total=tqdm_total_aroeira, desc=desc_tqdm_aroeira, file=sys.stdout):
            sentence = example.get(text_column_name)
            if isinstance(sentence, str) and sentence.strip(): all_aroeira_sentences.append(sentence.strip())

        if not all_aroeira_sentences: raise ValueError("Nenhuma sentença Aroeira foi extraída.")
        print(f"Total de sentenças Aroeira extraídas: {len(all_aroeira_sentences)}")

        temp_file = Path(args.temp_tokenizer_train_file)
        if temp_file.exists(): temp_file.unlink()
        with open(temp_file, "w", encoding="utf-8") as fp:
            for sentence in all_aroeira_sentences: fp.write(sentence + "\n")
        tokenizer_train_files_list = [str(temp_file)]
        print(f"Arquivo temporário para tokenizador: {tokenizer_train_files_list[0]}")

    except Exception as e: print(f"Erro ao preparar dados Aroeira: {e}"); import traceback; traceback.print_exc(); raise

    vocab_file = Path(".") / args.tokenizer_vocab_filename
    merges_file = Path(".") / args.tokenizer_merges_filename

    if not vocab_file.exists() or not merges_file.exists():
        if not tokenizer_train_files_list or not Path(tokenizer_train_files_list[0]).exists():
            raise FileNotFoundError("Arquivo de treino para tokenizador não encontrado.")
        tokenizer_save_prefix = args.tokenizer_vocab_filename.replace("-vocab.json", "")
        custom_tokenizer_obj = ByteLevelBPETokenizer(lowercase=True)
        print(f"Treinando tokenizador com {tokenizer_train_files_list}...")
        custom_tokenizer_obj.train(files=tokenizer_train_files_list, vocab_size=args.vocab_size, 
                                   min_frequency=args.min_frequency_tokenizer, 
                                   special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        custom_tokenizer_obj.save_model(".", prefix=tokenizer_save_prefix)
        print(f"Tokenizador treinado e salvo.")
    else: print(f"Tokenizador já existe. Carregando...")

    tokenizer = ByteLevelBPETokenizer(vocab=args.tokenizer_vocab_filename, merges=args.tokenizer_merges_filename, lowercase=True)
    tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")))
    tokenizer.enable_truncation(max_length=args.max_len)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=args.max_len)
    PAD_TOKEN_ID = tokenizer.token_to_id("<pad>")
    print(f"Vocabulário do Tokenizador: {tokenizer.get_vocab_size()}, PAD ID: {PAD_TOKEN_ID}")
    return tokenizer, PAD_TOKEN_ID


def run_pretraining_mlm(args, local_tokenizer, local_pad_token_id, local_all_aroeira_sentences):
    print("\n--- Configurando Pré-Treinamento MLM ---")
    val_split_mlm = 0.1
    num_val_mlm = int(len(local_all_aroeira_sentences) * val_split_mlm)
    if num_val_mlm < 1 and len(local_all_aroeira_sentences) > 1: num_val_mlm = 1
    
    train_sents_mlm = local_all_aroeira_sentences[num_val_mlm:]
    val_sents_mlm = local_all_aroeira_sentences[:num_val_mlm]
    if not train_sents_mlm: train_sents_mlm = local_all_aroeira_sentences; val_sents_mlm = []

    train_ds_mlm = BERTMLMDataset(train_sents_mlm, local_tokenizer, max_len=args.max_len)
    train_dl_mlm = DataLoader(train_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    val_dl_mlm = None
    if val_sents_mlm and len(val_sents_mlm) > 0:
        val_ds_mlm = BERTMLMDataset(val_sents_mlm, local_tokenizer, max_len=args.max_len)
        if len(val_ds_mlm) > 0: val_dl_mlm = DataLoader(val_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)

    bert_base_pt = BERTBaseModel(
        local_tokenizer.get_vocab_size(), args.model_hidden_size, args.model_num_layers, 
        args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
        local_pad_token_id, dropout_prob=0.1 # Usar args.dropout_prob se definido
    )
    bertlm_pt_model = BERTLM(bert_base_pt, local_tokenizer.get_vocab_size(), args.model_hidden_size)
    opt_mlm = torch.optim.AdamW(bertlm_pt_model.parameters(), lr=args.lr_pretrain, 
                                betas=(0.9,0.999), eps=1e-6, weight_decay=0.01) # Usar args para betas etc.
    crit_mlm = nn.CrossEntropyLoss(ignore_index=local_pad_token_id)
    
    trainer_mlm = GenericTrainer(
        bertlm_pt_model, train_dl_mlm, val_dl_mlm, opt_mlm, crit_mlm, args.device, 
        args.pretrained_bertlm_save_filename, "mlm", local_tokenizer.get_vocab_size()
    )
    print("Pré-treinamento MLM configurado. Iniciando...")
    try:
        trainer_mlm.train(num_epochs=args.epochs_pretrain)
    except Exception as e: print(f"Erro pré-treinamento MLM: {e}"); import traceback; traceback.print_exc(); raise


def run_finetune_nli(args, local_tokenizer, local_pad_token_id):
    print("\n--- Configurando Fine-tuning ASSIN (NLI/RTE) ---")
    # Constantes específicas do benchmark (podem vir de args)
    ASSIN_NLI_SAVE_PATH = Path(".") / args.assin_nli_model_save_filename # Usar Path
    ASSIN_SUBSET_TRAIN_SIZE = args.assin_subset_train
    ASSIN_SUBSET_VAL_SIZE = args.assin_subset_val
    try:
        assin_full = datasets.load_dataset("assin", name="ptbr", trust_remote_code=args.trust_remote_code)
        raw_train_assin = assin_full["train"]; raw_val_assin = assin_full["validation"]
        def filter_ej(ex): return ex.get('entailment_judgment') is not None
        train_assin_ej = raw_train_assin.filter(filter_ej); val_assin_ej = raw_val_assin.filter(filter_ej)
        
        train_assin_sub = train_assin_ej.select(range(min(ASSIN_SUBSET_TRAIN_SIZE,len(train_assin_ej)))) if len(train_assin_ej)>0 else train_assin_ej
        val_assin_sub = val_assin_ej.select(range(min(ASSIN_SUBSET_VAL_SIZE,len(val_assin_ej)))) if len(val_assin_ej)>0 else val_assin_ej
        print(f"ASSIN NLI Treino (subset): {len(train_assin_sub)}, Val (subset): {len(val_assin_sub)}")

        if len(train_assin_sub) == 0: print("Nenhum dado de treino para ASSIN NLI. Pulando."); return

        ft_train_ds_nli = ASSINNLIDataset(train_assin_sub, local_tokenizer, args.max_len)
        if len(ft_train_ds_nli)==0: raise ValueError("Dataset de treino ASSIN NLI vazio.")
        ft_train_dl_nli = DataLoader(ft_train_ds_nli, batch_size=args.finetune_batch_size, shuffle=True, num_workers=0)
        ft_val_dl_nli = None
        if len(val_assin_sub) > 0:
            ft_val_ds_nli = ASSINNLIDataset(val_assin_sub, local_tokenizer, args.max_len)
            if len(ft_val_ds_nli) > 0: ft_val_dl_nli = DataLoader(ft_val_ds_nli, batch_size=args.finetune_batch_size, shuffle=False, num_workers=0)
        
        base_nli = BERTBaseModel(local_tokenizer.get_vocab_size(), args.model_hidden_size, args.model_num_layers, 
                                 args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
                                 local_pad_token_id, dropout_prob=0.1) # Usar args.dropout_prob
        if Path(args.pretrained_bertlm_save_filename).exists():
            state = torch.load(args.pretrained_bertlm_save_filename, map_location=args.device)
            base_state = {k.replace("bert_base.",""):v for k,v in state.items() if k.startswith("bert_base.")}
            if base_state: base_nli.load_state_dict(base_state); print("Backbone BERT (MLM) carregado para NLI.")
        
        model_nli = BERTForSequencePairClassification(base_nli, args.model_hidden_size, ft_train_ds_nli.num_classes).to(args.device)
        opt_nli = torch.optim.AdamW(model_nli.parameters(), lr=args.finetune_lr) # Usar args para betas etc.
        crit_nli = nn.CrossEntropyLoss()
        trainer_nli = GenericTrainer(model_nli, ft_train_dl_nli, ft_val_dl_nli, opt_nli, crit_nli, args.device, 
                                   ASSIN_NLI_SAVE_PATH, "nli")
        print("Fine-tuning NLI configurado. Iniciando...")
        trainer_nli.train(num_epochs=args.finetune_epochs)
    except Exception as e: print(f"Erro Fine-tuning ASSIN NLI: {e}"); import traceback; traceback.print_exc()


def run_finetune_ner(args, local_tokenizer, local_pad_token_id):
    print("\n--- Configurando Fine-tuning HAREM (NER) ---")
    HAREM_SAVE_PATH = Path(".") / args.harem_model_save_filename # Usar Path
    HAREM_SUBSET_SIZE_CONFIG = args.harem_subset_size
    PAD_NER_LABEL_ID = args.pad_token_label_id_ner # Da config
    try:
        harem_raw = datasets.load_dataset("harem", "selective", trust_remote_code=args.trust_remote_code)
        ner_feat = harem_raw["train"].features["ner_tags"].feature
        id2lbl_harem = {i:n for i,n in enumerate(ner_feat.names)}; NUM_NER_TAGS = len(id2lbl_harem)
        print(f"HAREM Tags (primeiras 5 de {NUM_NER_TAGS}): {list(id2lbl_harem.items())[:5]}")
        train_harem_full = harem_raw["train"]
        train_harem_sub = train_harem_full
        if HAREM_SUBSET_SIZE_CONFIG and HAREM_SUBSET_SIZE_CONFIG < len(train_harem_full):
            train_harem_sub = train_harem_full.select(range(HAREM_SUBSET_SIZE_CONFIG))
        print(f"HAREM Treino (subset): {len(train_harem_sub)}")
        
        ft_train_ds_ner = HAREMNERDataset(train_harem_sub, local_tokenizer, id2lbl_harem, args.max_len, PAD_NER_LABEL_ID)
        if len(ft_train_ds_ner) == 0: raise ValueError("Dataset HAREM vazio após processamento.")
        ft_train_dl_ner = DataLoader(ft_train_ds_ner, batch_size=args.finetune_batch_size, shuffle=True, num_workers=0)
        ft_val_dl_ner = None # HAREM selective não tem val_split oficial.

        base_ner = BERTBaseModel(local_tokenizer.get_vocab_size(),args.model_hidden_size,args.model_num_layers,
                                 args.model_num_attention_heads,args.model_intermediate_size,args.max_len,
                                 local_pad_token_id,dropout_prob=0.1) # Usar args.dropout_prob
        if Path(args.pretrained_bertlm_save_filename).exists():
            state = torch.load(args.pretrained_bertlm_save_filename, map_location=args.device)
            base_state = {k.replace("bert_base.",""):v for k,v in state.items() if k.startswith("bert_base.")}
            if base_state: base_ner.load_state_dict(base_state); print("Backbone BERT (MLM) carregado para NER.")
                
        model_ner = BERTForTokenClassification(base_ner, args.model_hidden_size, NUM_NER_TAGS).to(args.device)
        opt_ner = torch.optim.AdamW(model_ner.parameters(), lr=args.finetune_lr) # Usar args para betas etc.
        crit_ner = nn.CrossEntropyLoss(ignore_index=PAD_NER_LABEL_ID)
        trainer_ner = GenericTrainer(model_ner, ft_train_dl_ner, ft_val_dl_ner, opt_ner, crit_ner, args.device, 
                                   HAREM_SAVE_PATH, "ner", NUM_NER_TAGS, id2lbl_harem)
        print("Fine-tuning NER configurado. Iniciando...")
        trainer_ner.train(num_epochs=args.finetune_epochs)
    except Exception as e: print(f"Erro Fine-tuning HAREM NER: {e}"); import traceback; traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino e Fine-tuning BERT no Aroeira.")
    # Configs Gerais
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aroeira_subset_size", type=int, default=None, help="Número de exemplos do Aroeira. None para dataset completo.")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument("--trust_remote_code", action='store_true', help="Confiar no código remoto para datasets.")

    # Configs Pré-treinamento MLM
    parser.add_argument("--epochs_pretrain", type=int, default=1)
    parser.add_argument("--batch_size_pretrain", type=int, default=8)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)

    # Arquitetura Modelo BERT
    parser.add_argument("--model_hidden_size", type=int, default=256)
    parser.add_argument("--model_num_layers", type=int, default=2)
    parser.add_argument("--model_num_attention_heads", type=int, default=4)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    # model_intermediate_size será hidden_size * 4

    # Nomes de Arquivos/Paths
    parser.add_argument("--output_dir", type=str, default=".", help="Diretório para salvar outputs.")
    parser.add_argument("--tokenizer_vocab_filename", type=str, default="aroeira_mlm_tokenizer-vocab.json")
    parser.add_argument("--tokenizer_merges_filename", type=str, default="aroeira_mlm_tokenizer-merges.txt")
    parser.add_argument("--pretrained_bertlm_save_filename", type=str, default="aroeira_bertlm_pretrained.pth")
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_tokenizer.txt")
    
    # Configs Fine-tuning
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--finetune_batch_size", type=int, default=8)
    parser.add_argument("--finetune_lr", type=float, default=3e-5)
    
    parser.add_argument("--assin_nli_model_save_filename", type=str, default="aroeira_assin_nli_model.pth")
    parser.add_argument("--assin_subset_train", type=int, default=500)
    parser.add_argument("--assin_subset_val", type=int, default=100)

    parser.add_argument("--harem_model_save_filename", type=str, default="aroeira_harem_ner_model.pth")
    parser.add_argument("--harem_subset_size", type=int, default=250)
    parser.add_argument("--pad_token_label_id_ner", type=int, default=-100)


    # Controle de Fluxo
    parser.add_argument("--do_dataprep_tokenizer", action='store_true', help="Executar preparação de dados Aroeira e treino do tokenizador.")
    parser.add_argument("--do_pretrain", action='store_true', help="Executar pré-treinamento MLM.")
    parser.add_argument("--do_finetune_nli", action='store_true', help="Executar fine-tuning NLI no ASSIN.")
    parser.add_argument("--do_finetune_ner", action='store_true', help="Executar fine-tuning NER no HAREM.")
    
    args = parser.parse_args()
    
    # Ajustes pós-parsing
    if args.device is None: args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_intermediate_size = args.model_hidden_size * 4 # Derivado
        
    # Ajustar nomes de arquivo para incluir output_dir
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True) # Cria o diretório de saída

    args.tokenizer_vocab_filename = str(output_dir_path / args.tokenizer_vocab_filename)
    args.tokenizer_merges_filename = str(output_dir_path / args.tokenizer_merges_filename)
    args.pretrained_bertlm_save_filename = str(output_dir_path / args.pretrained_bertlm_save_filename)
    args.temp_tokenizer_train_file = str(output_dir_path / args.temp_tokenizer_train_file)
    args.assin_nli_model_save_filename = str(output_dir_path / args.assin_nli_model_save_filename)
    args.harem_model_save_filename = str(output_dir_path / args.harem_model_save_filename)

    return args

# --- Função Principal e Execução ---
def main():
    ARGS = parse_args() # Parseia os argumentos da linha de comando
    
    # Imprime as configurações que serão usadas
    print("--- Configurações Atuais ---")
    for arg, value in vars(ARGS).items():
        print(f"{arg}: {value}")
    print("----------------------------")

    # Variáveis que precisam ser passadas entre as fases
    current_tokenizer = None
    current_pad_token_id = None
    
    # Fase 1: Preparação de Dados Aroeira e Treinamento do Tokenizador (Blocos 3 e 4)
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain or ARGS.do_finetune_nli or ARGS.do_finetune_ner:
        # Tokenizer é necessário para todas as etapas subsequentes
        print("\n--- Fase: Preparação de Dados Aroeira e Tokenizador ---")
        # Para simplificar, a função setup_environment_and_tokenizer agora só retorna tokenizer e pad_id
        # e usa ARGS internamente para os caminhos e outras configs.
        # all_aroeira_sentences é carregado dentro dela e usado para treinar o tokenizador.
        # Se formos usar all_aroeira_sentences para o pré-treino, precisamos retorná-lo também.
        
        # Reestruturando para ter `all_aroeira_sentences` disponível se necessário
        _all_aroeira_sentences_for_pretrain = [] # Escopo mais amplo
        
        # Sub-função para Bloco 3 e 4
        def prepare_data_and_train_tokenizer(args_config):
            nonlocal _all_aroeira_sentences_for_pretrain # Para modificar a lista no escopo de main
            _all_aroeira_sentences = [] # Local para esta função
            try:
                streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args_config.trust_remote_code)
                text_col = "text"
                collected_ex = []
                src_iter = streamed_ds
                desc_tqdm = "Extraindo sentenças Aroeira"
                total_tqdm = None

                if args_config.aroeira_subset_size is not None:
                    iterator = iter(streamed_ds)
                    print(f"Coletando {args_config.aroeira_subset_size} exemplos Aroeira...")
                    try:
                        for _ in range(args_config.aroeira_subset_size): collected_ex.append(next(iterator))
                    except StopIteration: print(f"Alerta: Stream Aroeira esgotado. Coletados {len(collected_ex)}.")
                    src_iter = collected_ex
                    desc_tqdm += " (subconjunto)"
                    total_tqdm = len(collected_ex)
                else:
                    desc_tqdm += " (stream completo)"
                
                for ex in tqdm(src_iter, total=total_tqdm, desc=desc_tqdm, file=sys.stdout):
                    sent = ex.get(text_col)
                    if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences.append(sent.strip())

                if not _all_aroeira_sentences: raise ValueError("Nenhuma sentença Aroeira extraída.")
                _all_aroeira_sentences_for_pretrain.extend(_all_aroeira_sentences) # Popula a lista do escopo externo
                print(f"Total de sentenças Aroeira extraídas: {len(_all_aroeira_sentences)}")

                temp_file = Path(args_config.temp_tokenizer_train_file)
                if temp_file.exists(): temp_file.unlink()
                with open(temp_file, "w", encoding="utf-8") as f:
                    for s in _all_aroeira_sentences: f.write(s + "\n")
                train_files = [str(temp_file)]
                
                vocab_f = Path(args_config.tokenizer_vocab_filename)
                merges_f = Path(args_config.tokenizer_merges_filename)

                if not vocab_f.exists() or not merges_f.exists():
                    if not train_files or not Path(train_files[0]).exists():
                        raise FileNotFoundError("Arquivo de treino para tokenizador não encontrado.")
                    prefix = args_config.tokenizer_vocab_filename.replace("-vocab.json", "")
                    tok_model = ByteLevelBPETokenizer(lowercase=True)
                    print(f"Treinando tokenizador com {train_files}...")
                    tok_model.train(files=train_files, vocab_size=args_config.vocab_size, 
                                      min_frequency=args_config.min_frequency_tokenizer, 
                                      special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
                    tok_model.save_model(str(args_config.output_dir), prefix=prefix) # Salva em output_dir
                else: print("Tokenizador já existe. Carregando...")
                
                loaded_tok = ByteLevelBPETokenizer(vocab=args_config.tokenizer_vocab_filename, merges=args_config.tokenizer_merges_filename, lowercase=True)
                loaded_tok._tokenizer.post_processor = BertProcessing(("</s>", loaded_tok.token_to_id("</s>")), ("<s>", loaded_tok.token_to_id("<s>")))
                loaded_tok.enable_truncation(max_length=args_config.max_len)
                loaded_tok.enable_padding(pad_id=loaded_tok.token_to_id("<pad>"), pad_token="<pad>", length=args_config.max_len)
                pad_id = loaded_tok.token_to_id("<pad>")
                print(f"Vocabulário do Tokenizador: {loaded_tok.get_vocab_size()}, PAD ID: {pad_id}")
                return loaded_tok, pad_id
            except Exception as e_tok: print(f"Erro na preparação de dados/tokenizador: {e_tok}"); import traceback; traceback.print_exc(); raise

        current_tokenizer, current_pad_token_id = prepare_data_and_train_tokenizer(ARGS)

    # Fase 2: Pré-Treinamento MLM (Blocos 5, 6, 7, 8, 9)
    if ARGS.do_pretrain:
        if not _all_aroeira_sentences_for_pretrain or not current_tokenizer:
            print("ERRO: Dados do Aroeira ou tokenizador não preparados para pré-treinamento. Execute --do_dataprep_tokenizer.")
        else:
            print("\n--- Fase: Pré-Treinamento MLM ---")
            val_s_mlm_ratio = 0.1; num_v_mlm = int(len(_all_aroeira_sentences_for_pretrain) * val_s_mlm_ratio)
            if num_v_mlm < 1 and len(_all_aroeira_sentences_for_pretrain) > 1: num_v_mlm = 1
            tr_s_mlm = _all_aroeira_sentences_for_pretrain[num_v_mlm:]; v_s_mlm = _all_aroeira_sentences_for_pretrain[:num_v_mlm]
            if not tr_s_mlm: tr_s_mlm = _all_aroeira_sentences_for_pretrain; v_s_mlm = []

            train_ds_mlm = BERTMLMDataset(tr_s_mlm, current_tokenizer, ARGS.max_len)
            train_dl_mlm = DataLoader(train_ds_mlm, batch_size=ARGS.batch_size_pretrain, shuffle=True, num_workers=0)
            val_dl_mlm = None
            if v_s_mlm and len(v_s_mlm) > 0:
                val_ds_mlm = BERTMLMDataset(v_s_mlm, current_tokenizer, ARGS.max_len)
                if len(val_ds_mlm) > 0: val_dl_mlm = DataLoader(val_ds_mlm, batch_size=ARGS.batch_size_pretrain, shuffle=False, num_workers=0)

            bert_base = BERTBaseModel(current_tokenizer.get_vocab_size(), ARGS.model_hidden_size, ARGS.model_num_layers, 
                                     ARGS.model_num_attention_heads, ARGS.model_intermediate_size, ARGS.max_len, 
                                     current_pad_token_id, dropout_prob=ARGS.model_dropout_prob)
            bertlm_model = BERTLM(bert_base, current_tokenizer.get_vocab_size(), ARGS.model_hidden_size)
            opt_mlm = torch.optim.AdamW(bertlm_model.parameters(), lr=ARGS.lr_pretrain, 
                                        betas=(0.9,0.999), eps=1e-6, weight_decay=0.01) # Usar args
            crit_mlm = nn.CrossEntropyLoss(ignore_index=current_pad_token_id)
            
            trainer_mlm = GenericTrainer(
                bertlm_model, train_dl_mlm, val_dl_mlm, opt_mlm, crit_mlm, ARGS.device, 
                ARGS.pretrained_bertlm_save_filename, "mlm", current_tokenizer.get_vocab_size()
            )
            print("Pré-treinamento MLM configurado. Iniciando...")
            trainer_mlm.train(num_epochs=ARGS.epochs_pretrain)

    # Fase 3: Fine-tuning NLI (Bloco 10)
    if ARGS.do_finetune_nli:
        if not current_tokenizer: print("ERRO: Tokenizador não treinado/carregado. Execute --do_dataprep_tokenizer."); return
        print("\n--- Fase: Fine-tuning NLI (ASSIN) ---")
        
        ASSIN_SAVE_PATH = Path(ARGS.output_dir) / ARGS.assin_nli_model_save_filename

        try:
            assin_full = datasets.load_dataset("assin", name="ptbr", trust_remote_code=ARGS.trust_remote_code)
            raw_train_assin, raw_val_assin = assin_full["train"], assin_full["validation"]
            def filter_ej(ex): return ex.get('entailment_judgment') is not None
            train_assin_ej, val_assin_ej = raw_train_assin.filter(filter_ej), raw_val_assin.filter(filter_ej)
            
            train_assin_sub = train_assin_ej.select(range(min(ARGS.assin_subset_train,len(train_assin_ej)))) if len(train_assin_ej)>0 else train_assin_ej
            val_assin_sub = val_assin_ej.select(range(min(ARGS.assin_subset_val,len(val_assin_ej)))) if len(val_assin_ej)>0 else val_assin_ej
            print(f"ASSIN NLI Treino: {len(train_assin_sub)}, Val: {len(val_assin_sub)}")

            if len(train_assin_sub) > 0:
                ft_train_ds_nli = ASSINNLIDataset(train_assin_sub, current_tokenizer, ARGS.max_len)
                if len(ft_train_ds_nli)==0: raise ValueError("Dataset de treino ASSIN NLI vazio.")
                ft_train_dl_nli = DataLoader(ft_train_ds_nli, batch_size=ARGS.finetune_batch_size, shuffle=True, num_workers=0)
                ft_val_dl_nli = None
                if len(val_assin_sub) > 0:
                    ft_val_ds_nli = ASSINNLIDataset(val_assin_sub, current_tokenizer, ARGS.max_len)
                    if len(ft_val_ds_nli) > 0: ft_val_dl_nli = DataLoader(ft_val_ds_nli, batch_size=ARGS.finetune_batch_size, shuffle=False, num_workers=0)
                
                base_nli = BERTBaseModel(current_tokenizer.get_vocab_size(),ARGS.model_hidden_size,ARGS.model_num_layers,
                                         ARGS.model_num_attention_heads,ARGS.model_intermediate_size,ARGS.max_len,
                                         current_pad_token_id,dropout_prob=ARGS.model_dropout_prob)
                if Path(ARGS.pretrained_bertlm_save_filename).exists():
                    state = torch.load(ARGS.pretrained_bertlm_save_filename, map_location=ARGS.device)
                    base_state = {k.replace("bert_base.",""):v for k,v in state.items() if k.startswith("bert_base.")}
                    if base_state: base_nli.load_state_dict(base_state); print("Backbone BERT (MLM) carregado para NLI.")
                
                model_nli = BERTForSequencePairClassification(base_nli, ARGS.model_hidden_size, ft_train_ds_nli.num_classes).to(ARGS.device)
                opt_nli = torch.optim.AdamW(model_nli.parameters(), lr=ARGS.finetune_lr)
                crit_nli = nn.CrossEntropyLoss()
                trainer_nli = GenericTrainer(model_nli, ft_train_dl_nli, ft_val_dl_nli, opt_nli, crit_nli, ARGS.device, 
                                           ASSIN_SAVE_PATH, "nli")
                trainer_nli.train(num_epochs=ARGS.finetune_epochs)
            else: print("Nenhum dado de treino para ASSIN NLI. Pulando.")
        except Exception as e_nli: print(f"Erro Fine-tuning ASSIN NLI: {e_nli}"); import traceback; traceback.print_exc()

    # Fase 4: Fine-tuning NER (Bloco 11)
    if ARGS.do_finetune_ner:
        if not current_tokenizer: print("ERRO: Tokenizador não treinado/carregado. Execute --do_dataprep_tokenizer."); return
        print("\n--- Fase: Fine-tuning NER (HAREM) ---")
        
        HAREM_SAVE_PATH = Path(ARGS.output_dir) / ARGS.harem_model_save_filename
        PAD_NER_LABEL_ID_VAL = ARGS.pad_token_label_id_ner

        try:
            harem_raw = datasets.load_dataset("harem", "selective", trust_remote_code=ARGS.trust_remote_code)
            ner_feat = harem_raw["train"].features["ner_tags"].feature
            id2lbl_harem = {i:n for i,n in enumerate(ner_feat.names)}; NUM_NER_TAGS = len(id2lbl_harem)
            print(f"HAREM Tags (primeiras 5 de {NUM_NER_TAGS}): {list(id2lbl_harem.items())[:5]}")
            train_harem_full = harem_raw["train"]
            train_harem_sub = train_harem_full
            if ARGS.harem_subset_size and ARGS.harem_subset_size < len(train_harem_full):
                train_harem_sub = train_harem_full.select(range(ARGS.harem_subset_size))
            print(f"HAREM Treino (subset): {len(train_harem_sub)}")
            
            ft_train_ds_ner = HAREMNERDataset(train_harem_sub, current_tokenizer, id2lbl_harem, ARGS.max_len, PAD_NER_LABEL_ID_VAL)
            if len(ft_train_ds_ner) == 0: raise ValueError("Dataset HAREM vazio após processamento.")
            ft_train_dl_ner = DataLoader(ft_train_ds_ner, batch_size=ARGS.finetune_batch_size, shuffle=True, num_workers=0)
            ft_val_dl_ner = None 

            base_ner = BERTBaseModel(current_tokenizer.get_vocab_size(),ARGS.model_hidden_size,ARGS.model_num_layers,
                                     ARGS.model_num_attention_heads,ARGS.model_intermediate_size,ARGS.max_len,
                                     current_pad_token_id,dropout_prob=ARGS.model_dropout_prob)
            if Path(ARGS.pretrained_bertlm_save_filename).exists():
                state = torch.load(ARGS.pretrained_bertlm_save_filename, map_location=ARGS.device)
                base_state = {k.replace("bert_base.",""):v for k,v in state.items() if k.startswith("bert_base.")}
                if base_state: base_ner.load_state_dict(base_state); print("Backbone BERT (MLM) carregado para NER.")
                    
            model_ner = BERTForTokenClassification(base_ner, ARGS.model_hidden_size, NUM_NER_TAGS).to(ARGS.device)
            opt_ner = torch.optim.AdamW(model_ner.parameters(), lr=ARGS.finetune_lr)
            crit_ner = nn.CrossEntropyLoss(ignore_index=PAD_NER_LABEL_ID_VAL)
            trainer_ner = GenericTrainer(model_ner, ft_train_dl_ner, ft_val_dl_ner, opt_ner, crit_ner, ARGS.device, 
                                       HAREM_SAVE_PATH, "ner", NUM_NER_TAGS, id2lbl_harem)
            print("Fine-tuning NER configurado. Iniciando...")
            trainer_ner.train(num_epochs=ARGS.finetune_epochs)
        except Exception as e_ner: print(f"Erro Fine-tuning HAREM NER: {e_ner}"); import traceback; traceback.print_exc()
    
    print("\n--- Script Finalizado ---")

if __name__ == "__main__":
    main()
