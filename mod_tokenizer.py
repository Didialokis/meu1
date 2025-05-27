# train_pipeline.py
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
import json # Mantido caso seja usado em outro lugar, mas não para tokenizer_config.json

print(f"PyTorch: {torch.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")

# --- Definições de Classes ---

class BERTMLMDataset(Dataset):
    """Dataset para Masked Language Modeling."""
    def __init__(self, sentences_or_hf_dataset, tokenizer_instance: BertTokenizer, max_len_config: int, pad_token_id_val: int):
        self.tokenizer = tokenizer_instance; self.max_len = max_len_config
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_id = self.tokenizer.mask_token_id; self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id; self.pad_id = pad_token_id_val
        if isinstance(sentences_or_hf_dataset, datasets.Dataset):
            self.sentences = [ex['text'] for ex in sentences_or_hf_dataset if ex.get('text') and ex['text'].strip()]
        elif isinstance(sentences_or_hf_dataset, list): self.sentences = sentences_or_hf_dataset
        else: raise TypeError("Input para BERTMLMDataset deve ser lista de sentenças ou datasets.Dataset.")
        if not self.sentences: print("AVISO: BERTMLMDataset inicializado com zero sentenças.")
        print(f"BERTMLMDataset: {len(self.sentences)} sentenças carregadas.")
    def __len__(self): return len(self.sentences)
    def _mask_tokens(self, input_ids_list: list):
        inputs, labels = list(input_ids_list), list(input_ids_list)
        for i, token_id in enumerate(inputs):
            if token_id in [self.cls_id, self.sep_id, self.pad_id]: labels[i] = self.pad_id; continue
            if random.random() < 0.15:
                act_prob = random.random()
                if act_prob < 0.8: inputs[i] = self.mask_id
                elif act_prob < 0.9: inputs[i] = random.randrange(self.vocab_size)
            else: labels[i] = self.pad_id
        return torch.tensor(inputs), torch.tensor(labels)
    def __getitem__(self, idx):
        sentence_text = self.sentences[idx]
        encoding = self.tokenizer(sentence_text, add_special_tokens=True, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_attention_mask=True)
        input_ids_original_list = encoding['input_ids']
        attention_mask_list = encoding['attention_mask']
        masked_input_ids_tensor, mlm_labels_tensor = self._mask_tokens(input_ids_original_list)
        return {"input_ids": masked_input_ids_tensor, 
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long), 
                "segment_ids": torch.zeros_like(masked_input_ids_tensor, dtype=torch.long), 
                "labels": mlm_labels_tensor}

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
    def __init__(self, bert_base: BERTBaseModel, vocab_size, hidden_size):
        super().__init__(); self.bert_base = bert_base
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
    def forward(self, input_ids, attention_mask, segment_ids):
        return self.mlm_head(self.bert_base(input_ids, attention_mask, segment_ids))

class SimplifiedTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 model_save_path, vocab_size_for_loss, log_freq=20):
        self.model, self.train_dl, self.val_dl = model.to(device), train_dataloader, val_dataloader
        self.opt, self.crit, self.dev = optimizer, criterion, device
        self.save_path = Path(model_save_path); self.best_val_loss = float('inf')
        self.vocab_size = vocab_size_for_loss; self.log_freq = log_freq

    def _run_epoch(self, epoch_num, is_training):
        self.model.train(is_training); dl = self.train_dl if is_training else self.val_dl
        if not dl: return None
        total_loss = 0.0
        desc = f"Epoch {epoch_num+1} [{'Train' if is_training else 'Val'}] (MLM)"
        progress_bar = tqdm(dl, total=len(dl) if hasattr(dl, '__len__') else None, desc=desc, file=sys.stdout)
        for i_batch, batch in enumerate(progress_bar):
            input_ids=batch["input_ids"].to(self.dev); attention_mask=batch["attention_mask"].to(self.dev)
            segment_ids=batch.get("segment_ids",torch.zeros_like(input_ids)).to(self.dev); 
            labels=batch["labels"].to(self.dev)
            if is_training: self.opt.zero_grad()
            with torch.set_grad_enabled(is_training):
                logits = self.model(input_ids, attention_mask, segment_ids)
                loss = self.crit(logits.view(-1, self.vocab_size), labels.view(-1))
            if is_training: loss.backward(); self.opt.step()
            total_loss += loss.item()
            if (i_batch + 1) % self.log_freq == 0: progress_bar.set_postfix({"loss": f"{total_loss / (i_batch + 1):.4f}"})
        avg_loss = total_loss / len(dl) if len(dl) > 0 else 0.0
        print(f"{desc} - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, num_epochs):
        print(f"Treinando (MLM) por {num_epochs} épocas. Observando 'loss' de validação.")
        model_saved_this_run = False
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_loss_epoch = self._run_epoch(epoch, is_training=False) 
            if self.val_dl and val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
                print(f"Melhor Val Loss (MLM): {self.best_val_loss:.4f}. Salvando: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
            print("-" * 30)
        if not self.val_dl and num_epochs > 0:
            print(f"(MLM) Sem validação. Salvando modelo da última época ({num_epochs}): {self.save_path}")
            torch.save(self.model.state_dict(), self.save_path); model_saved_this_run = True
        best_val_display = "N/A"
        if isinstance(self.best_val_loss,(int,float)) and not math.isinf(self.best_val_loss):
            if math.isnan(self.best_val_loss): best_val_display="N/A (NaN)"
            else:best_val_display=f"{self.best_val_loss:.4f}"
        print(f"Treinamento (MLM) concluído após {num_epochs} épocas. Melhor Val Loss: {best_val_display}")
        if model_saved_this_run: print(f"Modelo salvo em: {self.save_path}")
        elif num_epochs == 0: print("Nenhum treinamento realizado (0 épocas).")

# --- Funções para o Pipeline ---
def setup_data_and_train_tokenizer(args):
    print("\n--- Fase: Preparação de Dados Aroeira e Tokenizador ---")
    aroeira_data_source_for_mlm_local = None; _all_aroeira_sentences_list_local = []
    text_col = "text"; temp_file_for_tokenizer = Path(args.temp_tokenizer_train_file)

    if args.sagemaker_input_data_dir and args.input_data_filename:
        local_data_path = os.path.join(args.sagemaker_input_data_dir, args.input_data_filename)
        print(f"Lendo dados Aroeira do SageMaker: {local_data_path}")
        if Path(local_data_path).exists():
            import shutil; shutil.copyfile(local_data_path, temp_file_for_tokenizer)
            aroeira_data_source_for_mlm_local = temp_file_for_tokenizer
        else: raise FileNotFoundError(f"Arquivo {local_data_path} (SageMaker) não encontrado.")
    elif args.aroeira_subset_size is not None:
        print(f"Coletando {args.aroeira_subset_size} exemplos do Aroeira (Hub)...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        iterator = iter(streamed_ds); _collected_examples = []
        try:
            for _ in range(args.aroeira_subset_size): _collected_examples.append(next(iterator))
        except StopIteration: print(f"Alerta: Stream Aroeira (Hub) esgotado. Coletados {len(_collected_examples)}.")
        for ex in tqdm(_collected_examples, desc="Extraindo sentenças (subset Hub)", file=sys.stdout):
            sent = ex.get(text_col)
            if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences_list_local.append(sent.strip())
        if not _all_aroeira_sentences_list_local: raise ValueError("Nenhuma sentença Aroeira (subset Hub) extraída.")
        aroeira_data_source_for_mlm_local = _all_aroeira_sentences_list_local
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for s_line in _all_aroeira_sentences_list_local: f.write(s_line + "\n")
    else: 
        print("Processando stream completo do Aroeira (Hub) para arquivo de tokenizador...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        count_written = 0
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for ex in tqdm(streamed_ds, desc="Escrevendo Aroeira (stream completo) para arquivo", file=sys.stdout):
                sent = ex.get(text_col)
                if isinstance(sent, str) and sent.strip(): f.write(sent + "\n"); count_written +=1
        if count_written == 0: raise ValueError("Nenhuma sentença Aroeira (stream Hub) escrita.")
        aroeira_data_source_for_mlm_local = temp_file_for_tokenizer
    
    tokenizer_train_file_str = str(temp_file_for_tokenizer)
    TOKENIZER_SAVE_DIRECTORY = Path(args.output_dir) / "wordpiece_tokenizer_assets"
    TOKENIZER_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    vocab_file_in_save_dir = TOKENIZER_SAVE_DIRECTORY / "vocab.txt" 

    if not vocab_file_in_save_dir.exists():
        if not Path(tokenizer_train_file_str).exists():
            raise FileNotFoundError(f"Arquivo de treino para tokenizador '{tokenizer_train_file_str}' não encontrado.")
        wp_tokenizer_trainer = BertWordPieceTokenizer(
            clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True)
        print(f"Treinando tokenizador BertWordPiece com [{tokenizer_train_file_str}]...")
        wp_tokenizer_trainer.train(
            files=[tokenizer_train_file_str], vocab_size=args.vocab_size, 
            min_frequency=args.min_frequency_tokenizer, 
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
            limit_alphabet=1000, wordpieces_prefix="##")
        wp_tokenizer_trainer.save_model(str(TOKENIZER_SAVE_DIRECTORY), prefix=None) 
        print(f"Tokenizador BertWordPiece treinado e salvo em: {vocab_file_in_save_dir}")
    else: print(f"Tokenizador (vocab.txt) já existe em '{TOKENIZER_SAVE_DIRECTORY}'. Carregando...")
    
    print(f"Tentando carregar tokenizador BertTokenizerFast do DIRETÓRIO: {TOKENIZER_SAVE_DIRECTORY}")
    try:
        loaded_tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_SAVE_DIRECTORY), 
            do_lower_case=True, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", 
            cls_token="[CLS]", mask_token="[MASK]", local_files_only=True)
    except Exception as e_load_tok:
        print(f"ERRO AO CARREGAR TOKENIZADOR de '{TOKENIZER_SAVE_DIRECTORY}': {e_load_tok}"); raise
    pad_id_val = loaded_tokenizer.pad_token_id
    print(f"Vocabulário (BertTokenizerFast): {loaded_tokenizer.vocab_size}, PAD ID: {pad_id_val}")
    return loaded_tokenizer, pad_id_val, aroeira_data_source_for_mlm_local

def run_mlm_pretrain(args, tokenizer_obj: BertTokenizer, pad_token_id_val, data_source_for_mlm):
    print("\n--- Fase: Pré-Treinamento MLM ---")
    mlm_input_data_for_dataset = None
    if isinstance(data_source_for_mlm, Path):
        print(f"Carregando sentenças para MLM do arquivo: {data_source_for_mlm}")
        hf_text_ds = datasets.load_dataset("text", data_files=str(data_source_for_mlm), split="train", trust_remote_code=args.trust_remote_code)
        mlm_input_data_for_dataset = hf_text_ds
    elif isinstance(data_source_for_mlm, list): mlm_input_data_for_dataset = data_source_for_mlm
    else: raise TypeError("data_source_for_mlm deve ser lista de sentenças ou Path.")
    if not mlm_input_data_for_dataset or len(mlm_input_data_for_dataset) == 0 :
        raise ValueError("Nenhuma sentença disponível para o pré-treinamento MLM.")

    train_data_pt, val_data_pt = None, None
    if isinstance(mlm_input_data_for_dataset, datasets.Dataset):
        if len(mlm_input_data_for_dataset) > 10: # Split se houver dados suficientes
            split = mlm_input_data_for_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
            train_data_pt, val_data_pt = split['train'], split['test']
        else: train_data_pt = mlm_input_data_for_dataset # Usa tudo para treino se pequeno
    elif isinstance(mlm_input_data_for_dataset, list):
        val_s_mlm_r = 0.1; num_v_mlm = int(len(mlm_input_data_for_dataset) * val_s_mlm_r)
        if num_v_mlm < 1 and len(mlm_input_data_for_dataset) > 1: num_v_mlm = 1
        train_data_pt = mlm_input_data_for_dataset[num_v_mlm:]; val_data_pt = mlm_input_data_for_dataset[:num_v_mlm]
        if not train_data_pt: train_data_pt = mlm_input_data_for_dataset; val_data_pt = []
    
    print(f"Dados de Treino (MLM) para BERTMLMDataset: {len(train_data_pt) if train_data_pt else 0} exs.")
    if val_data_pt: print(f"Dados de Validação (MLM) para BERTMLMDataset: {len(val_data_pt)} exs.")

    train_ds_mlm = BERTMLMDataset(train_data_pt, tokenizer_obj, args.max_len, pad_token_id_val)
    if len(train_ds_mlm) == 0 and len(train_data_pt) > 0 : raise ValueError("Dataset de treino MLM vazio.")
    train_dl_mlm = DataLoader(train_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    val_dl_mlm = None
    if val_data_pt and len(val_data_pt) > 0:
        val_ds_mlm = BERTMLMDataset(val_data_pt, tokenizer_obj, args.max_len, pad_token_id_val)
        if len(val_ds_mlm) > 0: val_dl_mlm = DataLoader(val_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)

    bert_base = BERTBaseModel(tokenizer_obj.vocab_size, args.model_hidden_size, args.model_num_layers, 
                             args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
                             pad_token_id_val, dropout_prob=args.model_dropout_prob)
    bertlm_model = BERTLM(bert_base, tokenizer_obj.vocab_size, args.model_hidden_size)
    opt_mlm = torch.optim.AdamW(bertlm_model.parameters(), lr=args.lr_pretrain, 
                                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.weight_decay)
    crit_mlm = nn.CrossEntropyLoss(ignore_index=pad_token_id_val)
    
    trainer_mlm_instance = SimplifiedTrainer(
        bertlm_model, train_dl_mlm, val_dl_mlm, opt_mlm, crit_mlm, args.device, 
        args.pretrained_bertlm_save_filename, tokenizer_obj.vocab_size, log_freq=args.logging_steps
    )
    print("Pré-treinamento MLM configurado. Iniciando...")
    trainer_mlm_instance.train(num_epochs=args.epochs_pretrain) # Passa o nome correto do argumento

# --- Função Principal e Parseador de Argumentos ---
def parse_args(custom_args_list=None):
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino BERT no Aroeira (MLM-only com WordPiece).")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aroeira_subset_size", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency_tokenizer", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--sagemaker_input_data_dir", type=str, default=None)
    parser.add_argument("--input_data_filename", type=str, default=None)
    parser.add_argument("--epochs_pretrain", type=int, default=1) # Revertido para epochs_pretrain
    parser.add_argument("--batch_size_pretrain", type=int, default=8)
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)
    parser.add_argument("--model_hidden_size", type=int, default=256)
    parser.add_argument("--model_num_layers", type=int, default=2)
    parser.add_argument("--model_num_attention_heads", type=int, default=4)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './bert_wp_mlm_outputs_v3'))
    parser.add_argument("--tokenizer_vocab_filename", type=str, default="aroeira_wp_tokenizer-vocab.txt") # Nome base do vocab.txt
    parser.add_argument("--pretrained_bertlm_save_filename", type=str, default="aroeira_bertlm_wp_pretrained.pth")
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_wp_tokenizer.txt")
    parser.add_argument("--do_dataprep_tokenizer", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--do_pretrain", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    known_args, _ = parser.parse_known_args(custom_args_list if custom_args_list else None)

    if known_args.device is None: known_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    known_args.model_intermediate_size = known_args.model_hidden_size * 4
    output_dir_path = Path(known_args.output_dir)
    if not str(known_args.output_dir).startswith("/opt/ml/"):
         output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Caminhos completos são construídos aqui
    known_args.tokenizer_vocab_filename = str(output_dir_path / "wordpiece_tokenizer_files" / Path(known_args.tokenizer_vocab_filename).name) # Salva em subdiretório
    known_args.pretrained_bertlm_save_filename = str(output_dir_path / Path(known_args.pretrained_bertlm_save_filename).name)
    known_args.temp_tokenizer_train_file = str(output_dir_path / Path(known_args.temp_tokenizer_train_file).name)
    return known_args

def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)
    print("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items(): print(f"{arg_name}: {value}")
    print("----------------------------------")

    current_tokenizer_obj, current_pad_id, data_source_for_mlm = None, None, None
    
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain:
        current_tokenizer_obj, current_pad_id, data_source_for_mlm = setup_data_and_train_tokenizer(ARGS)
    else:
        print("Nenhuma ação de preparação de dados ou pré-treinamento solicitada. Encerrando.")
        return

    if ARGS.do_pretrain:
        if not data_source_for_mlm or not current_tokenizer_obj:
            print("ERRO: Fonte de dados Aroeira ou tokenizador não preparados para pré-treinamento.")
        else:
            run_mlm_pretrain(ARGS, current_tokenizer_obj, current_pad_id, data_source_for_mlm)
    
    print("\n--- Pipeline MLM com WordPiece Finalizado ---")

if __name__ == "__main__":
    main()

# --- Exemplo de como rodar no Jupyter Notebook ---
# notebook_args = [
#     "--max_len", "64",
#     "--aroeira_subset_size", "1000", 
#     "--epochs_pretrain", "1",
#     "--batch_size_pretrain", "4", 
#     "--model_hidden_size", "128", 
#     "--model_num_layers", "2",
#     "--model_num_attention_heads", "2",
#     "--output_dir", "./notebook_outputs_wp_mlm_final", 
#     # As flags --do_dataprep_tokenizer e --do_pretrain são True por padrão no parse_args
# ]
# # Para testar o carregamento do S3 (simulado):
# # Path("./caminho_simulado_sagemaker_input/aroeira_data/").mkdir(parents=True, exist_ok=True)
# # dummy_data_content = [{"text": "Primeira frase do aroeira."}, {"text": "Segunda frase do aroeira."}]
# # import json
# # with open("./caminho_simulado_sagemaker_input/aroeira_data/meu_arquivo_aroeira.json", "w", encoding="utf-8") as f_dummy:
# #     for entry in dummy_data_content:
# #         f_dummy.write(json.dumps(entry) + "\n")
# # notebook_args.extend([
# #     "--sagemaker_input_data_dir", "./caminho_simulado_sagemaker_input/aroeira_data/",
# #     "--input_data_filename", "meu_arquivo_aroeira.json"
# # ])
# # main(notebook_mode_args_list=notebook_args) 
# print("Execução de teste do notebook concluída.")
