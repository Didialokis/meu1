

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
import argparse
import os

print(f"PyTorch: {torch.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")

# --- Definições de Classes ---

class BERTMLMDataset(Dataset):
    """Dataset para Masked Language Modeling.
       Aceita uma lista de sentenças ou um objeto datasets.Dataset.
    """
    def __init__(self, hf_dataset_or_sentences_list, tokenizer_instance, max_len_config, pad_token_id_config):
        self.tokenizer = tokenizer_instance
        self.max_len = max_len_config
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.mask_id = self.tokenizer.token_to_id("<mask>")
        self.cls_id = self.tokenizer.token_to_id("<s>")
        self.sep_id = self.tokenizer.token_to_id("</s>")
        self.pad_id = pad_token_id_config

        if isinstance(hf_dataset_or_sentences_list, datasets.Dataset):
            # Se for um dataset do Hugging Face, extrai a coluna 'text'
            # (assumindo que load_dataset("text", ...) cria uma coluna 'text')
            print(f"BERTMLMDataset recebendo datasets.Dataset com {len(hf_dataset_or_sentences_list)} exemplos.")
            self.sentences = [ex['text'] for ex in hf_dataset_or_sentences_list if ex['text'] and ex['text'].strip()]
            print(f"Extraídas {len(self.sentences)} sentenças não vazias do datasets.Dataset.")
        elif isinstance(hf_dataset_or_sentences_list, list):
            self.sentences = hf_dataset_or_sentences_list
        else:
            raise TypeError("Entrada para BERTMLMDataset deve ser uma lista de sentenças ou um datasets.Dataset.")
        
        if not self.sentences:
            print("AVISO: BERTMLMDataset inicializado com zero sentenças.")

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
        sentence_text = self.sentences[idx]
        enc = self.tokenizer.encode(sentence_text)
        masked_ids, mlm_labels = self._mask_tokens(enc.ids)
        return {"input_ids": masked_ids, "attention_mask": torch.tensor(enc.attention_mask, dtype=torch.long), 
                "segment_ids": torch.zeros_like(masked_ids, dtype=torch.long), "labels": mlm_labels}

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

class SimplifiedTrainer: # Focado em MLM
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 model_save_path, vocab_size_for_loss, log_freq=20):
        self.model, self.train_dl, self.val_dl = model.to(device), train_dataloader, val_dataloader
        self.opt, self.crit, self.dev = optimizer, criterion, device
        self.save_path = model_save_path
        self.best_val_loss = float('inf')
        self.vocab_size = vocab_size_for_loss
        self.log_freq = log_freq

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
        for epoch in range(num_epochs):
            self._run_epoch(epoch, is_training=True)
            val_loss = self_run_epoch(epoch, is_training=False)
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"Nova melhor Val Loss (MLM): {self.best_val_loss:.4f}. Salvando modelo: {self.save_path}")
                torch.save(self.model.state_dict(), self.save_path)
            elif epoch == num_epochs - 1 and not self.val_dl:
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
    print("\n--- Fase: Preparação de Dados Aroeira e Tokenizador ---")
    # Retorna: tokenizer_obj, pad_id, data_source (lista de sentenças ou Path para arquivo de texto)
    
    text_col = "text" # Coluna de texto esperada
    temp_file_for_tokenizer = Path(args.temp_tokenizer_train_file)

    if args.sagemaker_input_data_dir and args.input_data_filename: # Prioriza dados do S3 via SageMaker
        local_data_path = os.path.join(args.sagemaker_input_data_dir, args.input_data_filename)
        print(f"Lendo dados Aroeira do caminho SageMaker: {local_data_path}")
        if Path(local_data_path).exists():
            # Assume-se que este arquivo será usado para treinar o tokenizador E o modelo
            # Copiar para o local do temp_tokenizer_train_file para consistência
            # Se for muito grande, esta cópia pode ser um problema.
            # Idealmente, o tokenizer leria direto de local_data_path se for um só.
            # Por simplicidade de fluxo, vamos copiar para o temp_file.
            # Em um cenário de produção com arquivos gigantes, evitaria esta cópia.
            import shutil
            shutil.copyfile(local_data_path, temp_file_for_tokenizer)
            print(f"Dados do S3 copiados para: {temp_file_for_tokenizer}")
            # Neste caso, aroeira_data_source_for_mlm será o temp_file_for_tokenizer
            # e o BERTMLMDataset lerá dele (ou de splits dele)
            aroeira_data_source_for_mlm = temp_file_for_tokenizer
            # Contar linhas para informação (opcional, pode ser lento para arquivos enormes)
            # line_count = 0
            # with open(temp_file_for_tokenizer, 'r', encoding='utf-8') as f_count:
            #     for _ in f_count: line_count += 1
            # print(f"Total de sentenças no arquivo de dados S3: {line_count}")
        else:
            raise FileNotFoundError(f"Arquivo de dados {local_data_path} do SageMaker não encontrado.")
    elif args.aroeira_subset_size is not None: # Subconjunto do Hub
        print(f"Coletando {args.aroeira_subset_size} exemplos do Aroeira (Hub)...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        stream_iterator = iter(streamed_ds)
        _collected_examples_list = []
        try:
            for _ in range(args.aroeira_subset_size): _collected_examples_list.append(next(stream_iterator))
        except StopIteration: print(f"Alerta: Stream Aroeira (Hub) esgotado. Coletados {len(_collected_examples_list)}.")
        
        _all_aroeira_sentences_list = []
        for ex in tqdm(_collected_examples_list, desc="Extraindo sentenças (subset Hub)", file=sys.stdout):
            sent = ex.get(text_col)
            if isinstance(sent, str) and sent.strip(): _all_aroeira_sentences_list.append(sent.strip())
        
        if not _all_aroeira_sentences_list: raise ValueError("Nenhuma sentença Aroeira (subset Hub) extraída.")
        print(f"Total de sentenças (subset Hub): {len(_all_aroeira_sentences_list)}")
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for s_line in _all_aroeira_sentences_list: f.write(s_line + "\n")
        aroeira_data_source_for_mlm = _all_aroeira_sentences_list # Passa a lista para MLM
    else: # Dataset completo do Hub, stream para arquivo de tokenizador
        print("Processando stream completo do Aroeira (Hub) para arquivo de tokenizador...")
        streamed_ds = datasets.load_dataset("Itau-Unibanco/aroeira",split="train",streaming=True,trust_remote_code=args.trust_remote_code)
        if temp_file_for_tokenizer.exists(): temp_file_for_tokenizer.unlink()
        count_written = 0
        with open(temp_file_for_tokenizer, "w", encoding="utf-8") as f:
            for ex in tqdm(streamed_ds, desc="Escrevendo Aroeira (stream completo) para arquivo", file=sys.stdout):
                sent = ex.get(text_col)
                if isinstance(sent, str) and sent.strip(): 
                    f.write(sent + "\n")
                    count_written +=1
        if count_written == 0: raise ValueError("Nenhuma sentença Aroeira (stream completo Hub) escrita no arquivo.")
        print(f"Total de sentenças (stream completo Hub) escritas em {temp_file_for_tokenizer}: {count_written}")
        aroeira_data_source_for_mlm = temp_file_for_tokenizer # Passa o caminho do arquivo para MLM
        
    # Treinamento do Tokenizador (usa sempre temp_file_for_tokenizer)
    tokenizer_train_files_list = [str(temp_file_for_tokenizer)]
    vocab_f = Path(args.tokenizer_vocab_filename); merges_f = Path(args.tokenizer_merges_filename)
    if not vocab_f.exists() or not merges_f.exists():
        if not Path(tokenizer_train_files_list[0]).exists():
            raise FileNotFoundError(f"Arquivo de treino para tokenizador '{tokenizer_train_files_list[0]}' não encontrado.")
        base_vocab_name = Path(args.tokenizer_vocab_filename).name
        prefix = base_vocab_name.replace("-vocab.json", "")
        tok_model_bpe = ByteLevelBPETokenizer(lowercase=True)
        print(f"Treinando tokenizador com {tokenizer_train_files_list}...")
        tok_model_bpe.train(files=tokenizer_train_files_list, vocab_size=args.vocab_size, 
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
    
    # Retorna o tokenizador, pad_id, e a fonte de dados para MLM (lista ou caminho)
    return loaded_tokenizer, pad_id_val, aroeira_data_source_for_mlm

def run_mlm_pretrain(args, tokenizer_obj, pad_token_id_val, data_source_for_mlm):
    print("\n--- Fase: Pré-Treinamento MLM ---")
    
    # Carregar dados para MLM
    # Se data_source_for_mlm for um Path, carregue como Hugging Face Dataset de texto
    # Se for uma lista, use diretamente.
    mlm_sentences_list = []
    if isinstance(data_source_for_mlm, Path):
        print(f"Carregando sentenças para MLM do arquivo: {data_source_for_mlm}")
        # Este load_dataset carrega o arquivo inteiro, depois podemos pegar a coluna 'text'
        # A biblioteca datasets é eficiente para arquivos grandes no disco (usa mmap).
        hf_text_dataset = datasets.load_dataset("text", data_files=str(data_source_for_mlm), split="train", trust_remote_code=args.trust_remote_code)
        # Se o dataset for muito grande para caber na RAM como lista de strings,
        # BERTMLMDataset precisaria ser adaptado para iterar sobre hf_text_dataset.
        # Por ora, vamos extrair para uma lista, assumindo que os splits de treino/val caberão.
        mlm_sentences_list = [ex['text'] for ex in hf_text_dataset if ex['text'] and ex['text'].strip()]
        print(f"Total de sentenças para MLM (do arquivo): {len(mlm_sentences_list)}")
    elif isinstance(data_source_for_mlm, list):
        mlm_sentences_list = data_source_for_mlm # Já é uma lista (caso do subset)
        print(f"Total de sentenças para MLM (da lista de subset): {len(mlm_sentences_list)}")
    else:
        raise TypeError("data_source_for_mlm deve ser uma lista de sentenças ou um Path para arquivo de texto.")

    if not mlm_sentences_list:
        raise ValueError("Nenhuma sentença disponível para o pré-treinamento MLM.")

    val_s_mlm_r = 0.1; num_v_mlm = int(len(mlm_sentences_list) * val_s_mlm_r)
    if num_v_mlm < 1 and len(mlm_sentences_list) > 1: num_v_mlm = 1
    tr_s_mlm = mlm_sentences_list[num_v_mlm:]; v_s_mlm = mlm_sentences_list[:num_v_mlm]
    if not tr_s_mlm: tr_s_mlm = mlm_sentences_list; v_s_mlm = []

    print(f"Sentenças de Treino (MLM): {len(tr_s_mlm)}, Sentenças de Validação (MLM): {len(v_s_mlm)}")
    train_ds_mlm = BERTMLMDataset(tr_s_mlm, tokenizer_obj, args.max_len, pad_token_id_val)
    if len(train_ds_mlm) == 0 and len(tr_s_mlm) > 0: raise ValueError("Dataset de treino MLM vazio após processamento BERTMLMDataset.")
    train_dl_mlm = DataLoader(train_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=True, num_workers=0)
    
    val_dl_mlm = None
    if v_s_mlm and len(v_s_mlm) > 0:
        val_ds_mlm = BERTMLMDataset(v_s_mlm, tokenizer_obj, args.max_len, pad_token_id_val)
        if len(val_ds_mlm) > 0: val_dl_mlm = DataLoader(val_ds_mlm, batch_size=args.batch_size_pretrain, shuffle=False, num_workers=0)
        elif len(v_s_mlm) > 0 : print("AVISO: Dataset de validação MLM resultou em 0 exemplos após BERTMLMDataset.")


    bert_base = BERTBaseModel(tokenizer_obj.get_vocab_size(), args.model_hidden_size, args.model_num_layers, 
                             args.model_num_attention_heads, args.model_intermediate_size, args.max_len, 
                             pad_token_id_val, dropout_prob=args.model_dropout_prob)
    bertlm_model = BERTLM(bert_base, tokenizer_obj.get_vocab_size(), args.model_hidden_size)
    opt_mlm = torch.optim.AdamW(bertlm_model.parameters(), lr=args.lr_pretrain, 
                                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.weight_decay)
    crit_mlm = nn.CrossEntropyLoss(ignore_index=pad_token_id_val)
    
    trainer_mlm_instance = SimplifiedTrainer(
        bertlm_model, train_dl_mlm, val_dl_mlm, opt_mlm, crit_mlm, args.device, 
        args.pretrained_bertlm_save_filename, tokenizer_obj.get_vocab_size(), log_freq=args.logging_steps
    )
    print("Pré-treinamento MLM configurado. Iniciando...")
    trainer_mlm_instance.train(num_epochs=args.epochs_pretrain)

def parse_args(custom_args_list=None):
    # ... (parse_args como na última versão, garantindo que --aroeira_subset_size default=None)
    parser = argparse.ArgumentParser(description="Pipeline de Pré-treino BERT no Aroeira (MLM-only).")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aroeira_subset_size", type=int, default=None, help="Exemplos Aroeira. None para dataset completo.")
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
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', './bert_mlm_outputs_final'))
    parser.add_argument("--tokenizer_vocab_filename", type=str, default="aroeira_mlm_tokenizer-vocab.json")
    parser.add_argument("--tokenizer_merges_filename", type=str, default="aroeira_mlm_tokenizer-merges.txt")
    parser.add_argument("--pretrained_bertlm_save_filename", type=str, default="aroeira_bertlm_pretrained.pth")
    parser.add_argument("--temp_tokenizer_train_file", type=str, default="temp_aroeira_for_tokenizer.txt")
    parser.add_argument("--do_dataprep_tokenizer", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--do_pretrain", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Adicione quaisquer outros argumentos que o SageMaker possa passar para evitar erros
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Ignorado neste script customizado")
    parser.add_argument('--warmup_steps', type=int, default=0, help="Ignorado neste script customizado")
    # ... (outros args do HF Trainer se necessário para compatibilidade com o dict do SageMaker)

    if custom_args_list: args = parser.parse_args(custom_args_list)
    else: args = parser.parse_args()
    
    if args.device is None: args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_intermediate_size = args.model_hidden_size * 4
    output_dir_path = Path(args.output_dir)
    if not str(args.output_dir).startswith("/opt/ml/"):
         output_dir_path.mkdir(parents=True, exist_ok=True)
    args.tokenizer_vocab_filename = str(output_dir_path / Path(args.tokenizer_vocab_filename).name)
    args.tokenizer_merges_filename = str(output_dir_path / Path(args.tokenizer_merges_filename).name)
    args.pretrained_bertlm_save_filename = str(output_dir_path / Path(args.pretrained_bertlm_save_filename).name)
    args.temp_tokenizer_train_file = str(output_dir_path / Path(args.temp_tokenizer_train_file).name)
    return args

def main(notebook_mode_args_list=None):
    ARGS = parse_args(notebook_mode_args_list)
    print("--- Configurações Utilizadas ---")
    for arg_name, value in vars(ARGS).items(): print(f"{arg_name}: {value}")
    print("----------------------------------")

    current_tokenizer_obj, current_pad_id, mlm_data_source = None, None, None
    
    if ARGS.do_dataprep_tokenizer or ARGS.do_pretrain:
        current_tokenizer_obj, current_pad_id, mlm_data_source = setup_data_and_train_tokenizer(ARGS)
    else:
        print("Nenhuma ação de preparação de dados ou pré-treinamento solicitada. Encerrando.")
        return

    if ARGS.do_pretrain:
        if not mlm_data_source or not current_tokenizer_obj:
            print("ERRO: Fonte de dados Aroeira ou tokenizador não preparados para pré-treinamento.")
        else:
            run_mlm_pretrain(ARGS, current_tokenizer_obj, current_pad_id, mlm_data_source)
    
    print("\n--- Pipeline MLM Finalizado ---")

if __name__ == "__main__":
    main()
