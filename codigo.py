import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import os

# --- 1. Definições Principais ---
DATASET_NAME = "Itau-Unibanco/aroeira"
MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased" # Modelo BERT em português
OUTPUT_DIR = "./bert_aroeira_subset_trained_v2" # Saída do modelo treinado
NUM_EXAMPLES_SUBSET = 30000  # Tamanho do subconjunto para treinamento inicial
MAX_SEQ_LENGTH = 128        # Comprimento máximo da sequência para tokenização
MLM_PROBABILITY = 0.15      # Probabilidade de mascaramento para Masked Language Modeling

# --- 2. Carregar e Preparar o Dataset ---
print(f"Carregando dataset '{DATASET_NAME}'...")
try:
    raw_datasets = load_dataset(DATASET_NAME)
    split_name = 'train'
    if split_name not in raw_datasets:
        split_name = list(raw_datasets.keys())[0] # Usa o primeiro split se 'train' não existir
        print(f"Aviso: Split 'train' não encontrado. Usando split '{split_name}'.")

    num_available = len(raw_datasets[split_name])
    actual_subset_size = min(NUM_EXAMPLES_SUBSET, num_available)
    
    print(f"Selecionando {actual_subset_size} exemplos do split '{split_name}'...")
    dataset_subset = raw_datasets[split_name].select(range(actual_subset_size))

    text_column_name = "sentence" # Coluna contendo o texto principal
    if text_column_name not in dataset_subset.features:
        # Tenta encontrar outra coluna de texto comum se 'sentence' não existir
        potential_cols = [col for col in dataset_subset.features if isinstance(dataset_subset.features[col], datasets.Value) and dataset_subset.features[col].dtype == 'string']
        if 'text' in potential_cols:
            text_column_name = 'text'
        elif potential_cols:
            text_column_name = potential_cols[0]
            print(f"Aviso: Coluna 'sentence' não encontrada. Usando a primeira coluna de texto encontrada: '{text_column_name}'.")
        else:
            raise ValueError(f"Coluna de texto principal ('sentence' ou 'text') não encontrada. Features: {dataset_subset.features}")
        
except Exception as e:
    print(f"Erro ao carregar/processar dataset: {e}")
    exit()

print(f"Usando coluna '{text_column_name}' para texto. Exemplo: {dataset_subset[0][text_column_name][:100]}...")

# --- 3. Carregar Tokenizador ---
print(f"Carregando tokenizador de '{MODEL_CHECKPOINT}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# --- 4. Tokenizar o Dataset ---
def tokenize_function(examples):
    return tokenizer(
        examples[text_column_name],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

print("Tokenizando o dataset...")
tokenized_datasets = dataset_subset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset_subset.column_names 
)

# --- 5. Preparar o Data Collator para MLM ---
# Cria lotes e aplica mascaramento dinamicamente para Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY
)

# --- 6. Carregar o Modelo ---
print(f"Carregando modelo '{MODEL_CHECKPOINT}' para Masked LM...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

# --- 7. Configurar Argumentos de Treinamento ---
# Hiperparâmetros e configurações para o loop de treinamento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1, # Poucas épocas para análise inicial
    per_device_train_batch_size=8, # Ajuste conforme a memória da GPU/CPU
    save_steps=1000, # Salvar checkpoints periodicamente
    save_total_limit=2, # Manter apenas os últimos checkpoints
    logging_steps=200, # Registrar métricas (loss) periodicamente
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Usa precisão mista se GPU compatível estiver disponível
    report_to="tensorboard", # Opcional: para visualização no TensorBoard
)

# --- 8. Inicializar o Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("Trainer inicializado.")

# --- 9. Treinar o Modelo ---
print("Iniciando o treinamento...")
try:
    train_result = trainer.train()
    print("Treinamento concluído.")

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # --- 10. Salvar o Modelo Final e o Tokenizador ---
    print(f"Salvando modelo e tokenizador em '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR) 
    # O tokenizador geralmente é salvo com save_model se foi passado ao Trainer
    # tokenizer.save_pretrained(OUTPUT_DIR) # Linha opcional para garantir
    print("Modelo e tokenizador salvos.")

except Exception as e:
    print(f"Erro durante o treinamento: {e}")

print("--- Processo Finalizado ---")
