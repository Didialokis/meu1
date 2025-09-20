import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import logging
import json
import math

# Desativa logs de informação da biblioteca 'transformers'
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased' 

# ATENÇÃO: Use o nome do arquivo gerado pelo ÚLTIMO script de tradução
GOLD_FILE = 'stereoset_validation_pt_nllb_formato_original.json' 

OUTPUT_FILE = 'predictions_bertimbau.json'
# ---------------------

def calculate_pll_score(text, model, tokenizer, device):
    """
    Calcula a Pseudo-Log-Likelihood (PLL) normalizada para uma dada sentença.
    """
    tokenized_input = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Ignora sentenças vazias ou com apenas tokens especiais
    num_tokens_to_score = tokenized_input.shape[1] - 2
    if num_tokens_to_score <= 0:
        return -math.inf

    total_log_prob = 0.0

    for i in range(1, tokenized_input.shape[1] - 1):
        masked_input = tokenized_input.clone()
        original_token_id = masked_input[0, i].item()
        masked_input[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits
        
        masked_token_logits = logits[0, i, :]
        log_probs = torch.nn.functional.log_softmax(masked_token_logits, dim=0)
        token_log_prob = log_probs[original_token_id].item()
        total_log_prob += token_log_prob
        
    return total_log_prob / num_tokens_to_score


def generate_predictions():
    """
    Função principal que carrega o modelo, os dados, calcula os scores
    e salva o arquivo de predições.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Usando dispositivo: {device.upper()}")

    print(f"💾 Carregando modelo '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("✅ Modelo carregado com sucesso!")

    try:
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            # Acessa os dados dentro da chave "data", conforme a estrutura correta
            gold_data = full_data['data']
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{GOLD_FILE}' não encontrado. Verifique o nome do arquivo.")
        return

    predictions = {"intrasentence": [], "intersentence": []}
    
    # --- INÍCIO DA CORREÇÃO 1 ---
    # Contagem total de sentenças para a barra de progresso
    total_sentences = 0
    for task_type in gold_data:
        for example in gold_data[task_type]:
            # Agora contamos o tamanho da lista 'sentences' diretamente
            total_sentences += len(example['sentences'])
    # --- FIM DA CORREÇÃO 1 ---
    
    print(f"📊 Processando {total_sentences} sentenças...")

    with tqdm(total=total_sentences, unit="sentença") as pbar:
        for task_type in gold_data:
            for example in gold_data[task_type]:
                
                # --- INÍCIO DA CORREÇÃO 2 ---
                # Iteramos sobre a lista de objetos de sentença, que é a estrutura correta
                for sentence_obj in example['sentences']:
                    sentence_id = sentence_obj['id']
                    sentence_text = sentence_obj['sentence']
                # --- FIM DA CORREÇÃO 2 ---
                    
                    score = calculate_pll_score(sentence_text, model, tokenizer, device)
                    
                    predictions[task_type].append({"id": sentence_id, "score": score})
                    pbar.update(1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Arquivo de predições foi salvo com sucesso em '{OUTPUT_FILE}'!")


if __name__ == "__main__":
    generate_predictions()
