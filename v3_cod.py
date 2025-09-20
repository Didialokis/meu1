import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import logging
import json
import math

# Desativa logs de informação da biblioteca 'transformers' para um output mais limpo
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- CONFIGURAções ---
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased' 

# ATENÇÃO: Verifique se este nome de arquivo é EXATAMENTE igual ao gerado pelo tradutor.
GOLD_FILE = 'stereoset_validation_pt_nllb_completo.json' 

OUTPUT_FILE = 'predictions_bertimbau.json'
# ---------------------

def calculate_pll_score(text, model, tokenizer, device):
    """
    Calcula a Pseudo-Log-Likelihood (PLL) normalizada para uma dada sentença.
    """
    # A lógica interna permanece a mesma, pois é robusta.
    tokenized_input = tokenizer.encode(text, return_tensors='pt').to(device)
    
    if tokenized_input.shape[1] <= 2: # Lida com sentenças vazias
        return -math.inf

    total_log_prob = 0.0

    for i in range(1, tokenized_input.shape[1] - 1): # Itera apenas sobre os tokens reais
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
        
    return total_log_prob / (tokenized_input.shape[1] - 2)


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

    # --- INÍCIO DA MODIFICAÇÃO ROBUSTA ---
    # Este bloco agora captura qualquer erro durante o carregamento e exibe uma mensagem clara.
    try:
        print(f"📂 Tentando carregar o arquivo de dados: '{GOLD_FILE}'...")
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # Acessa os dados dentro da chave "data", conforme a estrutura correta
        gold_data = full_data['data'] 
        print("✅ Arquivo de dados carregado e validado com sucesso!")
        
    except FileNotFoundError:
        print(f"❌ ERRO FATAL: Arquivo de dados '{GOLD_FILE}' não foi encontrado.")
        print("👉 Verifique se o nome do arquivo na variável GOLD_FILE está EXATAMENTE igual ao nome do arquivo gerado pelo tradutor.")
        return # Interrompe a execução
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ ERRO FATAL: O arquivo '{GOLD_FILE}' foi encontrado, mas há um erro de formato ou estrutura.")
        print(f"👉 Detalhe do erro: {e}")
        print("👉 Certifique-se de que o arquivo foi gerado corretamente pelo script de tradução mais recente.")
        return # Interrompe a execução
    # --- FIM DA MODIFICAÇÃO ROBUSTA ---

    predictions = {"intrasentence": [], "intersentence": []}
    
    total_sentences = 0
    for task_type in gold_data:
        for example in gold_data[task_type]:
            total_sentences += len(example['sentences']['id'])
    
    print(f"📊 Processando {total_sentences} sentenças...")

    with tqdm(total=total_sentences, unit="sentença") as pbar:
        for task_type in gold_data: # Itera sobre 'intrasentence' e 'intersentence'
            for example in gold_data[task_type]:
                sentences_data = example['sentences']
                sentence_ids = sentences_data['id']
                sentence_texts = sentences_data['sentence']
                
                for i in range(len(sentence_ids)):
                    sentence_id = sentence_ids[i]
                    sentence_text = sentence_texts[i]
                    
                    score = calculate_pll_score(sentence_text, model, tokenizer, device)
                    
                    predictions[task_type].append({"id": sentence_id, "score": score})
                    pbar.update(1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Arquivo de predições foi salvo com sucesso em '{OUTPUT_FILE}'!")


if __name__ == "__main__":
    generate_predictions()
