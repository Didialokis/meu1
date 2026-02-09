import pandas as pd
import json
import numpy as np

# --- CONFIGURAÇÕES ---
ARQUIVO_EXCEL = 'verificacao_amostra_25_porcento.xlsx'
ARQUIVO_JSON_SAIDA = 'stereoset_reconstruido.json'

def converter_excel_para_json():
    print(f"📂 Lendo: {ARQUIVO_EXCEL}...")
    
    # Lê o Excel e substitui NaN por string vazia para evitar erros
    df = pd.read_excel(ARQUIVO_EXCEL).replace({np.nan: None})
    
    # Estrutura raiz do StereoSet
    dados_json = {
        "version": "1.1",
        "data": {
            "intersentence": [],
            "intrasentence": []
        }
    }

    print("⚙️  Reconstruindo estrutura hierárquica...")

    for _, row in df.iterrows():
        tarefa = row['tarefa']
        
        # Estrutura base do exemplo
        exemplo = {
            "id": row['id'],
            "bias_type": row['dominio_vies'],
            "target": "Generic", # O target original muitas vezes não está no excel simplificado, mas o ID mantém a referência
            "context": str(row['CONTEXTO_PT']).strip(),
            "sentences": []
        }

        # Reconstrói as 3 frases (Alvos)
        for i in range(1, 4):
            texto_frase = row[f'Alvo_{i}_PT']
            label = row[f'Label_{i}'] # stereotype, anti-stereotype, unrelated
            
            # Pula se estiver vazio (caso raro de erro no Excel)
            if not texto_frase:
                continue

            sentenca_obj = {
                "id": f"{row['id']}_{i}", # ID único para a sentença
                "sentence": str(texto_frase).strip(),
                "gold_label": label,
                # Recriamos a lista 'labels' vazia ou com dados mínimos para compatibilidade
                # O script de avaliação oficial usa 'gold_label', mas a estrutura exige a chave 'labels'
                "labels": [] 
            }
            exemplo['sentences'].append(sentenca_obj)

        # Adiciona na lista correta (intra ou inter)
        if tarefa in dados_json['data']:
            dados_json['data'][tarefa].append(exemplo)

    # Salva o arquivo
    with open(ARQUIVO_JSON_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(dados_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Conversão concluída! Arquivo salvo em: {ARQUIVO_JSON_SAIDA}")

if __name__ == "__main__":
    converter_excel_para_json()
