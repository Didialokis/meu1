import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# Nome do seu arquivo JSON traduzido (o output do script anterior)
ARQUIVO_TRADUZIDO = 'stereoset_validation_pt_claude35_otimizado.json'
ARQUIVO_SAIDA = 'verificacao_amostra_25_porcento.xlsx'
PERCENTUAL_AMOSTRA = 0.25  # 25%

def gerar_planilha_verificacao():
    print("🚀 Iniciando preparação para verificação manual...")

    # 1. Carregar o Dataset Traduzido
    try:
        with open(ARQUIVO_TRADUZIDO, 'r', encoding='utf-8') as f:
            dados_pt = json.load(f)['data']
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo '{ARQUIVO_TRADUZIDO}' não encontrado.")
        return

    # 2. Carregar o Dataset Original (Inglês) para mapeamento
    print("📥 Baixando dataset original (Inglês) para comparação...")
    dataset_en = load_dataset("McGill-NLP/stereoset", split="validation")
    
    # Criar um mapa rápido: ID -> Dados em Inglês
    mapa_en = {}
    for item in dataset_en:
        # O dataset do HF carrega tudo junto, precisamos identificar se é intra ou inter pelo ID ou estrutura
        # Mas para o mapa, basta o ID como chave
        mapa_en[item['id']] = {
            'context': item['context'],
            'sentences': item['sentences']['sentence'], # Lista de 3 frases
            'sent_ids': item['sentences']['id']
        }

    # 3. Estruturar os dados traduzidos em uma lista plana para o Pandas
    lista_dados = []
    
    for tarefa in ['intersentence', 'intrasentence']:
        if tarefa not in dados_pt: continue
        
        for exemplo in dados_pt[tarefa]:
            ex_id = exemplo['id']
            bias = exemplo['bias_type']
            
            # Recupera o original em inglês
            original = mapa_en.get(ex_id)
            if not original: continue

            # Adiciona à lista. 
            # Dica: Vamos colocar o Contexto e as 3 frases na mesma linha para facilitar a leitura
            linha = {
                'id': ex_id,
                'tarefa': tarefa,
                'dominio_vies': bias,
                'CONTEXTO_EN': original['context'],
                'CONTEXTO_PT': exemplo['context'],
            }
            
            # Adiciona as frases alvo (Target Sentences)
            frases_pt = exemplo['sentences']
            for i in range(3):
                linha[f'Alvo_{i+1}_EN'] = original['sentences'][i]
                linha[f'Alvo_{i+1}_PT'] = frases_pt[i]['sentence']
                linha[f'Label_{i+1}'] = frases_pt[i]['gold_label'] # Ajuda a saber qual é estereótipo

            lista_dados.append(linha)

    # 4. Criar DataFrame e fazer a Amostragem Estratificada
    df = pd.DataFrame(lista_dados)
    
    print(f"📊 Total de exemplos traduzidos: {len(df)}")
    
    # A MÁGICA DO PANDAS:
    # Agrupa por Tarefa e Domínio de Viés e pega 25% aleatório de cada grupo
    # random_state=42 garante que o sorteio seja reproduzível
    df_amostra = df.groupby(['tarefa', 'dominio_vies'], group_keys=False).apply(
        lambda x: x.sample(frac=PERCENTUAL_AMOSTRA, random_state=42)
    )

    print(f"✅ Amostra selecionada: {len(df_amostra)} exemplos (25% estratificado).")
    print("   Distribuição da amostra:")
    print(df_amostra.groupby(['tarefa', 'dominio_vies']).size())

    # 5. Salvar em Excel com formatação básica
    print(f"💾 Salvando em '{ARQUIVO_SAIDA}'...")
    df_amostra.to_excel(ARQUIVO_SAIDA, index=False)
    
    print("\nConcluído! Abra o arquivo Excel para verificar as traduções lado a lado.")

if __name__ == "__main__":
    gerar_planilha_verificacao()
