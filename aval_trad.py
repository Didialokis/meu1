import json
import pandas as pd
from datasets import load_dataset, concatenate_datasets

# --- CONFIGURAÇÕES ---
ARQUIVO_TRADUZIDO = 'stereoset_validation_pt_claude35_otimizado.json' # Seu arquivo input
ARQUIVO_SAIDA = 'verificacao_amostra_25_porcento.xlsx'                # Arquivo output
PERCENTUAL = 0.25                                                       # 25%

def gerar_dataset_verificacao():
    print("⏳ Carregando dataset original do HuggingFace (Inglês)...")
    
    # 1. CORREÇÃO DO ERRO: Carregar as duas configs separadamente e unir
    # O StereoSet no HF exige que definamos "intrasentence" ou "intersentence"
    ds_intra = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    ds_inter = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
    
    # Unimos tudo em uma lista única para facilitar a busca
    ds_full = concatenate_datasets([ds_intra, ds_inter])
    
    # Criamos um dicionário para busca rápida por ID (Hash Map)
    # Chave: ID, Valor: O objeto completo em inglês
    mapa_ingles = {item['id']: item for item in ds_full}

    print(f"✅ Dataset original carregado: {len(mapa_ingles)} exemplos.")

    # 2. Carregar o arquivo traduzido (JSON)
    print(f"⏳ Carregando arquivo traduzido: {ARQUIVO_TRADUZIDO}...")
    try:
        with open(ARQUIVO_TRADUZIDO, 'r', encoding='utf-8') as f:
            dados_pt = json.load(f)
            # Verifica se o JSON tem a chave 'data' ou se é uma lista direta (ajuste conforme seu JSON)
            if 'data' in dados_pt:
                dados_pt = dados_pt['data']
    except FileNotFoundError:
        print("❌ Erro: Arquivo JSON não encontrado.")
        return

    # 3. Cruzamento de Dados (Inglês vs Português)
    lista_comparativa = []

    # Iterar pelas tarefas (intra e inter)
    for tarefa in ['intrasentence', 'intersentence']:
        if tarefa not in dados_pt: continue

        for item_pt in dados_pt[tarefa]:
            id_exemplo = item_pt['id']
            
            # Busca o original em inglês
            item_en = mapa_ingles.get(id_exemplo)
            
            if not item_en:
                continue # Pula se não achar o ID correspondente

            # Estrutura a linha para o Excel
            # Pegamos o contexto e as 3 frases (stereotype, anti-stereotype, unrelated)
            linha = {
                'ID': id_exemplo,
                'Tarefa': tarefa,
                'Viés': item_pt['bias_type'],
                'Contexto_EN': item_en['context'],
                'Contexto_PT': item_pt['context']
            }

            # Adiciona as 3 frases comparativas
            # Nota: A ordem das sentences no JSON original e traduzido deve ser respeitada
            sentences_pt = item_pt['sentences']
            sentences_en_list = item_en['sentences']['sentence'] # HF retorna lista de strings
            
            # Segurança para caso o tamanho das listas difira (não deve ocorrer)
            qtd = min(len(sentences_pt), len(sentences_en_list))
            
            for i in range(qtd):
                label = sentences_pt[i]['gold_label'] # Ex: stereotype
                linha[f'Frase_{i+1}_Label'] = label
                linha[f'Frase_{i+1}_EN'] = sentences_en_list[i]
                linha[f'Frase_{i+1}_PT'] = sentences_pt[i]['sentence']

            lista_comparativa.append(linha)

    # 4. Criação do DataFrame e Amostragem Estratificada
    df = pd.DataFrame(lista_comparativa)
    
    print(f"📊 Total processado: {len(df)} linhas.")
    print("🎲 Realizando amostragem estratificada de 25%...")

    # AQUI ESTÁ A LÓGICA DE ESTRATIFICAÇÃO:
    # Agrupamos por Tarefa e Viés para garantir representatividade de todos os grupos
    df_amostra = df.groupby(['Tarefa', 'Viés'], group_keys=False).apply(
        lambda x: x.sample(frac=PERCENTUAL, random_state=42) # random_state fixa a aleatoriedade
    )

    # 5. Exportar para Excel
    print(f"💾 Salvando {len(df_amostra)} exemplos em '{ARQUIVO_SAIDA}'...")
    
    # Ajustar largura das colunas (opcional, visual) ou apenas salvar
    df_amostra.to_excel(ARQUIVO_SAIDA, index=False)
    
    print("\n✅ Concluído! Resumo da Amostra:")
    print(df_amostra.groupby(['Tarefa', 'Viés']).size())

if __name__ == "__main__":
    gerar_dataset_verificacao()
