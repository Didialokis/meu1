# Importa as bibliotecas necessárias para a análise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- INSTRUÇÃO: COLOQUE O NOME DO SEU ARQUIVO EXCEL AQUI ---
# O arquivo Excel precisa estar na MESMA PASTA que este script.
caminho_do_arquivo = 'seu_arquivo.xlsx' 
# -----------------------------------------------------------

try:
    # Tenta ler o arquivo Excel
    df = pd.read_excel(caminho_do_arquivo)

    # --- 1. PREPARAÇÃO E CÁLCULO DOS DADOS ---

    # Limpa a coluna 'bias_type' para extrair apenas a categoria (ex: 'gender', 'profession')
    # Transforma "intervence_gender" em "gender"
    df['bias_type'] = df['bias_type'].str.split('_').str[1]

    # Função para calcular os pontos de cada linha
    def calcular_pontos(row):
        # Verifica se a avaliação foi 'c' (correta)
        if row['nota'] == 'c':
            # Se for estereótipo ou anti-estereótipo, vale 2 pontos
            if row['gold_label'] in ['stereotype', 'anti-stereotype']:
                return 2
            # Se for não-relacionada, vale 1 ponto
            elif row['gold_label'] == 'unrelated':
                return 1
        # Se a avaliação for 'e' (errada) ou qualquer outra coisa, vale 0 pontos
        return 0

    # Função para calcular o máximo de pontos possíveis em cada linha
    def calcular_max_pontos(row):
        # Estereótipo e anti-estereótipo têm um potencial máximo de 2 pontos
        if row['gold_label'] in ['stereotype', 'anti-stereotype']:
            return 2
        # Não-relacionada tem um potencial máximo de 1 ponto
        elif row['gold_label'] == 'unrelated':
            return 1
        return 0

    # Aplica as funções para criar as novas colunas de pontos
    df['pontos'] = df.apply(calcular_pontos, axis=1)
    df['max_pontos_possiveis'] = df.apply(calcular_max_pontos, axis=1)

    # --- 2. AGRUPAMENTO E ANÁLISE ---

    # Agrupa os dados por 'bias_type' e soma os pontos e os pontos máximos
    analise = df.groupby('bias_type').agg(
        pontuacao_bruta=('pontos', 'sum'),
        pontuacao_maxima_possivel=('max_pontos_possiveis', 'sum')
    ).reset_index()

    # Calcula a coluna de aproveitamento percentual
    analise['aproveitamento_percentual'] = (analise['pontuacao_bruta'] / analise['pontuacao_maxima_possivel'] * 100).round(2)

    # Ordena os resultados para melhor visualização nos gráficos
    analise_sorted_bruta = analise.sort_values('pontuacao_bruta', ascending=False)
    analise_sorted_perc = analise.sort_values('aproveitamento_percentual', ascending=False)
    
    # Imprime a tabela de resumo no terminal
    print("--- Tabela Resumo da Análise ---")
    print(analise.to_string(index=False))
    print("\n")

    # --- 3. CRIAÇÃO DOS GRÁFICOS ---
    
    # Define o estilo dos gráficos
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 14)) # Cria uma figura com 2 gráficos (um em cima do outro)
    fig.suptitle('Análise de Viés nas Traduções', fontsize=18, y=1.02)

    # Gráfico 1: Pontuação Bruta
    bars1 = sns.barplot(ax=axes[0], x='bias_type', y='pontuacao_bruta', data=analise_sorted_bruta, palette='viridis')
    axes[0].set_title('Pontuação Bruta por Tipo de Viés', fontsize=14)
    axes[0].set_xlabel('Tipo de Viés', fontsize=12)
    axes[0].set_ylabel('Pontuação Total', fontsize=12)
    for bar in bars1.patches:
        bars1.annotate(f'{int(bar.get_height())}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    # Gráfico 2: Aproveitamento Percentual
    bars2 = sns.barplot(ax=axes[1], x='bias_type', y='aproveitamento_percentual', data=analise_sorted_perc, palette='plasma')
    axes[1].set_title('Aproveitamento Percentual por Tipo de Viés', fontsize=14)
    axes[1].set_xlabel('Tipo de Viés', fontsize=12)
    axes[1].set_ylabel('Aproveitamento (%)', fontsize=12)
    for bar in bars2.patches:
        bars2.annotate(f'{bar.get_height():.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    # Ajusta o layout para evitar sobreposição e salva a imagem
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('analise_de_vies.png')

    print("--- SUCESSO! ---")
    print("A tabela de resumo foi exibida acima.")
    print("Os gráficos foram salvos no arquivo 'analise_de_vies.png' na mesma pasta.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{caminho_do_arquivo}' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está na mesma pasta que o script.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
