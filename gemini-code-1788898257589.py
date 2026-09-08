import boto3
import time
import pandas as pd
import json
import re
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

# ==============================================================================
# CONFIGURAÇÕES INICIAIS
# ==============================================================================
# O SageMaker assume automaticamente a IAM Role da instância. 
# Certifique-se de que a role tenha a permissão 'bedrock:InvokeModel'.
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Modelos (usaremos o 8B para os testes e o 70B para geração de dados e avaliação/juiz)
MODEL_ID = "meta.llama3-8b-instruct-v1:0"
JUDGE_MODEL_ID = "meta.llama3-70b-instruct-v1:0" 
MODEL_KWARGS = {"temperature": 0.0}

# ==============================================================================
# ETAPA 1: GERAÇÃO DE DATASET SINTÉTICO
# ==============================================================================
def gerar_cenarios_sinteticos(quantidade=5):
    """Gera cenários de negócios variados para o teste logístico-financeiro"""
    print(f"Gerando {quantidade} cenários sintéticos...")
    llm_gerador = ChatBedrock(client=bedrock_client, model_id=JUDGE_MODEL_ID, model_kwargs={"temperature": 0.7})
    
    prompt = f"""
    Gere {quantidade} cenários corporativos curtos e diretos sobre importação de hardware para o Brasil.
    Varie os seguintes parâmetros em cada cenário:
    - Tipo de hardware (ex: GPUs, Servidores, Roteadores, Laptops)
    - Custos e prazos de frete (aéreo vs marítimo)
    - Alíquotas de imposto de importação (ex: 40%, 60%, 80%)
    - Urgência da empresa (ex: 15 dias, 30 dias, 60 dias)
    
    Retorne APENAS um array JSON estrito no seguinte formato, sem formatação markdown:
    [
      "Cenário 1: [descrição]",
      "Cenário 2: [descrição]"
    ]
    """
    try:
        resposta = llm_gerador.invoke([HumanMessage(content=prompt)])
        texto_limpo = re.sub(r'```json\n|```', '', resposta.content.strip())
        cenarios = json.loads(texto_limpo)
        print("Cenários gerados com sucesso!\n")
        return cenarios
    except Exception as e:
        print(f"Erro na geração: {e}")
        # Fallback de segurança caso o JSON falhe
        return [
            "Importação de 500 GPUs da China para São Paulo. Aéreo: US$15.000 (7 dias). Marítimo: US$3.000 (45 dias). Imposto: 60%. Urgência: 15 dias."
        ]

# ==============================================================================
# ETAPA 2: DEFINIÇÃO DAS ARQUITETURAS (MONOLÍTICA E MULTIAGENTE)
# ==============================================================================
def executar_monolitico(cenario):
    """Abordagem de instrução única de ponta a ponta"""
    llm = ChatBedrock(client=bedrock_client, model_id=MODEL_ID, model_kwargs=MODEL_KWARGS)
    
    prompt_completo = f"Você é um consultor corporativo. Analise a viabilidade logística e financeira do seguinte cenário de importação: {cenario}. Forneça um parecer final."
    
    start_time = time.time()
    resposta = llm.invoke([HumanMessage(content=prompt_completo)])
    latencia = time.time() - start_time
    
    metrics = resposta.response_metadata.get('amazon-bedrock-invocationMetrics', {})
    tokens_totais = metrics.get('inputTokenCount', 0) + metrics.get('outputTokenCount', 0)
    
    return resposta.content, latencia, tokens_totais

def executar_multiagente(cenario):
    """Fluxo sequencial mimetizando cadeia produtiva: Logística -> Finanças"""
    llm = ChatBedrock(client=bedrock_client, model_id=MODEL_ID, model_kwargs=MODEL_KWARGS)
    
    start_time = time.time()
    
    # Agente 1: Logística
    sys_logistica = SystemMessage(content="Você é um especialista em logística. Analise apenas prazos e rotas para o cenário. Não faça cálculos financeiros.")
    resp_log = llm.invoke([sys_logistica, HumanMessage(content=cenario)])
    met_log = resp_log.response_metadata.get('amazon-bedrock-invocationMetrics', {})
    tok_log = met_log.get('inputTokenCount', 0) + met_log.get('outputTokenCount', 0)
    
    # Agente 2: Finanças
    sys_financas = SystemMessage(content="Você é um especialista em finanças. Com base no cenário original e no parecer logístico, calcule os custos, aplique os impostos e dê o veredito financeiro final.")
    prompt_fin = f"Cenário Original: {cenario}\n\nParecer Logístico: {resp_log.content}"
    resp_fin = llm.invoke([sys_financas, HumanMessage(content=prompt_fin)])
    
    met_fin = resp_fin.response_metadata.get('amazon-bedrock-invocationMetrics', {})
    tok_fin = met_fin.get('inputTokenCount', 0) + met_fin.get('outputTokenCount', 0)
    
    latencia = time.time() - start_time
    tokens_totais = tok_log + tok_fin
    
    return resp_fin.content, latencia, tokens_totais

# ==============================================================================
# ETAPA 3: AVALIAÇÃO AUTOMATIZADA (LLM-AS-A-JUDGE)
# ==============================================================================
def avaliar_qualidade(cenario, resposta_avaliada):
    """Utiliza um modelo superior para mitigar o viés de avaliação manual"""
    judge_llm = ChatBedrock(client=bedrock_client, model_id=JUDGE_MODEL_ID, model_kwargs=MODEL_KWARGS)
    
    prompt = f"""
    Atue como um auditor técnico rigoroso. Avalie a resposta gerada para o cenário abaixo.
    1. 'score_precisao': Nota de 0 a 10 avaliando o raciocínio lógico da resposta.
    2. 'alucinacao': 'sim' se o modelo inventou dados não presentes no cenário, 'nao' caso contrário.
    
    Cenário: {cenario}
    Resposta Gerada: {resposta_avaliada}
    
    Retorne APENAS um JSON válido: {{"score_precisao": int, "alucinacao": "sim"/"nao"}}
    """
    try:
        resultado = judge_llm.invoke([HumanMessage(content=prompt)])
        texto_limpo = re.sub(r'```json\n|```', '', resultado.content.strip())
        return json.loads(texto_limpo)
    except Exception:
        return {"score_precisao": 0, "alucinacao": "erro"}

# ==============================================================================
# ETAPA 4: ORQUESTRAÇÃO E CONSOLIDAÇÃO DOS DADOS
# ==============================================================================
def executar_pipeline():
    # 1. Obter cenários (ajuste a quantidade conforme a necessidade do seu estudo)
    cenarios = gerar_cenarios_sinteticos(quantidade=10)
    resultados = []
    
    for i, cenario in enumerate(cenarios):
        print(f"Executando simulação {i+1}/{len(cenarios)}...")
        
        # Teste Monolítico
        resp_mono, lat_mono, tok_mono = executar_monolitico(cenario)
        eval_mono = avaliar_qualidade(cenario, resp_mono)
        
        resultados.append({
            "cenario_id": i+1,
            "arquitetura": "Monolítica",
            "latencia_segundos": round(lat_mono, 2),
            "tokens_totais": tok_mono,
            "score_precisao": eval_mono.get("score_precisao", 0),
            "alucinacao": eval_mono.get("alucinacao", "erro")
        })
        
        # Pausa leve para evitar Throttling (TooManyRequestsException)
        time.sleep(1.5) 
        
        # Teste Multiagente
        resp_multi, lat_multi, tok_multi = executar_multiagente(cenario)
        eval_multi = avaliar_qualidade(cenario, resp_multi)
        
        resultados.append({
            "cenario_id": i+1,
            "arquitetura": "Multiagente",
            "latencia_segundos": round(lat_multi, 2),
            "tokens_totais": tok_multi,
            "score_precisao": eval_multi.get("score_precisao", 0),
            "alucinacao": eval_multi.get("alucinacao", "erro")
        })
        
        time.sleep(1.5)
        
    # Consolidando em DataFrame
    df = pd.DataFrame(resultados)
    
    print("\n================ RESUMO ESTATÍSTICO ================\n")
    # Agrupando métricas de custo e performance
    resumo_quantitativo = df.groupby('arquitetura')[['latencia_segundos', 'tokens_totais', 'score_precisao']].mean().round(2)
    print(resumo_quantitativo)
    
    print("\n================ TAXA DE ALUCINAÇÃO ================\n")
    # Calculando a frequência de alucinações por arquitetura
    taxa_alucinacao = df.groupby(['arquitetura', 'alucinacao']).size().unstack(fill_value=0)
    print(taxa_alucinacao)
    
    # Exportar os dados para incluir na sua tabela de resultados do TCC
    df.to_csv("resultados_experimento_tcc.csv", index=False)
    print("\nDataset completo salvo como 'resultados_experimento_tcc.csv'")
    
    return df

# Executar todo o fluxo
if __name__ == "__main__":
    df_final = executar_pipeline()