import streamlit as st
import boto3
import json
import os

# --- Configuração ---
# Substitua pelo ID do modelo Claude que você está usando
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" 
AWS_REGION = "us-east-1"  # Ou a região que você estiver usando
CONVERSATION_FILE = "conversa_salva.json"

# --- 1. Inicialização do Boto3 e Streamlit ---

# Inicializa o cliente do Bedrock
# O Boto3 usará automaticamente as credenciais do ambiente 
# (ex: a role da sua instância SageMaker)
try:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime', 
        region_name=AWS_REGION
    )
except Exception as e:
    st.error(f"Erro ao inicializar o Boto3: {e}")
    st.stop()


# Título da página
st.title("Meu Chat com Claude (via AWS)")

# Inicializa o histórico da conversa no "session_state" do Streamlit
# Isso garante que o histórico persista entre as interações na interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Funções de Salvar e Carregar Contexto ---

def salvar_conversa():
    """Salva o histórico atual em um arquivo JSON."""
    try:
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)
        st.success(f"Conversa salva em {CONVERSATION_FILE}")
    except Exception as e:
        st.error(f"Erro ao salvar conversa: {e}")

def carregar_conversa():
    """Carrega o histórico de um arquivo JSON."""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                st.session_state.messages = json.load(f)
            st.success(f"Conversa carregada de {CONVERSATION_FILE}")
            # Re-exibe a interface para mostrar a conversa carregada
            st.rerun() 
        except Exception as e:
            st.error(f"Erro ao carregar conversa: {e}")
    else:
        st.warning("Nenhum arquivo de conversa encontrado.")

# --- 3. Botões de Salvar/Carregar na Barra Lateral ---

with st.sidebar:
    st.header("Gerenciar Conversa")
    if st.button("Salvar Conversa"):
        salvar_conversa()
    
    if st.button("Carregar Conversa Anterior"):
        carregar_conversa()
        
    if st.button("Limpar Conversa Atual"):
        st.session_state.messages = []
        st.rerun()

# --- 4. Exibição do Histórico de Chat ---

# Exibe todas as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Lógica de Chamada do Claude ---

def formatar_mensagens_para_claude(messages_history):
    """
    Formata o histórico do Streamlit para o formato da API do Claude 3.
    Claude 3 espera uma lista de {"role": ..., "content": ...}
    Opcional: Adiciona uma mensagem de sistema.
    """
    # A API do Claude 3 espera que a conversa comece com um 'user'
    # Se a primeira mensagem for 'assistant', podemos ter problemas.
    # Esta é uma lógica simples para garantir isso.
    
    mensagens_formatadas = []
    for msg in messages_history:
        # Garante que o conteúdo seja uma string (API do Claude 3 só aceita texto)
        if isinstance(msg["content"], str):
            mensagens_formatadas.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Se a conversa estiver vazia ou a primeira mensagem não for 'user',
    # o Boto3 pode falhar. Este código assume que o usuário sempre começa.
    
    return mensagens_formatadas

# Captura a entrada do usuário (o "prompt")
if prompt := st.chat_input("Como posso ajudar?"):
    
    # 1. Adiciona a mensagem do usuário ao histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepara a chamada para a API
    # Mostra um "spinner" enquanto o modelo pensa
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")

        try:
            # Formata o histórico para a API
            mensagens_api = formatar_mensagens_para_claude(st.session_state.messages)
            
            # Monta o corpo da requisição (payload)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": mensagens_api
                # "system": "Você é um assistente prestativo." # (Opcional)
            })

            # 3. Invoca o modelo
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=MODEL_ID,
                contentType='application/json',
                accept='application/json'
            )

            # 4. Processa a resposta
            response_body = json.loads(response.get('body').read())
            
            # A resposta do Claude 3 está em response_body['content'][0]['text']
            resposta_claude = response_body['content'][0]['text']
            
            message_placeholder.markdown(resposta_claude)
            
            # 5. Adiciona a resposta do assistente ao histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta_claude})

        except Exception as e:
            st.error(f"Erro ao chamar o modelo: {e}")
            # Remove a última mensagem do usuário se a chamada falhar
            st.session_state.messages.pop()
