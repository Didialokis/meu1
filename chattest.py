# -*- coding: utf-8 -*-
# %% [markdown]
# ### 1. Configurações e Inicialização
# 
# Importamos as bibliotecas necessárias, definimos o modelo e iniciamos o cliente Boto3.
# Também inicializamos nossa lista de histórico de chat.

# %%
import boto3
import botocore.exceptions
import json
import logging
import sys

# --- 1. CONFIGURAÇÕES ---

# ATUALIZADO: ID do Modelo para Claude 3 Sonnet
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
REGION_NAME = 'us-east-1' # Mantenha a região do seu script original

# REUTILIZADO: Cliente Boto3 do seu script
try:
    client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
except Exception as e:
    print(f"Erro ao inicializar o cliente Boto3: {e}")
    print("Verifique se suas credenciais AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.) estão configuradas.")
    sys.exit(1)


# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NOVO: Histórico da conversa (para manter o contexto)
# A lista `messages` armazenará todos os turnos da conversa.
messages = []

# NOVO: System Prompt (Opcional, mas recomendado para o Claude)
# Define o comportamento/personalidade do assistente.
SYSTEM_PROMPT = "Você é um assistente de IA prestativo e amigável. Responda em português do Brasil."


# %% [markdown]
# ### 2. Função de Chamada do Bedrock (Adaptada para Chat)
# 
# Esta função envia a nova mensagem do usuário *junto com* o histórico da conversa para o Claude e trata a resposta.
# A lógica de retentativas do seu script original foi mantida.

# %%
def chamar_claude_chat(user_message, message_history):
    """
    Envia uma mensagem do usuário e o histórico para o Claude 3 Sonnet e retorna a resposta.
    Mantém a lógica de retentativas do script original.
    """
    
    # 1. Adiciona a nova mensagem do usuário ao histórico (antes de enviar)
    # Nota: A API do Claude espera que o histórico já contenha a mensagem do usuário.
    current_history = message_history + [{"role": "user", "content": user_message}]

    # 2. Monta o 'body' no formato do Claude 3 (Messages API)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", # Versão da API necessária
        "max_tokens": 1024,
        "temperature": 0.5,
        "system": SYSTEM_PROMPT,
        "messages": current_history 
    })

    # 3. Lógica de retentativas (REUTILIZADA do seu script)
    retries = 3
    delay = 5
    for i in range(retries):
        try:
            # 4. Chama o modelo
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # 5. Analisa a resposta (Formato do Claude)
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # A resposta do Claude está em 'content'
            if response_body.get('type') == 'message' and response_body.get('content'):
                assistant_response = response_body['content'][0]['text']
                
                # 6. Adiciona a resposta do assistente ao histórico
                # Retornamos o histórico ATUALIZADO para o loop principal
                updated_history = current_history + [{"role": "assistant", "content": assistant_response}]
                
                return assistant_response, updated_history
            else:
                raise Exception(f"Resposta JSON com estrutura inválida: {response_body}")

        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                logging.warning(f"Throttling... retentativa em {delay}s.")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Erro do cliente Bedrock: {e}")
                return None, current_history # Retorna None e o histórico *antes* da falha
        except Exception as e:
            logging.error(f"Erro ao processar a resposta: {e}")
            logging.error(f"Resposta JSON inválida ou incompleta: {response_body}")
            return None, current_history

    logging.error(f"Excedeu retentativas para a mensagem: {user_message}")
    return None, current_history


# %% [markdown]
# ### 3. Loop de Chat Interativo
# 
# Esta é a parte principal da "aplicação". Ela fica em loop, esperando seu input,
# chamando a função `chamar_claude_chat` e imprimindo a resposta.
# 
# **Para parar, digite `sair` ou `exit`.**

# %%
print("💬 Iniciando chat com Claude 3 Sonnet (via Bedrock).")
print("Digite 'sair' ou 'exit' para terminar a conversa.")
print("---")

# O 'messages' global será atualizado a cada turno
while True:
    try:
        # 1. Pega o input do usuário
        user_input = input("Você: ")

        if not user_input:
            continue
            
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("\n👋 Encerrando o chat. Até logo!")
            break
            
        # 2. Chama a função de chat
        # A variável 'messages' é sobrescrita com o histórico atualizado
        assistant_reply, messages = chamar_claude_chat(user_input, messages)
        
        # 3. Imprime a resposta
        if assistant_reply:
            print(f"\nClaude: {assistant_reply}\n")
        else:
            print("\nClaude: (Ocorreu um erro ao processar sua mensagem.)\n")
            # Se deu erro, remove a última mensagem do usuário do histórico
            # para não enviá-la novamente no próximo turno.
            if messages and messages[-1]["role"] == "user":
                messages.pop()

    except KeyboardInterrupt:
        print("\n👋 Encerrando o chat. Até logo!")
        break
    except Exception as e:
        print(f"Erro inesperado no loop principal: {e}")
        break
