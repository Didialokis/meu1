pip install aider-chat
export OPENAI_API_KEY=sua-chave-aqui
cd pasta-do-seu-projeto
aider
//////////////
# --- Essenciais para OpenAI & API ---
openai>=1.0.0       # A biblioteca oficial (versão mais recente)
python-dotenv       # Para gerenciar sua API Key via arquivo .env (segurança)
tiktoken            # Para contar tokens antes de enviar (economiza dinheiro)
requests            # Para chamadas HTTP genéricas (útil para testar APIs)

# --- Data Science & Analytics (Seus estudos e StereoSet) ---
numpy               # Cálculos matemáticos de alta performance
pandas              # Manipulação de CSV, JSON e tabelas (essencial)
openpyxl            # Dependência do Pandas para ler/escrever arquivos Excel (.xlsx)
matplotlib          # Criação de gráficos básicos
seaborn             # Gráficos estatísticos mais bonitos (baseado no matplotlib)
scikit-learn        # Machine Learning clássico (regressão, clustering para analytics)
tqdm                # Barra de progresso (ótimo para loops longos de processamento)

# --- Deep Learning & NLP (PyTorch e Textos) ---
# Nota: Veja a seção abaixo sobre a instalação do PyTorch
torch               # O framework de Deep Learning
torchvision         # Processamento de imagem (para seu projeto da balança/câmera)
transformers        # Hugging Face (padrão ouro para lidar com datasets como StereoSet)

# --- Hardware & IoT (ESP32) ---
pyserial            # Comunicação via porta USB/Serial com o ESP32
pillow              # Manipulação de imagens (redimensionar, converter antes de enviar)

# --- Cloud & Infra ---
boto3               # SDK da AWS (se for usar S3 ou outros serviços)
/////////////
def __create_intrasentence_examples__(self, examples):
        created_examples = []
        # Contador para não poluir o terminal (mostra apenas os 5 primeiros logs)
        debug_counter = 0 
        
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                
                sentence_obj = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                
                # Lógica existente: Tokens do contexto (sem 'blank') vs Tokens da frase
                context_tokens = [w for w in example['context'].lower().split() if 'blank' not in w]
                sentence_tokens = sentence['sentence'].lower().split()

                # Usa o SequenceMatcher
                matcher = SequenceMatcher(None, context_tokens, sentence_tokens)

                diff_words = []
                
                # --- INÍCIO DA VERIFICAÇÃO VISUAL ---
                if debug_counter < 5:
                    print(f"\n--- [DEBUG] ID: {sentence['id']} ---")
                    print(f"Contexto (sem blank): {context_tokens}")
                    print(f"Frase Alvo: {sentence_tokens}")

                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != 'equal':
                        # Captura as palavras diferentes
                        words_found = sentence['sentence'].split()[j1:j2]
                        diff_words.extend(words_found)
                        
                        if debug_counter < 5:
                            print(f"  > Diferença detectada ({tag}): {words_found}")

                if diff_words:
                    template_word = " ".join(diff_words)
                    # Remove pontuação para limpar o target
                    sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                    
                    if debug_counter < 5:
                        print(f"  > TARGET FINAL: '{sentence_obj.template_word}'")
                        debug_counter += 1
                        
                    sentences.append(sentence_obj)
                else:
                    if debug_counter < 5:
                        print("  > ALERTA: Nenhuma diferença encontrada (target vazio).")
                # --- FIM DA VERIFICAÇÃO VISUAL ---

            created_example = IntrasentenceExample(
                example['id'], example['bias_type'],
                example['target'], example['context'], sentences)
            created_examples.append(created_example)
            
        return created_examples
