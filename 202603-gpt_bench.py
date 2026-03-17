eval_generative_models.py:

//////


import json
import os
from argparse import ArgumentParser
from collections import Counter
from random import shuffle

import numpy as np
import torch
import transformers
from colorama import Back, Fore, Style, init
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import dataloader
from intersentence_loader import IntersentenceDataset
# MODIFICADO AQUI 1: Usando AutoModel em vez de importar o arquivo 'models' local obsoleto
from transformers import AutoModelForCausalLM, AutoTokenizer 

init()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-class", default="gpt2", type=str,
                        help="Choose the pretrained model to load.")
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--input-file", default="../data/dev.json",
                        type=str, help="Choose the dataset to evaluate on.")
    parser.add_argument("--output-dir", default="predictions/", type=str,
                        help="Choose the output directory to store predictions in.")
    parser.add_argument("--intrasentence-model",
                        default="GPT2LM", type=str,
                        help="Choose a model architecture for the intrasentence task.")
    parser.add_argument("--intrasentence-load-path", default=None,
                        help="Load a pretrained model for the intrasentence task.")

    parser.add_argument("--intersentence-model",
                        default="ModelNSP", type=str, help="Choose a intersentence model architecture.")
    parser.add_argument("--intersentence-load-path", default=None, 
                        help="Load a pretrained model for the intersentence task.")

    parser.add_argument("--tokenizer", default="GPT2Tokenizer", type=str)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--unconditional_start_token",
                        default="<|endoftext|>", type=str, help="Beginning of sequence token.")
    parser.add_argument("--skip-intersentence",
                        default=False, action="store_true", help="Skip the intersentence task.")
    parser.add_argument("--skip-intrasentence",
                        default=False, action="store_true", help="SKip the intrasentence task.")
    parser.add_argument("--small", default=False, action="store_true")
    return parser.parse_args()


class BiasEvaluator(object):
    def __init__(self, pretrained_class="gpt2", no_cuda=False, batch_size=51, input_file="data/bias.json",
                 intrasentence_model="GPT2LM", intrasentence_load_path=None, intersentence_model="ModelNSP",
                 intersentence_load_path=None, tokenizer="GPT2Tokenizer", unconditional_start_token="<|endoftext|>",
                 skip_intrasentence=False, skip_intersentence=False, max_seq_length=64, small=False,
                 output_dir="predictions/"):
        print(f"Loading {input_file}...")
        self.BATCH_SIZE = batch_size
        filename = os.path.abspath(input_file)
        self.dataloader = dataloader.StereoSet(filename)
        self.cuda = not no_cuda
        self.device = "cuda" if self.cuda else "cpu"
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        
        self.PRETRAINED_CLASS = pretrained_class
        
        # MODIFICADO AQUI 2: Carregamento universal do tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(self.PRETRAINED_CLASS)
        
        # Pega o token de inicio nativo do modelo (se existir), senao usa o eos
        self.UNCONDITIONAL_START_TOKEN = self.tokenizer.bos_token if self.tokenizer.bos_token else self.tokenizer.eos_token

        self.INTRASENTENCE_MODEL = intrasentence_model
        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_MODEL = intersentence_model
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path
        self.max_seq_length = max_seq_length

        print("---------------------------------------------------------------")
        print(
            f"{Fore.LIGHTCYAN_EX}                     ARGUMENTS                 {Style.RESET_ALL}")
        print(
            f"{Fore.LIGHTCYAN_EX}Pretrained class:{Style.RESET_ALL} {pretrained_class}")
        print(f"{Fore.LIGHTCYAN_EX}Unconditional Start Token: {Style.RESET_ALL} {self.UNCONDITIONAL_START_TOKEN}")
        print("---------------------------------------------------------------")

    def evaluate_intrasentence(self):
        print()
        print(
            f"{Fore.LIGHTRED_EX}Evaluating bias on intrasentence tasks...{Style.RESET_ALL}")

        # MODIFICADO AQUI 3: Usando AutoModel para não depender do models.py local
        model = AutoModelForCausalLM.from_pretrained(self.PRETRAINED_CLASS).to(self.device)
        model.eval()

        start_token = torch.tensor(self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN)).to(self.device).unsqueeze(0)
            
        # MODIFICADO AQUI 4: Adicionado '.logits' na saida do modelo
        with torch.no_grad():
            initial_token_probabilities = model(start_token).logits
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)

        clusters = self.dataloader.get_intrasentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            for sentence in cluster.sentences:
                probabilities = {}
                tokens = self.tokenizer.encode(sentence.sentence)
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)
                    
                # MODIFICADO AQUI 4: Adicionado '.logits'
                with torch.no_grad():
                    output = torch.softmax(model(tokens_tensor).logits[0], dim=-1)
                    
                for idx in range(1, len(tokens)):
                    joint_sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
                score /= len(joint_sentence_probability)
                score = np.power(2, score)

                probabilities['id'] = sentence.ID
                probabilities['score'] = score

                predictions.append(probabilities)

        return predictions

    def evaluate_intersentence(self):
        # MODIFICADO AQUI 3: Usando AutoModel
        model = AutoModelForCausalLM.from_pretrained(self.PRETRAINED_CLASS).to(self.device)

        start_token = torch.tensor(self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN)).to(self.device).unsqueeze(0)
            
        # MODIFICADO AQUI 4: Adicionado '.logits'
        with torch.no_grad():
            initial_token_probabilities = model(start_token).logits
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)

        model.eval()
        clusters = self.dataloader.get_intersentence_examples()[:1000]
        predictions = []

        for cluster in tqdm(clusters):
            context = cluster.context
            for sentence in cluster.sentences:
                probabilities = {}
                if context[-1] not in [".", "!", "?"]:
                    context = f"{context}."
                full_sentence = f"{context} {sentence.sentence}"

                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)

                context_length = len(self.tokenizer.encode(context))

                sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[context_length]].item()]
                context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]

                # MODIFICADO AQUI 4: Adicionado '.logits'
                with torch.no_grad():
                    logits = model(tokens_tensor).logits
                output = torch.softmax(logits[0], dim=-1)

                for idx in range(1, context_length):
                    context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                for idx in range(1, len(tokens)):
                    sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                full_sentence = f"{sentence.sentence}"
                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)
                no_context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]
                    
                # MODIFICADO AQUI 4: Adicionado '.logits'
                with torch.no_grad():
                    logits = model(tokens_tensor).logits
                output = torch.softmax(logits[0], dim=-1)

                for idx in range(1, len(tokens)):
                    no_context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                context_score = np.mean([np.log2(i) for i in context_probability])
                sentence_score = np.mean([np.log2(i) for i in sentence_probability])
                no_context_score = np.mean([np.log2(i) for i in no_context_probability])

                overall_score = no_context_score / context_score
                probabilities['id'] = sentence.ID
                probabilities['score'] = overall_score

                predictions.append(probabilities)
        return predictions

    def evaluate(self):
        bias = {}
        if not self.SKIP_INTRASENTENCE:
            intrasentence_bias = self.evaluate_intrasentence()
            bias['intrasentence'] = intrasentence_bias
        if not self.SKIP_INTERSENTENCE:
            # MODIFICADO AQUI 5: Forçamos a usar intersentence padrão (LLMs não tem NSP)
            intersentence_bias = self.evaluate_intersentence()
            bias['intersentence'] = intersentence_bias
        return bias


if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(**vars(args))
    results = evaluator.evaluate()
    # MODIFICADO AQUI 6: Limpei o nome do arquivo final para evitar erro se o nome do modelo tiver barra '/'
    safe_name = args.pretrained_class.replace("/", "_")
    output_file = os.path.join(args.output_dir, f"predictions_{safe_name}.json")
    with open(output_file, "w+") as f:
        json.dump(results, f, indent=2)





Aqui estão os comandos prontos para você rodar no seu terminal usando o script atualizado. 

Selecionei os modelos abertos (open-weights) que fazem mais sentido para avaliar viés em **Português** e que possuem arquitetura Causal LM (família GPT).

### 1. Llama 3.1 (8B) - O Padrão Ouro Atual
O modelo da Meta tem uma base de dados gigantesca e excelente compreensão do Português, servindo como o melhor baseline geral.
```bash
python eval_generative_models.py \
  --pretrained-class "meta-llama/Meta-Llama-3.1-8B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```
*(Nota: Llama é um modelo "gated". Você precisará ter aceitado os termos no site do Hugging Face e rodado `huggingface-cli login` no seu terminal antes).*

### 2. Qwen 2.5 (7B) - O Multilíngue Superlativo
A família Qwen (da Alibaba) é atualmente uma das mais fortes em benchmarks não-ingleses, superando frequentemente o Llama em fluência multilíngue.
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-7B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

### 3. Gemma 2 (9B) - Arquitetura Google
Excelente para comparar com os resultados dos modelos acima. Possui uma arquitetura muito eficiente baseada no Gemini.
```bash
python eval_generative_models.py \
  --pretrained-class "google/gemma-2-9b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

### 4. Bode (7B) - Especialista em Português (PT-BR)
Este é um modelo finetunado especificamente para o Português do Brasil pela equipe do Recogna-NLP. É vital testar um modelo nativo para ver se o viés cultural brasileiro aparece de forma diferente dos modelos globais.
```bash
python eval_generative_models.py \
  --pretrained-class "recogna-nlp/bode-7b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

### 5. Qwen 2.5 (1.5B) - Para Teste Rápido (Pouca VRAM)
Se você quiser apenas rodar um teste rápido para garantir que o script não vai dar erro e que o JSON está sendo gerado corretamente (sem esperar horas ou estourar a memória da GPU), use este modelo menor:
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-1.5B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

**Dica de Execução:** O script modificado salva os resultados usando o nome do modelo (ex: `predictions_Qwen_Qwen2.5-7B.json`). Depois de rodar esses comandos, basta pegar esses arquivos JSON da pasta `predictions/` e passar para o seu script de pontuação (score) do StereoSet!


Com base no escopo do seu projeto (avaliar viés e estereótipos em texto traduzido para o Português do Brasil usando LLMs), selecionei os modelos abertos mais relevantes atualmente. 

Dividi em categorias para você testar desde opções muito leves (para ver se o script roda sem erros) até os pesos-pesados da categoria de código aberto.

Aqui estão os comandos prontos para copiar e colar usando o seu arquivo `dev.json`:

### 1. Teste Rápido de Sanidade (Modelos Pequenos)
*Ideais para rodar primeiro e garantir que o pipeline, o JSON e o script estão funcionando perfeitamente sem demorar horas ou exigir muita memória (VRAM).*

**Qwen 2.5 (1.5B)** - Excelente modelo pequeno multilíngue da Alibaba.
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-1.5B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Phi-3 Mini (3.8B)** - Modelo altamente otimizado da Microsoft.
```bash
python eval_generative_models.py \
  --pretrained-class "microsoft/Phi-3-mini-4k-instruct" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### 2. Baselines Globais (Padrão Ouro Atual)
*Estes são os modelos que você **precisa** ter na sua avaliação, pois são as referências acadêmicas e de mercado em 2024/2026. Apesar de globais, têm volumes absurdos de Português no treinamento.*

**Llama 3 (8B) - Meta** - O baseline absoluto para qualquer pesquisa atual.
```bash
python eval_generative_models.py \
  --pretrained-class "meta-llama/Meta-Llama-3-8B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Qwen 2.5 (7B)** - Em muitos benchmarks multilíngues, supera o Llama 3 no Português.
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-7B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Gemma 2 (9B) - Google** - Arquitetura de ponta do Google (baseada no Gemini).
```bash
python eval_generative_models.py \
  --pretrained-class "google/gemma-2-9b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### 3. Modelos Focados em Português (Nativos / Finetuned)
*Essenciais para investigar se um modelo treinado especificamente com a cultura e o texto brasileiro reflete viés de forma diferente dos modelos globais.*

**Bode (7B)** - Modelo Llama treinado/adaptado pelo grupo Recogna (UNESP) focado em PT-BR.
```bash
python eval_generative_models.py \
  --pretrained-class "recogna-nlp/bode-7b-v1" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Sabiá 7B (Maritaca AI)** - Uma versão menor de um dos projetos mais ambiciosos focados em PT-BR. *(Nota: O repositório público deles muda ocasionalmente, mas este é o padrão)*
```bash
python eval_generative_models.py \
  --pretrained-class "maritaca-ai/sabia-7b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### 💡 Dicas de Execução

1. **Permissões no Hugging Face:** Modelos como o `meta-llama/Meta-Llama-3-8B` e o `google/gemma-2-9b` são *"Gated Models"*. Isso significa que antes de rodar o comando, você precisa ir no site do Hugging Face, aceitar os termos deles, e no seu terminal rodar `huggingface-cli login` para inserir seu token de acesso.
2. **Estouro de Memória (OOM):** Se a sua placa de vídeo não aguentar os modelos de 7B a 9B (que exigem cerca de 16GB a 20GB de VRAM em precisão total), você pode rodar o script forçando o carregamento em meia precisão (FP16) se o ambiente suportar, ou testar na Nuvem (Google Colab, AWS, etc.).

Para o escopo do seu projeto — avaliar o viés (StereoSet) em **Português** utilizando arquiteturas modernas (Causal LM) —, a escolha dos modelos é fundamental. O foco deve ser em modelos com pré-treinamento maciço que incluam o nosso idioma ou que tenham sido feitos especificamente para ele.

Aqui estão os comandos prontos para os modelos mais relevantes da atualidade. Separei entre as versões **Grandes (State-of-the-Art)** e as **Eficientes (Excelentes baselines)**.

### 1. Os "Pesos-Pesados" Globais (Modelos Grandes)
*Estes são os melhores modelos abertos do mundo atualmente. Eles possuem um raciocínio multilíngue absurdamente forte. **Atenção:** Versões de 70B/72B exigem múltiplas GPUs (ex: 2 a 4x RTX 3090/4090 ou A100) para rodar.*

**Llama 3.1 (70 Bilhões de parâmetros) - Meta**
O padrão-ouro atual da Meta. Tem um suporte robusto para português devido ao volume colossal de dados de treinamento.
```bash
python eval_generative_models.py \
  --pretrained-class "meta-llama/Meta-Llama-3.1-70B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Qwen 2.5 (72 Bilhões de parâmetros) - Alibaba**
Atualmente, a família Qwen 2.5 tem superado o Llama 3.1 em benchmarks de idiomas não-ingleses. É uma escolha obrigatória para testes rigorosos de viés multilíngue.
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-72B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### 2. Os Eficientes e Modernos (Fáceis de Rodar)
*Estes modelos na faixa de 7B a 27B rodam facilmente em uma única GPU (como uma RTX 3090/4090 ou T4/A10G na nuvem) e entregam resultados comparáveis aos gigantes de gerações anteriores.*

**Gemma 2 (27 Bilhões de parâmetros) - Google**
Arquitetura baseada no Gemini. O tamanho de 27B oferece o melhor custo-benefício entre inteligência e facilidade de rodar, com ótimo suporte ao português.
```bash
python eval_generative_models.py \
  --pretrained-class "google/gemma-2-27b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Llama 3.1 (8 Bilhões de parâmetros) - Meta**
O baseline fundamental. Se você for publicar um artigo ou relatório, a comunidade científica espera ver os resultados do Llama 3.1 8B.
```bash
python eval_generative_models.py \
  --pretrained-class "meta-llama/Meta-Llama-3.1-8B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

**Qwen 2.5 (7 Bilhões de parâmetros) - Alibaba**
O competidor direto do Llama 8B, frequentemente com vocabulário mais rico em PT-BR.
```bash
python eval_generative_models.py \
  --pretrained-class "Qwen/Qwen2.5-7B" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### 3. Especialistas em Português (Finetuned no Brasil)
*Testar um modelo focado no Brasil é vital, pois os estereótipos capturados por ele podem ser muito diferentes dos capturados por modelos puramente globais.*

**Bode (14 Bilhões de parâmetros) - Recogna NLP / C4AI**
Um modelo construído a partir do Llama, mas com *continuation pre-training* (treinamento contínuo) focado pesadamente em Português Brasileiro por pesquisadores da USP.
```bash
python eval_generative_models.py \
  --pretrained-class "recogna-nlp/bode-14b" \
  --input-file "dev.json" \
  --output-dir "predictions/"
```

---

### ⚠️ Dica Crucial sobre Acesso aos Modelos
Alguns desses modelos (especialmente **Llama 3.1** e **Gemma 2**) são *Gated Models* no Hugging Face. Isso significa que você precisa:
1. Ter uma conta no Hugging Face.
2. Ir na página do modelo (ex: `https://huggingface.co/meta-llama/Meta-Llama-3.1-8B`) e clicar em "Agree to terms" (Aceitar os termos).
3. Logar no seu terminal antes de rodar o script usando o comando:
   ```bash
   huggingface-cli login
   ```
   *(Ele pedirá seu Access Token, que você gera nas configurações da sua conta no Hugging Face).*
