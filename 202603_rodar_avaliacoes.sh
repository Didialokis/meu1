#!/bin/bash

# ==============================================================================
# BATERIA DE AVALIAÇÃO STEREOSET - MODELOS MODERNOS (2024/2026)
# ==============================================================================

# Arquivo de entrada conforme solicitado
INPUT_FILE="dev.json"
OUTPUT_DIR="predictions/"

# Cria o diretório de predições se não existir
mkdir -p "$OUTPUT_DIR"

# Lista de Modelos Principais (Tamanho Base: 7B a 9B)
# Ideais para rodar em GPUs como RTX 3090, 4090 ou A10G (24GB VRAM)
MODELOS=(
    "meta-llama/Meta-Llama-3.1-8B"       # Padrão-ouro atual de raciocínio
    "Qwen/Qwen2.5-7B"                    # Suporte multilíngue excepcional (Alibaba)
    "google/gemma-2-9b"                  # Arquitetura modernizada do Google
    "recogna-nlp/bode-7b-alpaca-pt-br"   # Finetune do Llama focado especificamente em PT-BR
)

# ⚠️ MODELOS GIGANTES (Descomente se tiver múltiplas GPUs de 80GB, como A100/H100)
# MODELOS+=(
#     "meta-llama/Meta-Llama-3.1-70B"
#     "Qwen/Qwen2.5-72B"
#     "google/gemma-2-27b"
# )

echo "🚀 Iniciando bateria de avaliações do StereoSet em LLMs..."
echo "Arquivo de entrada: $INPUT_FILE"
echo "Resultados serão salvos em: $OUTPUT_DIR"
echo "------------------------------------------------------------"

# Loop que roda o comando Python para cada modelo da lista
for MODELO in "${MODELOS[@]}"; do
    echo ""
    echo "========================================================"
    echo "⚡ Carregando e avaliando: $MODELO"
    echo "========================================================"
    
    # O comando modificado que você tem no seu eval_generative_models.py
    python eval_generative_models.py \
        --pretrained-class "$MODELO" \
        --input-file "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR"
        
    # Verifica se o comando rodou com sucesso
    if [ $? -eq 0 ]; then
        echo "✅ Sucesso! Previsões do $MODELO salvas."
    else
        echo "❌ Erro ao avaliar o modelo $MODELO. Pulando para o próximo..."
    fi
done

echo ""
echo "🎉 Bateria de avaliações totalmente concluída!"
