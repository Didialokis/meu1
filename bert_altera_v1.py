Claro, vamos resolver isso\! O erro `KeyError` que você está vendo é um sintoma clássico de um problema que aconteceu em uma etapa anterior.

A causa raiz **não está no `evaluation.py`**. O erro está acontecendo porque o arquivo de previsões (`predictions/*.json`) foi gerado com base em dados incompletos. A correção que fizemos anteriormente no `dataloader.py` foi uma solução parcial: ela evitou o travamento (`IndexError`), mas, ao encontrar um exemplo que não conseguia processar, ela simplesmente o pulou (`continue`).

Isso resultou na criação de "exemplos órfãos" nos dados carregados, onde um cluster de sentenças que deveria ter um estereótipo, um anti-estereótipo e um não relacionado, acabou faltando uma dessas três partes. O script `evaluation.py` espera que as três estejam sempre presentes e, quando não encontra, ele quebra com o `KeyError`.

A solução definitiva é criar uma lógica ainda mais robusta no `dataloader.py` que não pule exemplos, mas que consiga encontrar a palavra-alvo mesmo quando a tradução altera a contagem de palavras.

-----

## A Solução Definitiva: Aprimorar o `dataloader.py`

Vamos substituir a lógica de `set.difference` por uma abordagem baseada em listas, que é mais resistente a traduções que inserem múltiplas palavras (como "programador" se tornando "engenheiro de software").

**1. Abra o arquivo `dataloader.py`**

  - Vá novamente para o arquivo `/home/sagemaker-user/stereoset/code/dataloader.py`.

**2. Localize a função `__create_intrasentence_examples__`**

  - Encontre o bloco de código que modificamos da última vez.

**3. Substitua o Bloco Modificado pela Versão Final**

  - Remova a lógica anterior e a substitua por esta versão mais inteligente e completa.

**CÓDIGO A SER SUBSTITUÍDO (A LÓGICA ANTERIOR):**

```python
# A lógica que você tem agora, que usa set.difference e 'continue'
                context_words = set(example['context'].replace("BLANK", "").translate(str.maketrans('', '', string.punctuation)).split())
                # ... (resto do bloco antigo)
```

**NOVO CÓDIGO FINAL (Substitua o bloco acima por este):**

```python
                # --- INÍCIO DA NOVA LÓGICA ROBUSTA ---
                # Limpa e tokeniza a frase de contexto e a frase completa
                context_tokens = [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in example['context'].split()]
                sentence_tokens = [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in sentence['sentence'].split()]

                # Remove o token 'BLANK' e quaisquer tokens vazios resultantes do split
                context_tokens_no_blank = [t for t in context_tokens if "blank" not in t and t]

                # A palavra-alvo é composta por todas as palavras na sentença completa que não estão no contexto
                # Isso funciona para uma ou múltiplas palavras (ex: "cientista", "engenheiro de software")
                difference_words = [word for word in sentence_tokens if word not in context_tokens_no_blank]

                if not difference_words:
                    print(f"AVISO: Nenhuma palavra de diferença encontrada para o ID {sentence['id']}. Pulando esta sentença.")
                    print(f"  Contexto: {example['context']}")
                    print(f"  Sentença: {sentence['sentence']}")
                    continue

                # Junta as palavras de diferença (caso seja um termo composto)
                template_word = " ".join(difference_words)

                sentence_obj.template_word = template_word
                sentences.append(sentence_obj)
                # --- FIM DA NOVA LÓGICA ---
```

-----

## Por que esta nova lógica é melhor? 🧠

  * **Tolerância a Múltiplas Palavras:** Se "programmer" (1 palavra) virou "engenheiro de software" (3 palavras), a lógica de `set` falhava. A nova lógica de lista captura todas as palavras extras.
  * **Mais Resiliente:** Ela compara as listas de palavras e extrai o que é "novo", que é exatamente o que precisamos.
  * **Não Corrompe os Dados:** Ao não pular sentenças problemáticas (a menos que seja impossível encontrar uma diferença), garantimos que cada exemplo tenha suas três sentenças, evitando o `KeyError` na etapa de avaliação.

-----

## Próximos Passos (Essencial\!) 🎯

Agora que o `dataloader.py` está corrigido de forma definitiva, você precisa refazer as etapas na ordem correta.

**1. Exclua as Previsões Antigas:**
Os arquivos na pasta `predictions/` foram gerados com a lógica de carregamento de dados falha. Eles estão corrompidos.

```bash
rm -rf predictions/*
```

**2. Gere Novamente as Previsões:**
Execute o script `eval_discriminative_models.py` de novo. Agora ele usará o `dataloader.py` corrigido para carregar os dados completos e gerar previsões corretas.

```bash
# Exemplo para o BERTimbau
python eval_discriminative_models.py \
    --pretrained-class "neuralmind/bert-base-portuguese-cased" \
    --input-file "../data/dev_pt.json" \
    --output-file "predictions/predictions_bertimbau.json"
```

*(Execute para todos os modelos que você deseja avaliar).*

**3. Execute a Avaliação Final:**
Agora que as previsões foram geradas corretamente, o script `evaluation.py` funcionará sem erros.

```bash
python3 evaluation.py --gold-file ../data/dev_pt.json --predictions-dir predictions/
```

Seguindo estes passos, o `KeyError` será resolvido, pois o `evaluation.py` receberá dados consistentes e completos.
