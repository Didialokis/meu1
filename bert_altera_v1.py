# ... (início da função)
                # Nova lógica robusta para encontrar a palavra-alvo
                context_words = set(example['context'].replace("BLANK", "").translate(str.maketrans('', '', string.punctuation)).split())
                sentence_words = set(sentence['sentence'].translate(str.maketrans('', '', string.punctuation)).split())

                # A palavra-alvo é a que está no conjunto da sentença, mas não no do contexto
                difference = sentence_words.difference(context_words)

                if len(difference) != 1:
                    # Se a diferença não for exatamente uma palavra, algo deu errado na tradução ou na lógica.
                    # Isso ajuda a depurar casos estranhos.
                    print(f"AVISO: Não foi possível encontrar uma única palavra de diferença para o ID {sentence['id']}.")
                    print(f"Contexto: {example['context']}")
                    print(f"Sentença: {sentence['sentence']}")
                    print(f"Diferença encontrada: {difference}")
                    # Pular este exemplo problemático para não quebrar a execução
                    continue 

                template_word = difference.pop()
                sentence_obj.template_word = template_word
                sentences.append(sentence_obj)
# ... (resto da função)
