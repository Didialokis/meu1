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
