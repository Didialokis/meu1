import json

# --- CONFIGURAÇÕES ---
ARQUIVO_JSON_ENTRADA = 'stereoset_reconstruido.json'
RELATORIO_ERROS = 'relatorio_erros_posicao.txt'

def validar_posicionamento():
    print(f"🕵️  Iniciando validação rigorosa em: {ARQUIVO_JSON_ENTRADA}")
    
    with open(ARQUIVO_JSON_ENTRADA, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    erros = []
    total_intra = 0
    total_inter = 0

    # --- 1. VALIDAÇÃO INTRASENTENCE (A CRÍTICA) ---
    print("🔍 Verificando INTRASENTENCE (Posição do BLANK)...")
    
    for exemplo in data['intrasentence']:
        total_intra += 1
        ex_id = exemplo['id']
        contexto = exemplo['context']
        frases = exemplo['sentences']

        # Verifica se o BLANK existe
        if "BLANK" not in contexto:
            erros.append(f"[INTRA] ID {ex_id}: Contexto não contém a palavra chave 'BLANK'. Texto: '{contexto}'")
            continue

        # Verifica se há mais de um BLANK (erro de formatação)
        if contexto.count("BLANK") > 1:
            erros.append(f"[INTRA] ID {ex_id}: Contexto contém múltiplos 'BLANK'. Texto: '{contexto}'")
            continue

        # --- LÓGICA DE POSIÇÃO EXATA ---
        # Divide o contexto em antes e depois do BLANK
        # Ex: "Eu sou BLANK hoje" -> prefixo="Eu sou ", sufixo=" hoje"
        partes = contexto.split("BLANK")
        prefixo = partes[0]
        sufixo = partes[1] if len(partes) > 1 else ""

        for s in frases:
            texto_alvo = s['sentence']
            
            # Checagem 1: A frase começa com o prefixo?
            if not texto_alvo.startswith(prefixo):
                erros.append(
                    f"[INTRA] ID {ex_id}: Erro de Prefixo.\n"
                    f"   Contexto: '{contexto}'\n"
                    f"   Esperado início: '{prefixo}'\n"
                    f"   Encontrado: '{texto_alvo}'"
                )
                continue

            # Checagem 2: A frase termina com o sufixo?
            if not texto_alvo.endswith(sufixo):
                erros.append(
                    f"[INTRA] ID {ex_id}: Erro de Sufixo.\n"
                    f"   Contexto: '{contexto}'\n"
                    f"   Esperado final: '{sufixo}'\n"
                    f"   Encontrado: '{texto_alvo}'"
                )
                continue
            
            # Se passou, verificamos se o "miolo" (a palavra alvo) não está vazio
            # Removemos prefixo e sufixo para ver o que sobrou
            miolo = texto_alvo[len(prefixo):len(texto_alvo)-len(sufixo)]
            if not miolo.strip():
                erros.append(f"[INTRA] ID {ex_id}: A palavra alvo parece estar vazia ou é apenas espaço. Frase: '{texto_alvo}'")

    # --- 2. VALIDAÇÃO INTERSENTENCE (BÁSICA) ---
    print("🔍 Verificando INTERSENTENCE (Consistência)...")
    
    for exemplo in data['intersentence']:
        total_inter += 1
        if not exemplo['context'].strip():
            erros.append(f"[INTER] ID {exemplo['id']}: Contexto está vazio.")
        
        for s in exemplo['sentences']:
            if not s['sentence'].strip():
                erros.append(f"[INTER] ID {exemplo['id']}: Frase alvo vazia.")

    # --- RELATÓRIO FINAL ---
    print("\n" + "="*50)
    print(f"📊 RELATÓRIO FINAL DE VALIDAÇÃO")
    print("="*50)
    print(f"Intrasentence processados: {total_intra}")
    print(f"Intersentence processados: {total_inter}")
    print(f"Total de Erros Encontrados: {len(erros)}")
    print("-" * 50)

    if erros:
        print("❌ O JSON contém erros estruturais graves.")
        print(f"   Salvando detalhes em '{RELATORIO_ERROS}'...")
        
        with open(RELATORIO_ERROS, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ERROS DE POSIÇÃO E ESTRUTURA\n")
            f.write("===========================================\n\n")
            for erro in erros:
                f.write(erro + "\n\n")
                # Imprime alguns na tela para feedback imediato
                
        print("\nAMOSTRA DE ERROS (Veja o arquivo txt para todos):")
        for e in erros[:3]:
            print(e)
    else:
        print("✅ SUCESSO! Todos os exemplos respeitam a posição exata do BLANK e a estrutura do dataset.")
        print("   O arquivo JSON está pronto para ser usado no script de avaliação.")

if __name__ == "__main__":
    validar_posicionamento()
