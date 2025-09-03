import json

# Coloque aqui o nome do arquivo que está dando erro
ARQUIVO_PARA_DIAGNOSTICAR = "stereoset_intersentence_validation_pt.json" 
# Se o erro for no outro, troque o nome do arquivo acima.

def find_json_error(file_path):
    """
    Tenta carregar um arquivo JSON e, em caso de erro,
    fornece informações detalhadas sobre a localização e a causa.
    """
    print(f"--- Diagnosticando o arquivo: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("\n[SUCESSO] Nenhuma corrupção encontrada. O arquivo parece ser um JSON válido.")
    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{file_path}' não foi encontrado.")
    except json.JSONDecodeError as e:
        print("\n[FALHA] Corrupção encontrada no arquivo JSON!")
        print("-------------------------------------------------")
        print(f"Motivo do Erro: {e.msg}")
        print(f"Linha do Erro  : {e.lineno}")
        print(f"Coluna do Erro : {e.colno}")
        print("-------------------------------------------------")
        
        # Tenta mostrar o texto exato onde o erro ocorreu
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print("Trecho do código com o problema (linha {e.lineno}):")
            
            # Mostra a linha do erro e algumas linhas antes e depois para dar contexto
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            
            for i in range(start, end):
                line_num = i + 1
                prefix = ">>" if line_num == e.lineno else "  "
                print(f"{prefix} {line_num:4d}| {lines[i].strip()}")
                
                if line_num == e.lineno:
                    # Adiciona um marcador de coluna para apontar o local exato
                    marker = " " * (e.colno + 8) + "^"
                    print(marker)
                    
        except Exception as read_exc:
            print(f"Não foi possível exibir o trecho do código. Erro: {read_exc}")
            
if __name__ == "__main__":
    find_json_error(ARQUIVO_PARA_DIAGNOSTICAR)
