Com certeza! Para criar um resumo simples com os totais gerais e os valores separados por tipo de viés, usando apenas fórmulas do Excel, a melhor abordagem é criar uma pequena tabela de resumo.

Vamos usar as funções CONT.SES (para contar com múltiplos critérios) e CONT.SE (para contar com um critério).

Siga estes passos:

Passo 1: Prepare a Tabela de Resumo
Crie uma nova aba (planilha) no seu arquivo e chame-a de "Resumo".

Nesta nova aba, monte a seguinte estrutura. Você precisará digitar os cabeçalhos e os tipos de viés manualmente na primeira coluna.

A	B	C	D
1	Tipo de Viés	Total de Acertos (Contagem 'c')	Pontuação Obtida	Pontuação Máxima Possível
2	gender	(Fórmula aqui)	(Fórmula aqui)	(Fórmula aqui)
3	profession	(Fórmula aqui)	(Fórmula aqui)	(Fórmula aqui)
4	race	(Fórmula aqui)	(Fórmula aqui)	(Fórmula aqui)
5	religion	(Fórmula aqui)	(Fórmula aqui)	(Fórmula aqui)
6				
7	Total Geral	(Fórmula aqui)	(Fórmula aqui)	(Fórmula aqui)

Exportar para as Planilhas
Importante: Vou assumir que sua planilha com os dados brutos se chama "Dados". Se tiver outro nome, apenas substitua "Dados" pelo nome correto nas fórmulas.

Passo 2: Fórmulas para cada Tipo de Viés
Agora, vamos preencher as células da linha 2 (referente a "gender"). Depois, você poderá simplesmente arrastar as fórmulas para baixo para as outras categorias.

1. Total de Acertos (Célula B2)
Esta fórmula conta quantas vezes a nota "c" aparece para o tipo de viés "gender".

Clique na célula B2 e insira:

Excel

=CONT.SES(Dados!A:A;"*"&A2&"*";Dados!K:K;"c")
Dados!A:A;"*"&A2&"*": Procura na coluna A (bias_type) por qualquer texto que contenha a palavra da célula A2 (neste caso, "gender").

Dados!K:K;"c": Adiciona a condição de que a coluna K (nota) seja igual a "c".

2. Pontuação Máxima Possível (Célula D2)
Esta fórmula calcula qual seria a pontuação total para "gender" se todas as respostas fossem corretas.

Clique na célula D2 e insira:

Excel

=(CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"stereotype")*2)+(CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"anti-stereotype")*2)+CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"unrelated")
Ela conta todos os "stereotype" e "anti-stereotype" para "gender" e multiplica por 2, e depois soma a contagem de todos os "unrelated".

3. Pontuação Obtida (Célula C2)
Esta é a pontuação real, baseada apenas nos acertos ("c").

Clique na célula C2 e insira:

Excel

=(CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"stereotype";Dados!K:K;"c")*2)+(CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"anti-stereotype";Dados!K:K;"c")*2)+CONT.SES(Dados!A:A;"*"&A2&"*";Dados!J:J;"unrelated";Dados!K:K;"c")
É a mesma lógica da fórmula anterior, mas com o critério extra de que a coluna K (nota) deve ser "c".

Agora, para preencher o resto: Selecione as células B2, C2 e D2, clique no pequeno quadrado no canto inferior direito da seleção e arraste para baixo até a linha 5.

Passo 3: Fórmulas para o "Total Geral"
Estas fórmulas são mais simples, pois não precisam filtrar por tipo de viés.

1. Total de Acertos (Célula B7)
Clique na célula B7 e insira:

Excel

=CONT.SE(Dados!K:K;"c")
2. Pontuação Máxima Possível (Célula D7)
Clique na célula D7 e insira:

Excel

=(CONT.SE(Dados!J:J;"stereotype")*2)+(CONT.SE(Dados!J:J;"anti-stereotype")*2)+CONT.SE(Dados!J:J;"unrelated")
3. Pontuação Obtida (Célula C7)
Clique na célula C7 e insira:

Excel

=(CONT.SES(Dados!J:J;"stereotype";Dados!K:K;"c")*2)+(CONT.SES(Dados!J:J;"anti-stereotype";Dados!K:K;"c")*2)+CONT.SES(Dados!J:J;"unrelated";Dados!K:K;"c")
Resultado Final
Sua tabela de resumo ficará parecida com esta, preenchida automaticamente com os valores corretos da sua planilha de dados:

A	B	C	D
1	Tipo de Viés	Total de Acertos (Contagem 'c')	Pontuação Obtida	Pontuação Máxima Possível
2	gender	14	25	32
3	profession	8	14	20
4	race	5	8	11
5	religion	4	6	7
6				
7	Total Geral	31	53	70

Exportar para as Planilhas
(Nota: Os números no exemplo acima são ilustrativos, os seus serão calculados com base nos seus dados reais.)

Esta abordagem usa apenas fórmulas, é atualizada automaticamente se você alterar os dados brutos e fornece exatamente o resumo simples que você pediu.
