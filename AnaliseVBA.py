Perfeito, obrigado pela nova imagem. Ela confirma a estrutura exata das colunas. Vejo que os nomes e as posições são um pouco diferentes da primeira imagem, e vou ajustar as fórmulas exatamente para essa nova tabela.

Com base nesta imagem, vamos usar:

  * Sua aba com os dados se chama **Plan1** (lembre-se de trocar se o nome for outro).
  * **Coluna H**: `bias_type`
  * **Coluna I**: `gold_label`
  * **Coluna J**: `resposta` (com os valores "correto" e "incorreto")

O plano é o mesmo e continua bem simples. Vamos apenas ajustar as letras das colunas nas fórmulas.

-----

### Passo 1: Adicionar Colunas de Cálculo na sua Aba de Dados ("Plan1")

Vá para a sua planilha de dados. Vamos adicionar duas colunas auxiliares ao lado dos seus dados, nas colunas K e L.

1.  **Na célula K1**, digite o título: `Pontos Obtidos`

2.  **Na célula L1**, digite o título: `Pontos Máximos`

3.  **Na célula K2**, cole a seguinte fórmula (ajustada para as novas colunas):

    ```excel
    =SE(J2="correto";SE(OU(I2="stereotype";I2="anti-stereotype");2;SE(I2="unrelated";1;0));0)
    ```

4.  **Na célula L2**, cole a seguinte fórmula (ajustada para a nova coluna):

    ```excel
    =SE(OU(I2="stereotype";I2="anti-stereotype");2;SE(I2="unrelated";1;0))
    ```

5.  Agora, **selecione as células K2 e L2**, clique no pequeno quadrado no canto inferior direito da seleção e **arraste para baixo** até o final de todos os seus dados.

Com isso, os cálculos de cada linha já estão prontos. Agora vamos para o resumo.

### Passo 2: Criar a Tabela de Resumo em uma Nova Aba

1.  Crie uma nova aba e chame-a de **"Resumo"**.
2.  Monte a mesma estrutura que planejamos antes:

| | A | B | C | D |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Categoria** | **Pontos Obtidos** | **Pontos Máximos (100%)** | **Aproveitamento**|
| **2** | **Total Geral** | | | |
| **3** | | | | |
| **4** | **Por Viés** | | | |
| **5** | gender | | | |
| **6** | profession | | | |
| **7** | ... (etc) | | | |

*Lembre-se de listar seus tipos de viés na Coluna A, a partir da célula A5.*

### Passo 3: Inserir as Fórmulas Corrigidas na Aba "Resumo"

Agora, vamos usar as fórmulas de soma, apontando para as colunas corretas.

#### Para o Total Geral:

1.  **Na célula B2** (Total de Pontos Obtidos), cole:
    ```excel
    =SOMA(Plan1!K:K)
    ```
2.  **Na célula C2** (Total de Pontos Máximos), cole:
    ```excel
    =SOMA(Plan1!L:L)
    ```
3.  **Na célula D2** (Aproveitamento Geral), cole e formate como porcentagem:
    ```excel
    =B2/C2
    ```

#### Para o Detalhamento por Viés:

1.  **Na célula B5** (Pontos Obtidos para "gender"), cole a fórmula:

    ```excel
    =SOMASES(Plan1!K:K; Plan1!H:H; A5)
    ```

2.  **Na célula C5** (Pontos Máximos para "gender"), cole a fórmula:

    ```excel
    =SOMASES(Plan1!L:L; Plan1!H:H; A5)
    ```

    *(Observação: Como sua coluna `bias_type` agora está "limpa" (ex: "gender"), não precisamos mais do curinga `"*"` na fórmula. Ela ficou ainda mais simples\!)*

3.  **Na célula D5** (Aproveitamento para "gender"), cole a fórmula:

    ```excel
    =B5/C5
    ```

4.  Finalmente, **selecione B5, C5 e D5** e **arraste para baixo** para preencher as fórmulas para os outros tipos de viés.

Pronto\! Agora sua tabela de resumo funcionará perfeitamente com a estrutura de dados da imagem que você enviou.
