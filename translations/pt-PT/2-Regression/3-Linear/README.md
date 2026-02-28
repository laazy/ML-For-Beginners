# Construir um modelo de regressão usando Scikit-learn: regressão de quatro formas

## Nota para iniciantes

A regressão linear é usada quando queremos prever um **valor numérico** (por exemplo, preço de casa, temperatura ou vendas). Funciona encontrando uma linha reta que melhor representa a relação entre as características de entrada e a saída.

Nesta lição, focamos em entender o conceito antes de explorar técnicas de regressão mais avançadas.
![Linear vs polynomial regression infographic](../../../../translated_images/pt-PT/linear-polynomial.5523c7cb6576ccab.webp)
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introdução

Até agora exploraste o que é regressão com dados de exemplo recolhidos do conjunto de dados de preços de abóboras que iremos usar ao longo desta lição. Também os visualizaste usando Matplotlib.

Agora estás pronto para mergulhar mais fundo na regressão para ML. Enquanto a visualização permite fazer sentido dos dados, o verdadeiro poder do Machine Learning vem do _treino de modelos_. Os modelos são treinados com dados históricos para capturar automaticamente dependências dos dados, e permitem prever resultados para novos dados, que o modelo ainda não viu.

Nesta lição, vais aprender mais sobre dois tipos de regressão: _regressão linear básica_ e _regressão polinomial_, juntamente com alguma matemática subjacente a estas técnicas. Esses modelos permitir-nos-ão prever preços de abóboras dependendo de diferentes dados de entrada.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Clica na imagem acima para um vídeo curto sobre regressão linear.

> Ao longo deste currículo, assumimos conhecimentos mínimos de matemática e procuramos torná-lo acessível a estudantes de outras áreas, por isso observa as notas, 🧮 chamadas, diagramas e outras ferramentas de aprendizagem para ajudar na compreensão.

### Pré-requisitos

Deves estar familiarizado agora com a estrutura dos dados das abóboras que estamos a examinar. Podes encontrá-los pré-carregados e pré-limpos no ficheiro _notebook.ipynb_ desta lição. No ficheiro, o preço da abóbora é apresentado por alqueire num novo dataframe. Certifica-te de que consegues executar estes notebooks em kernels no Visual Studio Code.

### Preparação

Como lembrete, estás a carregar estes dados para poder colocar-lhes questões.

- Qual é a melhor época para comprar abóboras?
- Que preço posso esperar por uma caixa de abóboras miniatura?
- Devo comprá-las em cestos de meio alqueire ou por caixas de 1 1/9 alqueires?
Vamos continuar a explorar estes dados.

Na lição anterior, criaste um dataframe Pandas e populaste-o com uma parte do conjunto de dados original, padronizando os preços por alqueire. Ao fazer isso, no entanto, só conseguiste recolher cerca de 400 pontos de dados e apenas para os meses de outono.

Dá uma vista de olhos aos dados que pré-carregámos no notebook acompanhante desta lição. Os dados estão pré-carregados e um gráfico de dispersão inicial é apresentado para mostrar dados mensais. Talvez possamos obter um pouco mais de detalhe sobre a natureza dos dados limpando-os melhor.

## Uma linha de regressão linear

Como aprendeste na Lição 1, o objetivo de um exercício de regressão linear é ser capaz de traçar uma linha para:

- **Mostrar relações entre variáveis**. Mostrar a relação entre variáveis.
- **Fazer previsões**. Fazer previsões precisas sobre onde um novo ponto de dados cairia em relação a essa linha.

É típico da **Regressão dos Mínimos Quadrados** traçar esse tipo de linha. O termo "Mínimos Quadrados" refere-se ao processo de minimizar o erro total no nosso modelo. Para cada ponto de dados, medimos a distância vertical (chamada residual) entre o ponto real e a nossa linha de regressão.

Elevamos ao quadrado essas distâncias por duas razões principais:

1. **Magnitude em vez de Direção:** Queremos tratar um erro de -5 igual a um erro de +5. Elevando ao quadrado, todos os valores tornam-se positivos.

2. **Penalização de Outliers:** Elevar ao quadrado dá mais peso a erros maiores, forçando a linha a ficar mais próxima dos pontos mais afastados.

Depois somamos todos esses valores ao quadrado. O nosso objetivo é encontrar a linha específica onde essa soma final é mínima (o menor valor possível) — daí o nome "Mínimos Quadrados".

> **🧮 Mostra-me a matemática**
> 
> Esta linha, chamada de _linha de melhor ajuste_ pode ser expressa por [uma equação](https://en.wikipedia.org/wiki/Simple_linear_regression):
> 
> ```
> Y = a + bX
> ```
>
> `X` é a 'variável explicativa'. `Y` é a 'variável dependente'. A inclinação da linha é `b` e `a` é o intercepto no eixo y, que se refere ao valor de `Y` quando `X = 0`.
>
>![calculate the slope](../../../../translated_images/pt-PT/slope.f3c9d5910ddbfcf9.webp)
>
> Primeiro, calcula a inclinação `b`. Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> Por outras palavras, e referindo-nos à pergunta original dos nossos dados de abóboras: "prever o preço de uma abóbora por alqueire por mês", `X` referir-se-ia ao preço e `Y` ao mês de venda.
>
>![complete the equation](../../../../translated_images/pt-PT/calculation.a209813050a1ddb1.webp)
>
> Calcula o valor de Y. Se estiveres a pagar cerca de 4 dólares, deve ser abril! Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> A matemática que calcula a linha deve demonstrar a inclinação da linha, que também depende do intercepto, ou seja, onde `Y` se encontra quando `X = 0`.
>
> Podes observar o método de cálculo destes valores no site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visita também [este calculador de mínimos quadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver como os valores dos números afetam a linha.

## Correlação

Mais um termo para entender é o **Coeficiente de Correlação** entre variáveis X e Y dadas. Usando um gráfico de dispersão, podes visualizar rapidamente esse coeficiente. Um gráfico com pontos dispersos em linha organizada tem alta correlação, mas um gráfico com pontos dispersos por todo o lado entre X e Y tem baixa correlação.

Um bom modelo de regressão linear será aquele que tem um Coeficiente de Correlação alto (mais perto de 1 do que de 0) usando o método de Regressão dos Mínimos Quadrados com uma linha de regressão.

✅ Executa o notebook que acompanha esta lição e olha para o gráfico de dispersão Mês para Preço. Os dados que associam Mês a Preço para vendas de abóbora parecem ter correlação alta ou baixa, segundo a tua interpretação visual do gráfico de dispersão? Isso muda se usares uma medida mais detalhada em vez de `Mês`, p.ex. *dia do ano* (ou seja, número de dias desde o início do ano)?

No código abaixo, assumiremos que limpámos os dados e obtivemos um dataframe chamado `new_pumpkins`, semelhante ao seguinte:

ID | Mês | DiaDoAno | Variedade | Cidade | Embalagem | Preço Baixo | Preço Alto | Preço
---|-----|----------|-----------|--------|-----------|-------------|------------|--------
70 | 9 | 267 | TIPO PARA TORTA | BALTIMORE | caixas de 1 1/9 alqueire | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | TIPO PARA TORTA | BALTIMORE | caixas de 1 1/9 alqueire | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | TIPO PARA TORTA | BALTIMORE | caixas de 1 1/9 alqueire | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | TIPO PARA TORTA | BALTIMORE | caixas de 1 1/9 alqueire | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | TIPO PARA TORTA | BALTIMORE | caixas de 1 1/9 alqueire | 15.0 | 15.0 | 13.636364

> O código para limpar os dados está disponível em [`notebook.ipynb`](notebook.ipynb). Fizemos os mesmos passos de limpeza da lição anterior, e calculámos a coluna `DayOfYear` usando a seguinte expressão: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Agora que tens uma compreensão da matemática por detrás da regressão linear, vamos criar um modelo de Regressão para ver se conseguimos prever qual embalagem de abóboras terá os melhores preços de abóbora. Alguém a comprar abóboras para uma decoração de abóboras de feriado pode querer essa informação para otimizar as suas compras de embalagens de abóboras para a decoração.

## Procurar Correlação

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Clica na imagem acima para um vídeo curto sobre correlação.

Na lição anterior de certeza que viste que o preço médio para diferentes meses parece assim:

<img alt="Average price by month" src="../../../../translated_images/pt-PT/barchart.a833ea9194346d76.webp" width="50%"/>

Isto sugere que deve haver alguma correlação, e podemos tentar treinar um modelo de regressão linear para prever a relação entre `Mês` e `Preço`, ou entre `DiaDoAno` e `Preço`. Aqui está o gráfico de dispersão que mostra a última relação:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pt-PT/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vamos ver se existe correlação usando a função `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Parece que a correlação é bastante pequena, -0.15 pelo `Mês` e -0.17 pelo `DiaDoAno`, mas pode haver outra relação importante. Parece que existem diferentes aglomerados de preços correspondendo a várias variedades de abóboras. Para confirmar esta hipótese, vamos traçar cada categoria de abóbora usando uma cor diferente. Passando um parâmetro `ax` para a função de plotagem `scatter` podemos colocar todos os pontos no mesmo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pt-PT/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

A nossa investigação sugere que a variedade tem mais efeito sobre o preço global do que a data real de venda. Podemos ver isto com um gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/pt-PT/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Vamos focar por enquanto apenas numa variedade de abóbora, o 'tipo para torta', e ver o efeito que a data tem no preço:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pt-PT/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Se agora calcularmos a correlação entre `Preço` e `DiaDoAno` usando a função `corr`, obteremos algo como `-0.27` - o que significa que treinar um modelo preditivo faz sentido.

> Antes de treinar um modelo de regressão linear, é importante garantir que os nossos dados estão limpos. A regressão linear não funciona bem com valores em falta, por isso faz sentido eliminar todas as células vazias:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Outra abordagem seria preencher esses valores vazios com a média da coluna correspondente.

## Regressão Linear Simples

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Clica na imagem acima para um vídeo curto sobre regressão linear e polinomial.

Para treinar o nosso modelo de Regressão Linear, vamos usar a biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Começamos por separar os valores de entrada (características) e a saída esperada (rótulo) em arrays numpy separados:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Note que tivemos de fazer `reshape` na entrada de dados para que o pacote Linear Regression a entendesse corretamente. A Regressão Linear espera uma matriz 2D como entrada, onde cada linha da matriz corresponde a um vetor de características de entrada. No nosso caso, como temos apenas uma entrada - precisamos de um array com forma N&times;1, onde N é o tamanho do conjunto de dados.

Depois, precisamos de dividir os dados em conjuntos de treino e teste, para que possamos validar o nosso modelo depois do treino:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Por fim, treinar o modelo real de Regressão Linear ocupa apenas duas linhas de código. Definimos o objeto `LinearRegression` e ajustamo-lo aos nossos dados usando o método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

O objeto `LinearRegression` após ser `fit`-ado contém todos os coeficientes da regressão, que podem ser acedidos usando a propriedade `.coef_`. No nosso caso, há apenas um coeficiente, que deverá estar em torno de `-0.017`. Isso significa que os preços parecem diminuir um pouco com o tempo, mas não muito, cerca de 2 cêntimos por dia. Podemos também aceder ao ponto de interseção da regressão com o eixo Y usando `lin_reg.intercept_` - será cerca de `21` no nosso caso, indicando o preço no início do ano.

Para ver o quão preciso é o nosso modelo, podemos prever preços num conjunto de dados de teste e então medir quão próximas as nossas previsões estão dos valores esperados. Isto pode ser feito usando a métrica de erro quadrático médio (MSE), que é a média de todas as diferenças ao quadrado entre o valor esperado e o previsto.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

O nosso erro parece estar em torno de 2 pontos, o que é ~17%. Não é muito bom. Outro indicador da qualidade do modelo é o **coeficiente de determinação**, que pode ser obtido assim:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Se o valor for 0, significa que o modelo não considera os dados de entrada e atua como o *pior preditor linear*, que é simplesmente a média dos resultados. O valor 1 significa que conseguimos predizer perfeitamente todas as saídas esperadas. No nosso caso, o coeficiente está em torno de 0.06, que é bastante baixo.

Podemos também fazer um gráfico dos dados de teste juntamente com a linha de regressão para ver melhor como funciona a regressão no nosso caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pt-PT/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regressão Polinomial

Outro tipo de Regressão Linear é a Regressão Polinomial. Embora por vezes exista uma relação linear entre variáveis - quanto maior a abóbora em volume, maior o preço - às vezes essas relações não podem ser representadas como um plano ou linha reta.

✅ Aqui estão [alguns exemplos adicionais](https://online.stat.psu.edu/stat501/lesson/9/9.8) de dados que poderiam utilizar Regressão Polinomial.

Observe novamente a relação entre Data e Preço. Este gráfico de dispersão parece algo que deveria necessariamente ser analisado por uma linha reta? Os preços não podem flutuar? Neste caso, pode tentar regressão polinomial.

✅ Polinómios são expressões matemáticas que podem consistir em uma ou mais variáveis e coeficientes.

A regressão polinomial cria uma linha curva para se ajustar melhor aos dados não lineares. No nosso caso, se incluirmos uma variável `DayOfYear` ao quadrado nos dados de entrada, deveremos conseguir ajustar os nossos dados com uma curva parabólica, que terá um mínimo num certo ponto ao longo do ano.

O Scikit-learn inclui uma API útil [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para combinar diferentes etapas do processamento de dados. Um **pipeline** é uma cadeia de **estimadores**. No nosso caso, vamos criar um pipeline que primeiramente adiciona características polinomiais ao nosso modelo e depois treina a regressão:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que iremos incluir todos os polinómios de segundo grau a partir dos dados de entrada. No nosso caso, isso significará apenas `DayOfYear`<sup>2</sup>, mas dado dois variáveis de entrada X e Y, isto adicionará X<sup>2</sup>, XY e Y<sup>2</sup>. Podemos também usar polinómios de grau superior, se desejarmos.

Pipelines podem ser usados da mesma forma que o objeto original `LinearRegression`, ou seja, podemos `fit` o pipeline, e depois usar `predict` para obter os resultados das previsões. Aqui está o gráfico mostrando os dados de teste e a curva de aproximação:

<img alt="Polynomial regression" src="../../../../translated_images/pt-PT/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando a Regressão Polinomial, podemos obter um MSE ligeiramente mais baixo e um coeficiente de determinação mais alto, mas não significativamente. Precisamos de considerar outras características!

> Pode ver que os preços mínimos das abóboras ocorrem por volta do Halloween. Como pode explicar isso?

🎃 Parabéns, acabou de criar um modelo que pode ajudar a prever o preço das abóboras para torta. Provavelmente pode repetir o mesmo procedimento para todos os tipos de abóboras, mas isso seria trabalhoso. Vamos agora aprender como considerar a variedade da abóbora no nosso modelo!

## Características Categóricas

No mundo ideal, queremos ser capazes de prever preços para diferentes variedades de abóbora usando o mesmo modelo. No entanto, a coluna `Variety` é algo diferente de colunas como `Month`, porque contém valores não numéricos. Estas colunas são chamadas **categóricas**.

[![ML para iniciantes - Previsões de Características Categóricas com Regressão Linear](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML para iniciantes - Previsões de Características Categóricas com Regressão Linear")

> 🎥 Clique na imagem acima para um vídeo curto sobre o uso de características categóricas.

Aqui pode ver como o preço médio depende da variedade:

<img alt="Average price by variety" src="../../../../translated_images/pt-PT/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para considerar a variedade, primeiro precisamos convertê-la para forma numérica, ou **codificá-la**. Existem várias formas de o fazer:

* Uma simples **codificação numérica** construiria uma tabela das diferentes variedades, e depois substituiria o nome da variedade pelo índice nessa tabela. Esta não é a melhor ideia para regressão linear, porque a regressão linear utiliza o valor numérico real do índice e adiciona-o ao resultado, multiplicando por algum coeficiente. No nosso caso, a relação entre o número do índice e o preço é claramente não linear, mesmo se nos certificarmos que os índices são ordenados de uma forma específica.
* A **codificação one-hot** substitui a coluna `Variety` por 4 colunas diferentes, uma para cada variedade. Cada coluna conterá `1` se a linha correspondente for da variedade dada, e `0` caso contrário. Isto significa que haverá quatro coeficientes na regressão linear, um para cada variedade de abóbora, responsável pelo "preço inicial" (ou melhor, "preço adicional") para essa variedade em particular.

O código abaixo mostra como podemos codificar one-hot uma variedade:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Para treinar a regressão linear utilizando a variedade codificada one-hot como entrada, só precisamos inicializar corretamente os dados `X` e `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

O resto do código é o mesmo que usamos acima para treinar a Regressão Linear. Se tentar, verá que o erro quadrático médio é aproximadamente o mesmo, mas obtemos um coeficiente de determinação muito maior (~77%). Para obter previsões ainda mais precisas, podemos considerar mais características categóricas, bem como variáveis numéricas, como `Month` ou `DayOfYear`. Para obter um único grande array de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aqui também consideramos `City` e o tipo de `Package`, o que nos dá um MSE de 2.84 (10%) e determinação 0.94!

## Juntando tudo

Para fazer o melhor modelo, podemos usar dados combinados (categóricos codificados one-hot + numéricos) do exemplo acima juntamente com a Regressão Polinomial. Aqui está o código completo para sua conveniência:

```python
# preparar dados de treino
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# fazer divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurar e treinar o pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prever resultados para dados de teste
pred = pipeline.predict(X_test)

# calcular MSE e determinação
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Isto deverá dar-nos o melhor coeficiente de determinação de quase 97%, e MSE=2.23 (~8% de erro de previsão).

| Modelo | MSE | Determinação |
|--------|-----|--------------|
| Linear com `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomial com `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Linear com `Variety` | 5.24 (19.7%) | 0.77 |
| Linear com todas as características | 2.84 (10.5%) | 0.94 |
| Polinomial com todas as características | 2.23 (8.25%) | 0.97 |

🏆 Muito bem! Criou quatro modelos de regressão numa só lição, e melhorou a qualidade do modelo para 97%. Na secção final sobre Regressão, irá aprender sobre Regressão Logística para determinar categorias.

---
## 🚀Desafio

Teste várias variáveis diferentes neste notebook para ver como a correlação corresponde à precisão do modelo.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Autónomo

Nesta lição aprendemos sobre Regressão Linear. Existem outros tipos importantes de Regressão. Leia sobre as técnicas Stepwise, Ridge, Lasso e Elasticnet. Um bom curso para estudar e aprender mais é o [curso de Aprendizagem Estatística de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tarefa

[Construa um Modelo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, por favor, tenha em conta que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->