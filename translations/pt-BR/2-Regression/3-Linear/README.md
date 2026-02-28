# Construir um modelo de regressão usando Scikit-learn: regressão de quatro maneiras

## Nota para iniciantes

A regressão linear é usada quando queremos prever um **valor numérico** (por exemplo, preço da casa, temperatura ou vendas).
Ela funciona encontrando uma linha reta que melhor representa a relação entre as características de entrada e a saída.

Nesta lição, focamos em entender o conceito antes de explorar técnicas mais avançadas de regressão.
![Infográfico regressão linear vs polinomial](../../../../translated_images/pt-BR/linear-polynomial.5523c7cb6576ccab.webp)
> Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introdução

Até agora, você explorou o que é regressão com dados de exemplo coletados do conjunto de dados de preços de abóboras que usaremos ao longo desta lição. Você também os visualizou usando Matplotlib.

Agora você está pronto para aprofundar em regressão para ML. Enquanto a visualização permite entender os dados, o verdadeiro poder do Machine Learning vem do _treinamento de modelos_. Modelos são treinados com dados históricos para capturar automaticamente dependências dos dados, e eles permitem prever resultados para novos dados, que o modelo não viu antes.

Nesta lição, você aprenderá mais sobre dois tipos de regressão: _regressão linear básica_ e _regressão polinomial_, junto com um pouco da matemática que fundamenta essas técnicas. Esses modelos nos permitirão prever preços de abóboras dependendo de diferentes dados de entrada.

[![ML para iniciantes - Entendendo Regressão Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para iniciantes - Entendendo Regressão Linear")

> 🎥 Clique na imagem acima para um breve vídeo com visão geral da regressão linear.

> Ao longo deste currículo, assumimos conhecimento mínimo de matemática e buscamos torná-lo acessível para estudantes de outras áreas, então fique atento às notas, 🧮 chamadas, diagramas e outras ferramentas de aprendizagem para ajudar na compreensão.

### Pré-requisitos

Você já deve estar familiarizado com a estrutura dos dados de abóboras que estamos examinando. Você pode encontrá-los pré-carregados e pré-limpados no arquivo _notebook.ipynb_ desta lição. No arquivo, o preço da abóbora é exibido por alqueire em um novo dataframe. Certifique-se de conseguir executar esses notebooks em kernels no Visual Studio Code.

### Preparação

Como lembrete, você está carregando esses dados para poder fazer perguntas sobre eles.

- Qual é o melhor momento para comprar abóboras?
- Qual preço posso esperar de uma caixa de abóboras miniatura?
- Devo comprá-las em cestas de meio alqueire ou por caixas de 1 1/9 alqueires?
Vamos continuar explorando esses dados.

Na lição anterior, você criou um dataframe Pandas e o preencheu com parte do conjunto de dados original, padronizando o preço por alqueire. Fazendo isso, porém, você conseguiu reunir cerca de 400 pontos de dados e apenas para os meses de outono.

Dê uma olhada nos dados que pré-carregamos no notebook acompanhado desta lição. Os dados estão pré-carregados e um gráfico de dispersão inicial é traçado para mostrar os dados de meses. Talvez possamos obter um pouco mais de detalhes sobre a natureza dos dados limpando-os mais.

## Uma linha de regressão linear

Como você aprendeu na Lição 1, o objetivo de um exercício de regressão linear é poder traçar uma linha para:

- **Mostrar relações entre variáveis**. Mostrar a relação entre variáveis
- **Fazer previsões**. Fazer previsões precisas sobre onde um novo ponto de dados cairia em relação a essa linha.

É típico da **Regressão dos Mínimos Quadrados** desenhar esse tipo de linha. O termo "Mínimos Quadrados" refere-se ao processo de minimizar o erro total em nosso modelo. Para cada ponto de dados, medimos a distância vertical (chamada resíduo) entre o ponto real e nossa linha de regressão.

Elevamos essas distâncias ao quadrado por duas razões principais:

1. **Magnitude sobre direção:** Queremos tratar um erro de -5 da mesma forma que um erro de +5. Quadrar torna todos os valores positivos.

2. **Penalização de outliers:** Quadrar atribui mais peso a erros maiores, forçando a linha a ficar mais próxima dos pontos que estão longe.

Depois somamos todos esses valores quadrados. Nosso objetivo é encontrar a linha específica onde essa soma final é a menor possível — daí o nome "Mínimos Quadrados".

> **🧮 Mostre a matemática** 
> 
> Essa linha, chamada _linha de melhor ajuste_, pode ser expressa por [uma equação](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` é a 'variável explicativa'. `Y` é a 'variável dependente'. A inclinação da linha é `b` e `a` é a interceptação no eixo y, que se refere ao valor de `Y` quando `X = 0`. 
>
>![calcular a inclinação](../../../../translated_images/pt-BR/slope.f3c9d5910ddbfcf9.webp)
>
> Primeiro, calcule a inclinação `b`. Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> Em outras palavras, e referindo-se à pergunta original dos dados das abóboras: "prever o preço de uma abóbora por alqueire por mês", `X` se referiria ao preço e `Y` ao mês de venda.
>
>![completar a equação](../../../../translated_images/pt-BR/calculation.a209813050a1ddb1.webp)
>
> Calcule o valor de Y. Se você está pagando cerca de $4, deve ser abril! Infográfico por [Jen Looper](https://twitter.com/jenlooper)
>
> A matemática que calcula a linha deve demonstrar a inclinação da linha, que também depende da interceptação, ou de onde `Y` está situado quando `X = 0`.
>
> Você pode observar o método de cálculo desses valores no site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visite também [este calculador de mínimos quadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver como os valores numéricos impactam a linha.

## Correlação

Mais um termo para entender é o **Coeficiente de Correlação** entre as variáveis X e Y dadas. Usando um gráfico de dispersão, você pode rapidamente visualizar esse coeficiente. Um gráfico com pontos de dados alinhados em uma linha bem definida tem alta correlação, mas um gráfico com pontos de dados espalhados entre X e Y tem baixa correlação.

Um bom modelo de regressão linear será aquele com um alto Coeficiente de Correlação (mais próximo de 1 do que de 0) usando o método de Regressão dos Mínimos Quadrados com uma linha de regressão.

✅ Execute o notebook que acompanha esta lição e observe o gráfico de dispersão de Mês para Preço. Os dados associando Mês a Preço para vendas de abóbora parecem ter alta ou baixa correlação, segundo sua interpretação visual do gráfico? Isso muda se você usar uma medida mais detalhada em vez de `Mês`, por exemplo, *dia do ano* (ou seja, número de dias desde o início do ano)?

No código abaixo, assumiremos que limpamos os dados e obtivemos um dataframe chamado `new_pumpkins`, semelhante ao seguinte:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> O código para limpar os dados está disponível em [`notebook.ipynb`](notebook.ipynb). Fizemos os mesmos passos de limpeza da lição anterior e calculamos a coluna `DayOfYear` usando a seguinte expressão:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Agora que você entende a matemática por trás da regressão linear, vamos criar um modelo de Regressão para ver se conseguimos prever qual embalagem de abóboras terá os melhores preços. Alguém comprando abóboras para uma horta para o feriado pode querer essa informação para otimizar suas compras de embalagens de abóbora para a horta.

## Procurando Correlação

[![ML para iniciantes - Procurando Correlação: A Chave para Regressão Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para iniciantes - Procurando Correlação: A Chave para Regressão Linear")

> 🎥 Clique na imagem acima para um breve vídeo com visão geral de correlação.

Na lição anterior, você provavelmente viu que o preço médio para diferentes meses se parece com isto:

<img alt="Preço médio por mês" src="../../../../translated_images/pt-BR/barchart.a833ea9194346d76.webp" width="50%"/>

Isso sugere que deveria haver alguma correlação, e podemos tentar treinar um modelo de regressão linear para prever a relação entre `Month` e `Price`, ou entre `DayOfYear` e `Price`. Aqui está o gráfico de dispersão que mostra essa última relação:

<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vamos ver se há correlação usando a função `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Parece que a correlação é bem pequena, -0,15 pelo `Month` e -0,17 pelo `DayOfMonth`, mas pode haver outra relação importante. Parece que existem diferentes grupos de preços correspondendo a diferentes variedades de abóboras. Para confirmar essa hipótese, vamos traçar cada categoria de abóbora usando uma cor diferente. Passando um parâmetro `ax` para a função `scatter` podemos traçar todos os pontos no mesmo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Nossa investigação sugere que a variedade tem mais efeito no preço geral do que a própria data de venda. Podemos ver isso com um gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Gráfico de barras de preço por variedade" src="../../../../translated_images/pt-BR/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Vamos focar por enquanto apenas em uma variedade de abóbora, a 'pie type', e ver qual efeito a data tem no preço:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Gráfico de dispersão de Preço vs. Dia do Ano" src="../../../../translated_images/pt-BR/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Se agora calcularmos a correlação entre `Price` e `DayOfYear` usando a função `corr`, obteremos algo como `-0.27` — o que significa que faz sentido treinar um modelo preditivo.

> Antes de treinar um modelo de regressão linear, é importante garantir que nossos dados estejam limpos. A regressão linear não funciona bem com valores ausentes, portanto faz sentido eliminar todas as células vazias:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Outra abordagem seria preencher esses valores vazios com a média da respectiva coluna.

## Regressão Linear Simples

[![ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para iniciantes - Regressão Linear e Polinomial usando Scikit-learn")

> 🎥 Clique na imagem acima para um breve vídeo com visão geral de regressão linear e polinomial.

Para treinar nosso modelo de Regressão Linear, usaremos a biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Começamos separando os valores de entrada (features) e a saída esperada (rótulo) em arrays numpy separados:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Observe que tivemos que realizar um `reshape` nos dados de entrada para que o pacote de Regressão Linear os entendesse corretamente. Regressão Linear espera um array 2D como entrada, onde cada linha corresponde a um vetor de características de entrada. No nosso caso, como temos apenas uma entrada, precisamos de um array com forma N&times;1, onde N é o tamanho do conjunto de dados.

Em seguida, precisamos dividir os dados em conjuntos de treino e teste, para que possamos validar nosso modelo após o treinamento:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finalmente, o treinamento do modelo de Regressão Linear em si leva apenas duas linhas de código. Definimos o objeto `LinearRegression`, e o ajustamos aos nossos dados usando o método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

O objeto `LinearRegression` após o `fit` contém todos os coeficientes da regressão, que podem ser acessados usando a propriedade `.coef_`. No nosso caso, há apenas um coeficiente, que deve estar em torno de `-0.017`. Isso significa que os preços parecem cair um pouco com o tempo, mas não muito, cerca de 2 centavos por dia. Também podemos acessar o ponto de interseção da regressão com o eixo Y usando `lin_reg.intercept_` - que será em torno de `21` no nosso caso, indicando o preço no início do ano.

Para ver quão preciso nosso modelo é, podemos prever os preços em um conjunto de dados de teste, e então medir o quão próximas estão nossas previsões dos valores esperados. Isso pode ser feito usando a métrica de erro médio quadrático (MSE), que é a média de todas as diferenças quadráticas entre o valor esperado e o valor previsto.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Nosso erro parece estar em torno de 2 pontos, o que é ~17%. Nada muito bom. Outro indicador da qualidade do modelo é o **coeficiente de determinação**, que pode ser obtido assim:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Se o valor for 0, significa que o modelo não leva em conta os dados de entrada, e age como o *pior preditor linear*, que é simplesmente o valor médio do resultado. O valor 1 significa que podemos prever perfeitamente todas as saídas esperadas. No nosso caso, o coeficiente está em torno de 0.06, o que é bem baixo.

Também podemos plotar os dados de teste junto com a linha da regressão para ver melhor como a regressão funciona no nosso caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pt-BR/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regressão Polinomial

Outro tipo de Regressão Linear é a Regressão Polinomial. Embora às vezes haja uma relação linear entre as variáveis - quanto maior a abóbora em volume, maior o preço - às vezes essas relações não podem ser representadas por um plano ou uma linha reta.

✅ Aqui estão [mais alguns exemplos](https://online.stat.psu.edu/stat501/lesson/9/9.8) de dados que poderiam usar Regressão Polinomial

Dê outra olhada na relação entre Data e Preço. Esse gráfico de dispersão parece que necessariamente deveria ser analisado por uma linha reta? Os preços não podem flutuar? Nesse caso, você pode tentar regressão polinomial.

✅ Polinômios são expressões matemáticas que podem consistir de uma ou mais variáveis e coeficientes

A regressão polinomial cria uma linha curva para ajustar melhor dados não lineares. No nosso caso, se incluirmos uma variável `DayOfYear` ao quadrado nos dados de entrada, poderemos ajustar nossos dados com uma curva parabólica, que terá um mínimo em um certo ponto dentro do ano.

O Scikit-learn inclui uma útil [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para combinar diferentes etapas do processamento de dados. Um **pipeline** é uma cadeia de **estimadores**. No nosso caso, vamos criar um pipeline que primeiro adiciona características polinomiais ao nosso modelo, e depois treina a regressão:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que incluiremos todos os polinômios de segundo grau dos dados de entrada. No nosso caso, isso significará apenas `DayOfYear`<sup>2</sup>, mas dado duas variáveis de entrada X e Y, isso adicionará X<sup>2</sup>, XY e Y<sup>2</sup>. Também podemos usar polinômios de grau mais elevado se quisermos.

Pipelines podem ser usados da mesma forma que o objeto original `LinearRegression`, ou seja, podemos `fit` o pipeline, e depois usar `predict` para obter os resultados da previsão. Aqui está o gráfico mostrando os dados de teste e a curva de aproximação:

<img alt="Polynomial regression" src="../../../../translated_images/pt-BR/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando Regressão Polinomial, podemos obter um MSE ligeiramente menor e coeficiente de determinação maior, mas não significativamente. Precisamos levar em conta outras características!

> Você pode ver que os preços mínimos das abóboras são observados mais ou menos perto do Halloween. Como você pode explicar isso? 

🎃 Parabéns, você acabou de criar um modelo que pode ajudar a prever o preço das abóboras para torta. Provavelmente, você pode repetir o mesmo procedimento para todos os tipos de abóbora, mas isso seria tedioso. Vamos aprender agora como levar a variedade de abóbora em consideração no nosso modelo!

## Características Categóricas

No mundo ideal, queremos ser capazes de prever preços para diferentes variedades de abóbora usando o mesmo modelo. Contudo, a coluna `Variety` é um pouco diferente de colunas como `Month`, porque contém valores não numéricos. Essas colunas são chamadas de **categóricas**.

[![ML para iniciantes - Predições com regressão linear para características categóricas](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML para iniciantes - Predições com regressão linear para características categóricas")

> 🎥 Clique na imagem acima para um vídeo curto sobre o uso de características categóricas.

Aqui você pode ver como o preço médio depende da variedade:

<img alt="Average price by variety" src="../../../../translated_images/pt-BR/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para levar a variedade em consideração, primeiro precisamos convertê-la para forma numérica, ou **codificá-la**. Existem várias formas de fazer isso:

* A simples **codificação numérica** construirá uma tabela das diferentes variedades, e depois substituirá o nome da variedade por um índice nessa tabela. Isso não é a melhor ideia para regressão linear, porque a regressão linear considera o valor numérico real do índice, e o adiciona ao resultado, multiplicando por algum coeficiente. No nosso caso, a relação entre o número do índice e o preço é claramente não linear, mesmo que nos certifiquemos que os índices estão ordenados de algum jeito específico.
* A **codificação one-hot** substituirá a coluna `Variety` por 4 colunas diferentes, uma para cada variedade. Cada coluna conterá `1` se a linha correspondente corresponder àquela variedade, e `0` caso contrário. Isso significa que haverá quatro coeficientes na regressão linear, um para cada variedade de abóbora, responsáveis pelo "preço inicial" (ou melhor, "preço adicional") para aquela variedade em particular.

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

Para treinar a regressão linear usando a variedade codificada one-hot como entrada, só precisamos inicializar os dados `X` e `y` corretamente:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

O restante do código é o mesmo que usamos acima para treinar Regressão Linear. Se você tentar, verá que o erro médio quadrático é mais ou menos o mesmo, mas obtemos um coeficiente de determinação muito maior (~77%). Para obter previsões ainda mais precisas, podemos levar mais características categóricas em consideração, assim como características numéricas, como `Month` ou `DayOfYear`. Para obter um grande array de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aqui também levamos em conta `City` e tipo de `Package`, o que nos dá MSE de 2.84 (10%), e determinação 0.94!

## Juntando tudo

Para fazer o melhor modelo, podemos usar dados combinados (categóricos codificados one-hot + numéricos) do exemplo acima junto com a Regressão Polinomial. Aqui está o código completo para sua conveniência:

```python
# preparar dados de treinamento
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

# prever resultados para os dados de teste
pred = pipeline.predict(X_test)

# calcular MSE e determinação
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Isso deve nos dar o melhor coeficiente de determinação de quase 97%, e MSE=2.23 (~8% de erro de previsão).

| Modelo | MSE | Determinação |
|-------|-----|---------------|
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polinomial | 2.73 (17.0%) | 0.08 |
| `Variety` Linear | 5.24 (19.7%) | 0.77 |
| Todas as características Linear | 2.84 (10.5%) | 0.94 |
| Todas as características Polinomial | 2.23 (8.25%) | 0.97 |

🏆 Muito bem! Você criou quatro modelos de Regressão em uma lição, e melhorou a qualidade do modelo para 97%. Na seção final sobre Regressão, você aprenderá sobre Regressão Logística para determinar categorias. 

---
## 🚀Desafio

Teste várias variáveis diferentes neste notebook para ver como a correlação corresponde à precisão do modelo.

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Nesta lição aprendemos sobre Regressão Linear. Existem outros tipos importantes de Regressão. Leia sobre as técnicas Stepwise, Ridge, Lasso e Elasticnet. Um bom curso para estudar e aprender mais é o [curso de Aprendizado Estatístico da Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tarefa

[Construa um Modelo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer equívocos ou interpretações erradas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->