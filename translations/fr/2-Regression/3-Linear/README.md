# Construire un modèle de régression avec Scikit-learn : quatre manières de faire la régression

## Note pour débutants

La régression linéaire est utilisée lorsque nous voulons prédire une **valeur numérique** (par exemple, le prix d'une maison, la température ou les ventes).
Elle fonctionne en trouvant une ligne droite qui représente au mieux la relation entre les caractéristiques d'entrée et la sortie.

Dans cette leçon, nous nous concentrons sur la compréhension du concept avant d'explorer des techniques de régression plus avancées.
![Linear vs polynomial regression infographic](../../../../translated_images/fr/linear-polynomial.5523c7cb6576ccab.webp)
> Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz pré-conférence](https://ff-quizzes.netlify.app/en/ml/)

> ### [Cette leçon est disponible en R !](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduction 

Jusqu'à présent, vous avez exploré ce qu'est la régression avec des données d'exemple issues du jeu de données sur les prix des citrouilles que nous utiliserons tout au long de cette leçon. Vous l'avez aussi visualisée avec Matplotlib.

Vous êtes maintenant prêt à plonger plus profondément dans la régression pour le ML. Alors que la visualisation vous permet de comprendre les données, la véritable puissance de l'apprentissage automatique vient de _l'entraînement des modèles_. Les modèles sont entraînés sur des données historiques pour capturer automatiquement les dépendances des données, et ils vous permettent de prédire les résultats pour de nouvelles données que le modèle n'a jamais vues auparavant.

Dans cette leçon, vous apprendrez davantage sur deux types de régression : la _régression linéaire basique_ et la _régression polynomiale_, ainsi que sur une partie des mathématiques sous-jacentes à ces techniques. Ces modèles nous permettront de prédire les prix des citrouilles en fonction de différentes données d'entrée. 

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo sur la régression linéaire.

> Tout au long de ce cursus, nous supposons des connaissances mathématiques minimales, et cherchons à les rendre accessibles aux étudiants venant d'autres domaines, donc faites attention aux notes, 🧮 encadrés, diagrammes et autres outils pédagogiques pour faciliter la compréhension.

### Prérequis

Vous devriez maintenant être familier avec la structure des données sur les citrouilles que nous examinons. Vous pouvez les trouver préchargées et pré-nettoyées dans le fichier _notebook.ipynb_ de cette leçon. Dans ce fichier, le prix des citrouilles est affiché par boisseau dans un nouveau dataframe. Assurez-vous de pouvoir exécuter ces notebooks dans les kernels de Visual Studio Code.

### Préparation

Pour rappel, vous chargez ces données afin de pouvoir poser des questions à leur sujet.

- Quel est le meilleur moment pour acheter des citrouilles ?
- Quel prix puis-je attendre pour un cas de mini-citrouilles ?
- Dois-je les acheter en paniers d’un demi-boisseau ou par boîte de 1 1/9 boisseau ?
Continuons à explorer ces données.

Dans la leçon précédente, vous avez créé un dataframe Pandas et l'avez rempli avec une partie du jeu de données initial, en standardisant les prix par boisseau. Cependant, cela ne vous a permis de rassembler qu'environ 400 points de données et uniquement pour les mois d'automne.

Jetez un œil aux données que nous avons préchargées dans le notebook accompagnant cette leçon. Les données sont préchargées et un premier nuage de points est tracé pour montrer les données par mois. Peut-être pouvons-nous en apprendre un peu plus sur la nature des données en les nettoyant davantage.

## Une ligne de régression linéaire

Comme vous l'avez appris dans leçon 1, l'objectif d'un exercice de régression linéaire est de pouvoir tracer une ligne pour :

- **Montrer les relations entre variables**. Montrer la relation entre les variables
- **Faire des prédictions**. Faire des prédictions précises sur la position d'un nouveau point de données par rapport à cette ligne.

Il est typique de la **régression des moindres carrés** de tracer ce type de ligne. Le terme "moindres carrés" fait référence au processus de minimisation de l'erreur totale dans notre modèle. Pour chaque point de données, nous mesurons la distance verticale (appelée résidu) entre le point réel et notre ligne de régression.

Nous élevons au carré ces distances pour deux raisons principales :

1. **Importance plus que direction :** Nous voulons traiter une erreur de -5 de la même façon qu'une erreur de +5. L'élévation au carré rend toutes les valeurs positives.

2. **Pénaliser les valeurs aberrantes :** L'élévation au carré donne plus de poids aux erreurs plus grandes, forçant la ligne à rester plus proche des points éloignés.

Nous additionnons ensuite toutes ces valeurs au carré. Notre objectif est de trouver la ligne spécifique où cette somme finale est la plus petite possible — d'où le nom "moindres carrés".

> **🧮 Montrez-moi les mathématiques**  
>  
> Cette ligne, appelée _ligne de meilleur ajustement_, peut être exprimée par [une équation](https://en.wikipedia.org/wiki/Simple_linear_regression) :  
>  
> ```
> Y = a + bX
> ```
>
> `X` est la « variable explicative ». `Y` est la « variable dépendante ». La pente de la ligne est `b` et `a` est l'ordonnée à l'origine, soit la valeur de `Y` lorsque `X = 0`.
>
>![calculate the slope](../../../../translated_images/fr/slope.f3c9d5910ddbfcf9.webp)
>
> Commencez par calculer la pente `b`. Infographie par [Jen Looper](https://twitter.com/jenlooper)
>
> En d'autres termes, et en se référant à la question initiale de notre jeu de données sur la citrouille : « prédire le prix d’une citrouille par boisseau selon le mois », `X` ferait référence au prix et `Y` au mois de vente.
>
>![complete the equation](../../../../translated_images/fr/calculation.a209813050a1ddb1.webp)
>
> Calculez la valeur de Y. Si vous payez environ 4 $, cela doit être en avril ! Infographie par [Jen Looper](https://twitter.com/jenlooper)
>
> La méthode mathématique qui calcule la ligne doit démontrer la pente de la ligne, qui dépend aussi de l'ordonnée à l'origine, ou de la position de `Y` lorsque `X = 0`.
>
> Vous pouvez observer la méthode de calcul de ces valeurs sur le site [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visitez aussi [ce calculateur des moindres carrés](https://www.mathsisfun.com/data/least-squares-calculator.html) pour voir comment les valeurs des nombres impactent la ligne.

## Corrélation

Un terme supplémentaire à comprendre est le **coefficient de corrélation** entre les variables X et Y données. À l’aide d’un nuage de points, vous pouvez rapidement visualiser ce coefficient. Un graphique avec des points de données alignés proprement a une forte corrélation, tandis qu’un graphique avec des points dispersés partout entre X et Y a une faible corrélation.

Un bon modèle de régression linéaire aura un coefficient de corrélation élevé (plus proche de 1 que de 0) utilisant la méthode de régression des moindres carrés avec une ligne de régression.

✅ Exécutez le notebook accompagnant cette leçon et examinez le nuage Month vs Price. Les données associant le mois au prix des citrouilles semblent-elles offrir une corrélation élevée ou faible, selon votre interprétation visuelle du scatterplot ? Est-ce que cela change si vous utilisez une mesure plus fine que `Month`, par ex. *jour de l'année* (nombre de jours depuis le début de l'année) ?

Dans le code ci-dessous, nous supposerons que nous avons nettoyé les données et obtenu un dataframe appelé `new_pumpkins`, similaire au tableau suivant :

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 boisseau cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 boisseau cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 boisseau cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 boisseau cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 boisseau cartons | 15.0 | 15.0 | 13.636364

> Le code pour nettoyer les données est disponible dans [`notebook.ipynb`](notebook.ipynb). Nous avons effectué les mêmes étapes de nettoyage que dans la leçon précédente, et avons calculé la colonne `DayOfYear` avec l'expression suivante : 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Maintenant que vous comprenez les mathématiques derrière la régression linéaire, créons un modèle de régression pour voir si nous pouvons prédire quel emballage de citrouilles aura les meilleurs prix. Quelqu’un achetant des citrouilles pour un patch de citrouilles de vacances pourrait vouloir cette information afin d’optimiser ses achats.

## Recherche de Corrélation

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo sur la corrélation.

D’après la leçon précédente, vous avez probablement vu que le prix moyen pour différents mois ressemble à ceci :

<img alt="Average price by month" src="../../../../translated_images/fr/barchart.a833ea9194346d76.webp" width="50%"/>

Cela suggère qu’il devrait y avoir une certaine corrélation, et nous pouvons essayer d’entraîner un modèle de régression linéaire pour prédire la relation entre `Month` et `Price`, ou entre `DayOfYear` et `Price`. Voici le graphique en nuage de points montrant cette dernière relation :

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/fr/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Voyons s’il y a une corrélation en utilisant la fonction `corr` :

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Il semble que la corrélation soit assez faible, -0.15 pour `Month` et -0.17 pour `DayOfMonth`, mais il pourrait y avoir une autre relation importante. Il semble qu’il existe différents clusters de prix correspondant à différentes variétés de citrouilles. Pour confirmer cette hypothèse, traçons chaque catégorie de citrouille avec une couleur différente. En passant un paramètre `ax` à la fonction `scatter`, nous pouvons afficher tous les points sur le même graphique :

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/fr/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Notre investigation suggère que la variété a plus d’effet sur le prix global que la date de vente réelle. Nous pouvons en voir un graphique en barres :

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/fr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Concentrons-nous pour le moment sur une seule variété de citrouille, le « type tarte » (pie type), et voyons quel effet la date a sur le prix :

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/fr/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Si nous calculons maintenant la corrélation entre `Price` et `DayOfYear` avec la fonction `corr`, nous obtenons quelque chose comme `-0.27` — ce qui signifie que l’entraînement d’un modèle prédictif a du sens.

> Avant d’entraîner un modèle de régression linéaire, il est important de s’assurer que nos données sont propres. La régression linéaire ne fonctionne pas bien avec des valeurs manquantes, il est donc judicieux d’éliminer toutes les cellules vides :

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Une autre approche consisterait à remplir ces valeurs vides avec la moyenne de la colonne correspondante.

## Régression Linéaire Simple

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo d'introduction à la régression linéaire et polynomiale.

Pour entraîner notre modèle de régression linéaire, nous allons utiliser la bibliothèque **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Nous commençons par séparer les valeurs d'entrée (features) et la sortie attendue (label) dans des tableaux numpy distincts :

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Notez que nous avons dû effectuer un `reshape` sur les données d'entrée pour que le package de régression linéaire les comprenne correctement. La régression linéaire attend un tableau 2D en entrée, où chaque ligne du tableau correspond à un vecteur de caractéristiques d'entrée. Dans notre cas, puisque nous n’avons qu’une seule entrée, nous avons besoin d’un tableau avec pour forme N&times;1, où N est la taille du jeu de données.

Ensuite, il faut diviser les données en ensembles d’entraînement et de test, afin de pouvoir valider notre modèle après l'entraînement :

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Enfin, l’entraînement du modèle de régression linéaire réel ne prend que deux lignes de code. Nous définissons l’objet `LinearRegression`, puis l’ajustons à nos données avec la méthode `fit` :

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

L’objet `LinearRegression` après avoir été `fit` contient tous les coefficients de la régression, auxquels on peut accéder via la propriété `.coef_`. Dans notre cas, il n’y a qu’un seul coefficient, qui devrait être autour de `-0.017`. Cela signifie que les prix semblent diminuer un peu avec le temps, mais pas beaucoup, d’environ 2 centimes par jour. On peut également accéder au point d’intersection de la régression avec l’axe Y en utilisant `lin_reg.intercept_` – il sera d’environ `21` dans notre cas, indiquant le prix au début de l’année.

Pour voir à quel point notre modèle est précis, nous pouvons prédire les prix sur un ensemble de test, puis mesurer la proximité de nos prédictions par rapport aux valeurs attendues. Cela peut être fait à l’aide de la métrique de l’erreur quadratique moyenne (MSE), qui est la moyenne de toutes les différences au carré entre la valeur attendue et la valeur prédite.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Notre erreur semble être d’environ 2 points, soit ~17%. Pas très bon. Un autre indicateur de qualité du modèle est le **coefficient de détermination**, qui peut être obtenu ainsi :

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Si la valeur est 0, cela signifie que le modèle ne prend pas en compte les données d’entrée et agit comme le *pire prédicteur linéaire*, qui est simplement une valeur moyenne du résultat. Une valeur de 1 signifie que nous pouvons prédire parfaitement toutes les sorties attendues. Dans notre cas, le coefficient est autour de 0.06, ce qui est assez bas.

Nous pouvons aussi tracer les données de test avec la ligne de régression pour mieux voir comment fonctionne la régression dans notre cas :

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/fr/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Régression Polynômiale

Un autre type de régression linéaire est la régression polynomiale. Parfois, il existe une relation linéaire entre les variables — plus la citrouille est grosse en volume, plus le prix est élevé — mais parfois ces relations ne peuvent pas être représentées par un plan ou une ligne droite.

✅ Voici [quelques autres exemples](https://online.stat.psu.edu/stat501/lesson/9/9.8) de données pouvant être analysées par régression polynomiale.

Regardez de nouveau la relation entre Date et Prix. Ce nuage de points semble-t-il devoir nécessairement être analysé par une ligne droite ? Les prix ne peuvent-ils pas fluctuer ? Dans ce cas, vous pouvez essayer la régression polynomiale.

✅ Les polynômes sont des expressions mathématiques qui peuvent contenir une ou plusieurs variables et coefficients.

La régression polynomiale crée une courbe pour mieux ajuster des données non linéaires. Dans notre cas, si nous incluons une variable au carré `DayOfYear` dans les données d'entrée, nous devrions pouvoir ajuster nos données avec une courbe parabolique, qui aura un minimum à un certain moment de l’année.

Scikit-learn inclut une API [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) utile pour combiner différentes étapes de traitement des données. Un **pipeline** est une chaîne d’**estimateurs**. Dans notre cas, nous allons créer un pipeline qui d’abord ajoute des caractéristiques polynomiales à notre modèle, puis entraîne la régression :

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Utiliser `PolynomialFeatures(2)` signifie que nous inclurons tous les polynômes du second degré issus des données d’entrée. Dans notre cas, cela signifie juste `DayOfYear`<sup>2</sup>, mais avec deux variables d’entrée X et Y, cela ajouterait X<sup>2</sup>, XY et Y<sup>2</sup>. On peut aussi utiliser des polynômes de degré supérieur si on le souhaite.

Les pipelines peuvent être utilisés comme l’objet `LinearRegression` original, c’est-à-dire qu’on peut les `fit`, puis utiliser `predict` pour obtenir les résultats de la prédiction. Voici le graphique montrant les données de test et la courbe d’approximation :

<img alt="Polynomial regression" src="../../../../translated_images/fr/poly-results.ee587348f0f1f60b.webp" width="50%" />

Avec la régression polynomiale, on peut obtenir une MSE un peu plus faible et un coefficient de détermination plus élevé, mais pas de manière significative. Il faut prendre en compte d’autres caractéristiques !

> Vous voyez que les prix minimaux des citrouilles sont observés quelque part autour d’Halloween. Comment pouvez-vous l’expliquer ?

🎃 Félicitations, vous venez de créer un modèle qui peut aider à prédire le prix des citrouilles à tarte. Vous pouvez probablement refaire la même procédure pour tous les types de citrouilles, mais ce serait fastidieux. Apprenons maintenant comment prendre en compte la variété de citrouille dans notre modèle !

## Caractéristiques Catégorielles

Dans un monde idéal, nous voulons pouvoir prédire les prix pour différentes variétés de citrouilles avec le même modèle. Cependant, la colonne `Variety` est quelque peu différente de colonnes comme `Month`, car elle contient des valeurs non numériques. Ces colonnes sont appelées **catégorielles**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo présentant l’utilisation des caractéristiques catégorielles.

Ici vous pouvez voir comment le prix moyen dépend de la variété :

<img alt="Average price by variety" src="../../../../translated_images/fr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Pour prendre la variété en compte, il faut d’abord la convertir en forme numérique, ou **l’encoder**. Plusieurs méthodes existent :

* L’**encodage numérique simple** crée un tableau des différentes variétés, puis remplace le nom de la variété par un indice numérique dans ce tableau. Ce n’est pas la meilleure idée pour la régression linéaire, car cette dernière prend la valeur numérique réelle de l’indice et la multiplie par un coefficient. Dans notre cas, la relation entre le numéro d’indice et le prix est clairement non linéaire, même si on ordonne les indices d’une manière spécifique.
* L’**encodage one-hot** remplace la colonne `Variety` par 4 colonnes différentes, une pour chaque variété. Chaque colonne contiendra `1` si la ligne correspond à cette variété, et `0` sinon. Cela signifie qu’il y aura quatre coefficients dans la régression linéaire, un pour chaque variété de citrouille, responsables du "prix de départ" (ou plutôt du "prix additionnel") pour cette variété particulière.

Le code ci-dessous montre comment faire un encodage one-hot d’une variété :

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

Pour entraîner la régression linéaire avec la variété encodée en one-hot comme entrée, il suffit d’initialiser correctement les données `X` et `y` :

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Le reste du code est le même que celui que nous avons utilisé précédemment pour entraîner la régression linéaire. Si vous essayez, vous verrez que l’erreur quadratique moyenne reste à peu près la même, mais on obtient un coefficient de détermination beaucoup plus élevé (~77%). Pour obtenir des prédictions encore plus précises, on peut prendre en compte davantage de caractéristiques catégorielles, ainsi que des caractéristiques numériques, comme `Month` ou `DayOfYear`. Pour avoir un grand tableau de caractéristiques combinées, on peut utiliser `join` :

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Ici, nous prenons aussi en compte la `City` et le type `Package`, ce qui donne une MSE de 2.84 (10%) et un coefficient de détermination de 0.94 !

## Mettre tout ensemble

Pour faire le meilleur modèle, nous pouvons utiliser les données combinées (catégorielles encodées one-hot + numériques) de l’exemple ci-dessus avec la régression polynomiale. Voici le code complet pour votre commodité :

```python
# configurer les données d'entraînement
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# faire la séparation train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurer et entraîner le pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prédire les résultats pour les données de test
pred = pipeline.predict(X_test)

# calculer la MSE et le coefficient de détermination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Cela devrait nous donner le meilleur coefficient de détermination de presque 97%, et une MSE de 2.23 (~8% d’erreur de prédiction).

| Modèle | MSE | Détermination |
|--------|-----|---------------|
| `DayOfYear` Linéaire | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomiale | 2.73 (17.0%) | 0.08 |
| `Variety` Linéaire | 5.24 (19.7%) | 0.77 |
| Toutes caractéristiques Linéaire | 2.84 (10.5%) | 0.94 |
| Toutes caractéristiques Polynomiale | 2.23 (8.25%) | 0.97 |

🏆 Bravo ! Vous avez créé quatre modèles de régression en une leçon, et amélioré la qualité du modèle à 97%. Dans la section finale sur la régression, vous apprendrez la régression logistique pour déterminer des catégories.

---
## 🚀 Défi

Testez plusieurs variables différentes dans ce carnet pour voir comment la corrélation correspond à la précision du modèle.

## [Quiz post-cours](https://ff-quizzes.netlify.app/en/ml/)

## Révision & Auto-apprentissage

Dans cette leçon, nous avons appris la régression linéaire. Il existe d’autres types importants de régression. Lisez sur les techniques Stepwise, Ridge, Lasso et Elasticnet. Un bon cours à suivre pour en apprendre davantage est le [cours d’apprentissage statistique de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Devoir

[Construisez un modèle](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l’aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d’assurer l’exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des imprécisions. Le document original dans sa langue d’origine doit être considéré comme la source faisant autorité. Pour les informations critiques, une traduction professionnelle réalisée par un humain est recommandée. Nous déclinons toute responsabilité en cas de malentendus ou de mauvaises interprétations résultant de l’utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->