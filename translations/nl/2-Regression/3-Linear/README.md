# Bouw een regressiemodel met Scikit-learn: regressie op vier manieren

## Beginnersnotitie

Lineaire regressie wordt gebruikt wanneer we een **numerieke waarde** willen voorspellen (bijvoorbeeld de prijs van een huis, temperatuur of verkoopcijfers).  
Het werkt door een rechte lijn te vinden die het beste de relatie tussen invoerkenmerken en de uitvoer weergeeft.

In deze les richten we ons op het begrijpen van het concept voordat we meer geavanceerde regressietechnieken verkennen.  
![Linear vs polynomial regression infographic](../../../../translated_images/nl/linear-polynomial.5523c7cb6576ccab.webp)  
> Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Voorcollege quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is ook beschikbaar in R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Inleiding

Tot nu toe heb je verkend wat regressie is met voorbeeldgegevens verzameld uit de dataset van pompoenprijzen die we in deze les zullen gebruiken. Je hebt dit ook gevisualiseerd met Matplotlib.

Nu ben je klaar om dieper in regressie voor ML te duiken. Visualisatie helpt je om data te begrijpen, maar de echte kracht van Machine Learning komt van het _trainen van modellen_. Modellen worden getraind op historische data om automatisch afhankelijkheden in de data vast te leggen en stellen je in staat uitkomsten te voorspellen voor nieuwe data die het model nog niet eerder heeft gezien.

In deze les leer je meer over twee soorten regressie: _basis lineaire regressie_ en _polynomiale regressie_, samen met wat van de wiskunde achter deze technieken. Deze modellen stellen ons in staat om pompoenprijzen te voorspellen op basis van verschillende invoerdata.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van lineaire regressie.

> Gedurende dit curriculum gaan we uit van minimale wiskundige voorkennis, en streven we ernaar het toegankelijk te maken voor studenten uit andere disciplines. Let daarom op notities, 🧮 uitlegpaaltjes, diagrammen en andere leermiddelen om het begrip te ondersteunen.

### Vereisten

Je zou nu bekend moeten zijn met de structuur van de pompoen data die we onderzoeken. Je kunt deze vooraf geladen en voorbewerkt vinden in het bestand _notebook.ipynb_ van deze les. In het bestand wordt de pompoenprijs per bushel weergegeven in een nieuw DataFrame. Zorg ervoor dat je deze notebooks kunt draaien in kernels in Visual Studio Code.

### Voorbereiding

Even ter herinnering, je laadt deze data om er vragen over te kunnen stellen.

- Wanneer is het beste moment om pompoenen te kopen?  
- Welke prijs kan ik verwachten van een doos mini-pompoenen?  
- Moet ik ze kopen in halve bushel manden of per 1 1/9 bushel doos?  
Laten we deze data verder onderzoeken.

In de vorige les maakte je een Pandas DataFrame en vulde je die met een deel van de originele dataset, waarbij je de prijzen standaardiseerde per bushel. Daarmee kon je echter slechts ongeveer 400 datapunten ophalen en alleen voor de herfstmaanden.

Bekijk de data die we vooraf geladen hebben in het bijbehorende notebook van deze les. De gegevens zijn vooraf geladen en er is een eerste scatterplot gemaakt om maanddata te tonen. Misschien kunnen we meer details over de aard van de data krijgen door deze verder te schonen.

## Een lineaire regressielijn

Zoals je in les 1 hebt geleerd, is het doel van een lineaire regressie oefening het kunnen plotten van een lijn om:

- **Relaties tussen variabelen te tonen**. Toon de relatie tussen variabelen  
- **Voorspellingen te maken**. Maak accurate voorspellingen waar een nieuw datapunt zou vallen ten opzichte van die lijn.

Typisch voor de **Least-Squares Regression** is om dit soort lijn te tekenen. De term "Least-Squares" verwijst naar het proces van het minimaliseren van de totale fout in ons model. Voor elk datapunt meten we de verticale afstand (residueel genoemd) tussen het daadwerkelijke punt en onze regressielijn.

We kwadrateren deze afstanden om twee belangrijke redenen:  

1. **Grootte boven richting:** We willen een fout van -5 hetzelfde behandelen als een fout van +5. Kwadrateren zorgt ervoor dat alle waarden positief worden.

2. **Straffen van uitschieters:** Kwadrateren geeft meer gewicht aan grotere fouten, waardoor de lijn dichter bij veraf liggende punten blijft.

We tellen vervolgens al deze gekwadrateerde waarden op. Ons doel is om die specifieke lijn te vinden waarbij deze som het kleinst is (de kleinste mogelijke waarde)—vandaar de naam "Least-Squares".

> **🧮 Laat me de wiskunde zien**  
>  
> Deze lijn, genaamd de _line of best fit_, kan worden uitgedrukt door [een vergelijking](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>  
> `X` is de 'verklarende variabele'. `Y` is de 'afhankelijke variabele'. De helling van de lijn is `b` en `a` is het snijpunt met de y-as, wat verwijst naar de waarde van `Y` wanneer `X = 0`.  
>  
>![bereken de helling](../../../../translated_images/nl/slope.f3c9d5910ddbfcf9.webp)  
>  
> Bereken eerst de helling `b`. Infographic door [Jen Looper](https://twitter.com/jenlooper)  
>  
> Met andere woorden, en verwijzend naar onze oorspronkelijke pompoendata-vraag: "voorspel de prijs van een pompoen per bushel per maand", zou `X` verwijzen naar de prijs en `Y` naar de verkoopmaand.  
>  
>![voltooi de vergelijking](../../../../translated_images/nl/calculation.a209813050a1ddb1.webp)  
>  
> Bereken de waarde van Y. Als je rond de $4 betaalt, moet het april zijn! Infographic door [Jen Looper](https://twitter.com/jenlooper)  
>  
> De wiskunde die de lijn berekent moet de helling van de lijn tonen, die ook afhankelijk is van het intercept, oftewel waar `Y` ligt als `X = 0`.  
>  
> Je kunt de methode voor het berekenen van deze waarden zien op de website [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Bezoek ook [deze Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) om te zien hoe de waardes van getallen de lijn beïnvloeden.

## Correlatie

Nog een term om te begrijpen is de **Correlatiecoëfficiënt** tussen gegeven X- en Y-variabelen. Met een scatterplot kun je deze coëfficiënt snel visualiseren. Een plot met datapunten netjes op een lijn wijzen op hoge correlatie, terwijl een plot waar punten overal verspreid liggen tussen X en Y op lage correlatie wijst.

Een goed lineair regressiemodel is er een met een hoge Correlatiecoëfficiënt (dichter bij 1 dan bij 0) met behulp van de Least-Squares Regression methode met een regressielijn.

✅ Draai het notebook bij deze les en bekijk de scatterplot van Maand tegen Prijs. Lijkt de data die Maand aan Prijs koppelt voor pompoenverkopen een hoge of lage correlatie te hebben, volgens jouw visuele interpretatie van de scatterplot? Verandert dat als je een fijnmaziger maat gebruikt in plaats van `Maand`, bijvoorbeeld *dag van het jaar* (d.w.z. aantal dagen sinds begin van het jaar)?

In onderstaande code gaan we ervan uit dat we de data hebben opgeschoond, en een DataFrame `new_pumpkins` hebben verkregen, vergelijkbaar met het volgende:

ID | Maand | DagVanHetJaar | Variëteit | Stad | Verpakking | Lage Prijs | Hoge Prijs | Prijs  
---|-------|---------------|-----------|------|------------|------------|------------|-------  
70 | 9     | 267           | PIE TYPE  | BALTIMORE | 1 1/9 bushel kartonnen dozen | 15.0 | 15.0 | 13.636364  
71 | 9     | 267           | PIE TYPE  | BALTIMORE | 1 1/9 bushel kartonnen dozen | 18.0 | 18.0 | 16.363636  
72 | 10    | 274           | PIE TYPE  | BALTIMORE | 1 1/9 bushel kartonnen dozen | 18.0 | 18.0 | 16.363636  
73 | 10    | 274           | PIE TYPE  | BALTIMORE | 1 1/9 bushel kartonnen dozen | 17.0 | 17.0 | 15.454545  
74 | 10    | 281           | PIE TYPE  | BALTIMORE | 1 1/9 bushel kartonnen dozen | 15.0 | 15.0 | 13.636364

> De code om de data te schonen is beschikbaar in [`notebook.ipynb`](notebook.ipynb). We hebben dezelfde schoonmaakstappen uitgevoerd als in de vorige les, en hebben de kolom `DagVanHetJaar` berekend met de volgende expressie:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Nu je inzicht hebt in de wiskunde achter lineaire regressie, laten we een regressiemodel maken om te zien of we kunnen voorspellen welke verpakking van pompoenen de beste pompoenprijzen zal hebben. Iemand die pompoenen koopt voor een pompoenveld voor de feestdagen wil deze informatie mogelijk gebruiken om hun aankopen van pompoenverpakkingen voor het veld te optimaliseren.

## Op zoek naar correlatie

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht over correlatie.

In de vorige les heb je waarschijnlijk gezien dat de gemiddelde prijs voor verschillende maanden er zo uitziet:

<img alt="Gemiddelde prijs per maand" src="../../../../translated_images/nl/barchart.a833ea9194346d76.webp" width="50%"/>

Dit suggereert dat er enige correlatie zou moeten zijn, en we kunnen proberen een lineair regressiemodel te trainen om de relatie tussen `Maand` en `Prijs` te voorspellen, of tussen `DagVanHetJaar` en `Prijs`. Hier is de scatterplot die de laatste relatie toont:

<img alt="Scatterplot van Prijs versus Dag van het Jaar" src="../../../../translated_images/nl/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Laten we kijken of er een correlatie is met de `corr` functie:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Het lijkt erop dat de correlatie vrij klein is, -0.15 op basis van `Maand` en -0.17 op basis van `DagVanHetJaar`, maar er zou een andere belangrijke relatie kunnen zijn. Het lijkt erop dat er verschillende clusters van prijzen zijn die overeenkomen met verschillende pompoenvariëteiten. Om deze hypothese te bevestigen, laten we elke pompoencategorie met een andere kleur plotten. Door een `ax` parameter door te geven aan de `scatter` plotfunctie kunnen we alle punten op dezelfde grafiek weergeven:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Scatterplot van Prijs versus Dag van het Jaar" src="../../../../translated_images/nl/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Ons onderzoek suggereert dat variëteit meer effect heeft op de totale prijs dan de daadwerkelijke verkoopdatum. Dit kunnen we zien met een staafgrafiek:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Staafgrafiek van prijs versus variëteit" src="../../../../translated_images/nl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Laten we ons voor nu alleen richten op één pompoenvariëteit, het 'pie type', en kijken wat voor effect de datum heeft op de prijs:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
  
<img alt="Scatterplot van Prijs versus Dag van het Jaar" src="../../../../translated_images/nl/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />  

Als we nu de correlatie berekenen tussen `Prijs` en `DagVanHetJaar` met de `corr` functie, krijgen we iets als `-0.27` - wat betekent dat het trainen van een voorspellend model zinvol is.

> Voordat je een lineair regressiemodel traint, is het belangrijk om ervoor te zorgen dat onze data schoon is. Lineaire regressie werkt niet goed met ontbrekende waarden, dus het is logisch om alle lege cellen kwijt te raken:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
Een andere aanpak is om die lege waarden op te vullen met gemiddelde waardes uit de corresponderende kolom.

## Eenvoudige lineaire regressie

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van lineaire en polynomiale regressie.

Om ons Lineaire Regressiemodel te trainen, gebruiken we de **Scikit-learn** bibliotheek.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
We beginnen met het scheiden van invoerwaarden (kenmerken) en de verwachte uitvoer (label) in aparte numpy arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Let op dat we `reshape` op de invoerdata moesten uitvoeren zodat het Linear Regression-pakket het correct begrijpt. Lineaire Regressie verwacht een 2D-array als invoer, waarbij elke rij van de array overeenkomt met een vector van invoerkenmerken. In ons geval, omdat we maar één invoer hebben, hebben we een array nodig met vorm N&times;1, waarbij N de datasetgrootte is.

Vervolgens moeten we de data splitsen in train- en testdatasets, zodat we ons model na het trainen kunnen valideren:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
Tot slot neemt het trainen van het daadwerkelijke Lineaire Regressiemodel slechts twee regels code in beslag. We definiëren het `LinearRegression` object, en passen het toe op onze data met de `fit` methode:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Het `LinearRegression`-object na het `fit`-ten bevat alle coëfficiënten van de regressie, die toegankelijk zijn via de `.coef_`-eigenschap. In ons geval is er maar één coëfficiënt, die ongeveer `-0.017` zou moeten zijn. Dit betekent dat de prijzen iets lijken te dalen in de loop van de tijd, maar niet te veel, ongeveer 2 cent per dag. We kunnen ook het snijpunt van de regressie met de Y-as benaderen met `lin_reg.intercept_` - dit zal in ons geval rond `21` liggen, wat de prijs aan het begin van het jaar aangeeft.

Om te zien hoe nauwkeurig ons model is, kunnen we prijzen voorspellen op een testdataset en vervolgens meten hoe dicht onze voorspellingen bij de verwachte waarden liggen. Dit kan worden gedaan met behulp van de mean square error (MSE) metriek, wat het gemiddelde is van alle gekwadrateerde verschillen tussen verwachte en voorspelde waarde.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Onze fout lijkt ongeveer 2 punten te zijn, wat ~17% is. Niet zo goed. Een andere indicator van modelkwaliteit is de **coëfficiënt van determinatie**, die op deze manier verkregen kan worden:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Als de waarde 0 is, betekent dit dat het model de invoergegevens niet meeneemt en fungeert als de *slechtste lineaire voorspeller*, wat gewoon de gemiddelde waarde van het resultaat is. De waarde 1 betekent dat we perfect alle verwachte uitkomsten kunnen voorspellen. In ons geval is de coëfficiënt ongeveer 0.06, wat vrij laag is.

We kunnen ook de testgegevens plotten samen met de regressielijn om beter te zien hoe regressie in ons geval werkt:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/nl/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomiale regressie

Een ander type lineaire regressie is polynomiale regressie. Soms is er een lineair verband tussen variabelen - hoe groter de pompoen qua volume, hoe hoger de prijs - maar soms kunnen deze relaties niet worden weergegeven als een vlak of rechte lijn.

✅ Hier zijn [nog meer voorbeelden](https://online.stat.psu.edu/stat501/lesson/9/9.8) van gegevens die baat zouden hebben bij polynomiale regressie

Kijk nog eens goed naar het verband tussen Datum en Prijs. Lijkt deze spreidingsgrafiek per se door een rechte lijn te moeten worden geanalyseerd? Kunnen prijzen niet schommelen? In dat geval kun je polynomiale regressie proberen.

✅ Polynomiale uitdrukkingen zijn wiskundige uitdrukkingen die uit een of meer variabelen en coëfficiënten kunnen bestaan

Polynomiale regressie creëert een gebogen lijn om niet-lineaire data beter te passen. In ons geval, als we een kwadratische `DayOfYear` variabele toevoegen aan de invoergegevens, zouden we onze data moeten kunnen passen met een parabolische kromme, die een minimumpunt heeft op een bepaald punt binnen het jaar.

Scikit-learn bevat een handige [pipeline-API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) om verschillende stappen van gegevensverwerking te combineren. Een **pipeline** is een keten van **schatters**. In ons geval maken we een pipeline die eerst polynomiale kenmerken toevoegt aan ons model en vervolgens de regressie traint:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Met `PolynomialFeatures(2)` wordt bedoeld dat we alle tweedegraads polynomen van de invoergegevens opnemen. In ons geval betekent dat alleen `DayOfYear`<sup>2</sup>, maar gegeven twee invoervariabelen X en Y, worden ook X<sup>2</sup>, XY en Y<sup>2</sup> toegevoegd. We kunnen ook hogere graads polynomen gebruiken als we dat willen.

Pipelines kunnen op dezelfde manier worden gebruikt als het oorspronkelijke `LinearRegression`-object, d.w.z. we kunnen de pipeline `fit`ten en vervolgens `predict` gebruiken om de voorspellingen te krijgen. Hier is de grafiek met testgegevens en de benaderingscurve:

<img alt="Polynomial regression" src="../../../../translated_images/nl/poly-results.ee587348f0f1f60b.webp" width="50%" />

Met polynomiale regressie kunnen we iets lagere MSE en hogere determinatie krijgen, maar niet significant. We moeten andere kenmerken in rekening brengen!

> Je kunt zien dat de minimale pompoenprijzen ergens rond Halloween worden waargenomen. Hoe kun je dat verklaren?

🎃 Gefeliciteerd, je hebt zojuist een model gemaakt dat kan helpen de prijs van taartpompoenen te voorspellen. Waarschijnlijk kun je dezelfde procedure herhalen voor alle pompoentypes, maar dat zou saai zijn. Laten we nu leren hoe we rekening kunnen houden met pompoensoorten in ons model!

## Categorische kenmerken

In een ideale wereld willen we prijzen voor verschillende pompoensoorten voorspellen met hetzelfde model. Echter, de kolom `Variety` is anders dan kolommen zoals `Month`, omdat deze niet-numerieke waarden bevat. Dergelijke kolommen worden **categorisch** genoemd.

[![ML voor beginners - Categorievoorspellingen met lineaire regressie](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML voor beginners - Categorievoorspellingen met lineaire regressie")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van het gebruik van categorische kenmerken.

Hier zie je hoe de gemiddelde prijs afhangt van de soort:

<img alt="Average price by variety" src="../../../../translated_images/nl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Om rekening te houden met de soort, moeten we deze eerst omzetten in numerieke vorm, of **encoderen**. Er zijn verschillende manieren om dat te doen:

* Eenvoudige **numerieke codering** bouwt een tabel van verschillende soorten, en vervangt dan de soortnaam door een index in die tabel. Dit is niet het beste idee voor lineaire regressie, omdat lineaire regressie de numerieke waarde van de index neemt en optelt bij het resultaat, maal een coëfficiënt. In ons geval is de relatie tussen het indexnummer en de prijs duidelijk niet-lineair, zelfs als we de indices in een specifieke volgorde plaatsen.
* **One-hot encoding** vervangt de kolom `Variety` door 4 verschillende kolommen, een voor elke soort. Elke kolom bevat `1` als de overeenkomstige rij van die betreffende soort is, en `0` anders. Dit betekent dat er vier coëfficiënten in lineaire regressie zijn, één voor elke pompoensoort, verantwoordelijk voor de "startprijs" (of liever gezegd "extra prijs") voor die specifieke soort.

De onderstaande code toont hoe we een soort one-hot kunnen encoderen:

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

Om lineaire regressie te trainen met one-hot gecodeerde variëteit als invoer, hoeven we alleen `X` en `y` correct te initialiseren:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

De rest van de code is hetzelfde als wat we hierboven gebruikten om lineaire regressie te trainen. Als je het probeert, zul je zien dat de mean squared error ongeveer hetzelfde is, maar we krijgen een veel hogere coëfficiënt van determinatie (~77%). Om nog nauwkeurigere voorspellingen te krijgen, kunnen we nog meer categorische kenmerken overwegen, evenals numerieke kenmerken zoals `Month` of `DayOfYear`. Om één grote array van features te krijgen, kunnen we `join` gebruiken:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Hier nemen we ook `City` en het type `Package` mee, wat ons een MSE van 2.84 (10%) en een determinatie van 0.94 geeft!

## Alles samenvoegen

Om het beste model te maken, kunnen we gecombineerde (one-hot gecodeerde categorische + numerieke) gegevens uit het bovenstaande voorbeeld samen met polynomiale regressie gebruiken. Hier is de volledige code voor jouw gemak:

```python
# trainingsgegevens instellen
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# maak train-test splitsing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# stel de pijplijn in en train deze
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# voorspel resultaten voor testgegevens
pred = pipeline.predict(X_test)

# bereken MSE en bepaling
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Dit zou ons de beste determinatiecoëfficiënt van bijna 97% en MSE=2.23 (~8% voorspellingsfout) moeten geven.

| Model | MSE | Determinatie |
|-------|-----|--------------|
| `DayOfYear` Lineair | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomiaal | 2.73 (17.0%) | 0.08 |
| `Variety` Lineair | 5.24 (19.7%) | 0.77 |
| Alle kenmerken Lineair | 2.84 (10.5%) | 0.94 |
| Alle kenmerken Polynomiaal | 2.23 (8.25%) | 0.97 |

🏆 Goed gedaan! Je hebt vier regressiemodellen gemaakt in één les en de modelkwaliteit verbeterd tot 97%. In de laatste sectie over regressie leer je over logistische regressie om categorieën te bepalen.

---
## 🚀Uitdaging

Test verschillende variabelen in dit notebook om te zien hoe correlatie overeenkomt met modelnauwkeurigheid.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

In deze les hebben we geleerd over lineaire regressie. Er zijn ook andere belangrijke soorten regressie. Lees over Stepwise, Ridge, Lasso en Elasticnet technieken. Een goede cursus om meer te leren is de [Stanford Statistical Learning cursus](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Opdracht

[Maak een model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Dit document is vertaald met behulp van de AI-vertalingsdienst [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u er rekening mee te houden dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het oorspronkelijke document in de oorspronkelijke taal moet als gezaghebbende bron worden beschouwd. Voor belangrijke informatie wordt een professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->