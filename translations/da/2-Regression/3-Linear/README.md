# Byg en regressionsmodel ved hjælp af Scikit-learn: regression på fire måder

## Begynderbemærkning

Lineær regression bruges, når vi vil forudsige en **numerisk værdi** (for eksempel huspris, temperatur eller salg).  
Den fungerer ved at finde en ret linje, der bedst repræsenterer forholdet mellem inputfunktioner og output.

I denne lektion fokuserer vi på at forstå konceptet, før vi udforsker mere avancerede regressionsteknikker.  
![Lineær vs polynomiel regressions infografik](../../../../translated_images/da/linear-polynomial.5523c7cb6576ccab.webp)  
> Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)  
## [Forventningsquiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne lektion findes på R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### Introduktion

Indtil nu har du udforsket, hvad regression er med prøve-data indsamlet fra græskarpris-datasættet, som vi vil bruge igennem hele denne lektion. Du har også visualiseret det ved hjælp af Matplotlib.

Nu er du klar til at dykke dybere ned i regression inden for ML. Mens visualisering gør det muligt at forstå data, kommer den egentlige kraft ved maskinlæring fra _træning af modeller_. Modeller trænes på historiske data for automatisk at fange dataafhængigheder, og de giver dig mulighed for at forudsige resultater for nye data, som modellen ikke har set før.

I denne lektion vil du lære mere om to typer regression: _basal lineær regression_ og _polynomiel regression_, sammen med noget af den matematik, der ligger til grund for disse teknikker. Disse modeller vil give os mulighed for at forudsige græskarpriser afhængigt af forskellige inputdata.

[![ML for begyndere - Forstå lineær regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over lineær regression.

> I hele dette pensum antager vi minimal matematikkundskab og søger at gøre det tilgængeligt for studerende med baggrund fra andre felter, så hold øje med noter, 🧮 forklaringer, diagrammer og andre læringsværktøjer til hjælp med forståelsen.

### Forudsætning

Du bør nu være bekendt med strukturen i græskardataene, som vi undersøger. Du kan finde det forudindlæst og forudrense i denne lektions _notebook.ipynb_-fil. I filen vises græskarprisen pr. bushel i en ny data frame. Sørg for, at du kan køre disse notebooks i kerner i Visual Studio Code.

### Forberedelse

Som en påmindelse læser du disse data ind for at kunne stille spørgsmål til dem.

- Hvornår er det bedste tidspunkt at købe græskar?
- Hvilken pris kan jeg forvente for en kasse miniaturegræskar?
- Skal jeg købe dem i halvbushel-kurve eller i 1 1/9 bushel-kasser?

Lad os blive ved med at grave i disse data.

I den foregående lektion oprettede du en Pandas data frame og fyldte den med en del af det oprindelige datasæt, hvor priserne blev standardiseret pr. bushel. Ved at gøre det kunne du dog kun samle omkring 400 datapunkter og kun for efterårsmånederne.

Tag et kig på de data, vi forudindlæste i denne lektions ledsagende notebook. Dataene er forudindlæst, og et indledende spredningsdiagram er tegnet for at vise månedsdata. Måske kan vi få lidt mere indsigt i dataenes natur ved at rense det yderligere.

## En lineær regressionslinje

Som du lærte i Lektion 1, er målet med en lineær regressionsøvelse at kunne plotte en linje for at:

- **Vise variabelrelationer**. Vise forholdet mellem variable  
- **Foretage forudsigelser**. Foretage præcise forudsigelser om, hvor et nyt datapunkt vil falde i forhold til den linje.

Det er typisk for **Minste Kvadraters Regression** at tegne denne type linje. Udtrykket "Minste Kvadrater" henviser til processen med at minimere den samlede fejl i vores model. For hvert datapunkt måler vi den lodrette afstand (kaldet residual) mellem det faktiske punkt og vores regressionslinje.

Vi kvadrerer disse afstande af to hovedårsager:

1. **Størrelse frem for retning:** Vi vil behandle en fejl på -5 på samme måde som en fejl på +5. Kvadrering gør alle værdier positive.

2. **Straffe for afvigere:** Kvadrering giver større vægt til større fejl og tvinger linjen til at holde sig tættere på punkter, som er langt væk.

Derefter lægger vi alle disse kvadrerede værdier sammen. Vores mål er at finde den specifikke linje, hvor dette endelige summen er mindst (mindste mulige værdi)—deraf navnet "Minste Kvadrater".

> **🧮 Vis mig matematikken**  
>  
> Denne linje, kaldet _den bedste tilpassede linje_, kan udtrykkes ved [en ligning](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>  
> `X` er den 'forklarende variabel'. `Y` er den 'afhængige variabel'. Hældningen af linjen er `b` og `a` er skæringspunktet med y-aksen, hvilket henviser til værdien af `Y` når `X = 0`.  
>  
>![beregn hældningen](../../../../translated_images/da/slope.f3c9d5910ddbfcf9.webp)  
>  
> Først beregnes hældningen `b`. Infografik af [Jen Looper](https://twitter.com/jenlooper)  
>  
> Med andre ord, og med henvisning til vores græskardatas oprindelige spørgsmål: "forudsig prisen på et græskar pr. bushel efter måned", ville `X` referere til prisen og `Y` henvise til salgs-måneden.  
>  
>![fuldfør ligningen](../../../../translated_images/da/calculation.a209813050a1ddb1.webp)  
>  
> Beregn værdien af Y. Hvis du betaler omkring 4 $, må det være april! Infografik af [Jen Looper](https://twitter.com/jenlooper)  
>  
> Matematikken, der beregner linjen, skal demonstrere hældningen på linjen, som også afhænger af skæringspunktet, eller hvor `Y` er placeret, når `X = 0`.  
>  
> Du kan se beregningsmetoden for disse værdier på [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) websitet. Besøg også [denne mindst-kvadrater-lommeregner](https://www.mathsisfun.com/data/least-squares-calculator.html) for at se, hvordan værdiernes tal påvirker linjen.

## Korrelation

Et begreb mere at forstå er **Korrelationskoefficienten** mellem givne X- og Y-variable. Ved hjælp af et spredningsdiagram kan du hurtigt visualisere denne koefficient. En graf med datapunkter spredt i en pæn linje har høj korrelation, mens en graf med datapunkter spredt overalt mellem X og Y har lav korrelation.

En god lineær regressionsmodel vil være en, der har en høj (tættere på 1 end 0) korrelationskoefficient ved brug af Minste Kvadraters-regressionsmetoden med en regressionslinje.

✅ Kør notebook'en til denne lektion og se på spredningsdiagrammet for måned og pris. Virker dataene, der forbinder måned til pris for græskarsalg, til at have høj eller lav korrelation baseret på din visuelle fortolkning af spredningsdiagrammet? Ændres det, hvis du bruger en mere finmasket måling i stedet for `Month`, f.eks. *dag på året* (dvs. antal dage siden årets start)?

I koden nedenfor antager vi, at vi har renset dataene og opnået en data frame kaldet `new_pumpkins`, som ligner følgende:

ID | Måned | DagPåÅret | Sort | By | Pakke | Lav Pris | Høj Pris | Pris  
---|-------|-----------|-------|----|--------|----------|----------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel-kasser | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel-kasser | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel-kasser | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel-kasser | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel-kasser | 15.0 | 15.0 | 13.636364  

> Koden til at rense dataene findes i [`notebook.ipynb`](notebook.ipynb). Vi har udført de samme rensningsskridt som i den foregående lektion og har beregnet `DayOfYear` kolonnen ved hjælp af følgende udtryk:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Nu hvor du har en forståelse for matematikken bag lineær regression, lad os oprette en regressionsmodel for at se, om vi kan forudsige, hvilken pakke med græskar der vil have de bedste priser. En person, der køber græskar til en ferie-græskarmark, kunne have brug for denne information for at optimere deres køb af græskarpakker til markedet.

## Søger efter korrelation

[![ML for begyndere - Søger efter korrelation: nøglen til lineær regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over korrelation.

Fra den foregående lektion har du sandsynligvis set, at gennemsnitsprisen for forskellige måneder ser sådan ud:

<img alt="Gennemsnitspris efter måned" src="../../../../translated_images/da/barchart.a833ea9194346d76.webp" width="50%"/>

Dette tyder på, at der burde være en vis korrelation, og vi kan prøve at træne en lineær regressionsmodel til at forudsige forholdet mellem `Month` og `Price`, eller mellem `DayOfYear` og `Price`. Her er spredningsdiagrammet, der viser sidstnævnte forhold:

<img alt="Spredningsdiagram over pris vs. dag på året" src="../../../../translated_images/da/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Lad os se, om der er en korrelation ved hjælp af `corr` funktionen:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Det ser ud til, at korrelationen er ret lille, -0.15 for `Month` og -0.17 for `DayOfMonth`, men der kunne være en anden vigtig sammenhæng. Det ser ud til, at der er forskellige grupper af priser svarende til forskellige græskarsorter. For at bekræfte denne hypotese, lad os plotte hver græskarkategori med en anden farve. Ved at sende en `ax` parameter til `scatter` plotting-funktionen kan vi plotte alle punkter på samme graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Spredningsdiagram over pris vs. dag på året" src="../../../../translated_images/da/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Vores undersøgelse antyder, at sorten har større effekt på den samlede pris end den faktiske salgsdato. Dette kan vi se med et søjlediagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Søjlediagram over pris vs. sort" src="../../../../translated_images/da/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Lad os indtil videre fokusere på én græskartype, 'pie type', og undersøge, hvilken effekt datoen har på prisen:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Spredningsdiagram over pris vs. dag på året" src="../../../../translated_images/da/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Hvis vi nu beregner korrelationen mellem `Price` og `DayOfYear` ved hjælp af `corr` funktionen, får vi noget i retning af `-0.27` - hvilket betyder, at det giver mening at træne en forudsigelsesmodel.

> Før vi træner en lineær regressionsmodel, er det vigtigt at sikre, at vores data er rensede. Lineær regression fungerer ikke godt med manglende værdier, så det giver mening at fjerne alle tomme celler:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
En anden tilgang ville være at udfylde de tomme værdier med gennemsnitsværdier fra den tilsvarende kolonne.

## Simpel lineær regression

[![ML for begyndere - Lineær og polynomiel regression ved brug af Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over lineær og polynomiel regression.

For at træne vores lineære regressionsmodel vil vi bruge **Scikit-learn** biblioteket.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
Vi starter med at adskille inputværdier (features) og den forventede output (label) i separate numpy-arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Bemærk, at vi var nødt til at udføre `reshape` på inputdataene, for at Linear Regression-pakken kunne forstå dem korrekt. Lineær Regression forventer et 2D-array som input, hvor hver række af arrayet svarer til en vektor af inputfunktioner. I vores tilfælde, da vi kun har én input, har vi brug for et array med form N&times;1, hvor N er datamængdens størrelse.

Derefter skal vi splitte dataene op i trænings- og testdatasæt, så vi kan validere vores model efter træning:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
Endelig tager træningen af den faktiske lineære regressionsmodel kun to kodelinjer. Vi definerer et `LinearRegression`-objekt og fitter det til vores data ved hjælp af `fit`-metoden:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objektet efter `fit`-ning indeholder alle koefficienterne for regressionen, som kan tilgås ved hjælp af `.coef_`-egenskaben. I vores tilfælde er der kun én koefficient, som burde være omkring `-0.017`. Det betyder, at priserne tilsyneladende falder lidt over tid, men ikke for meget, omkring 2 cent per dag. Vi kan også tilgå skæringspunktet for regressionen med Y-aksen ved hjælp af `lin_reg.intercept_` - det vil være omkring `21` i vores tilfælde, hvilket angiver prisen i begyndelsen af året.

For at se hvor nøjagtig vores model er, kan vi forudsige priser på et testdatasæt, og derefter måle hvor tæt vores forudsigelser er på de forventede værdier. Dette kan gøres ved hjælp af mean square error (MSE) målemetrik, som er gennemsnittet af alle kvadrerede forskelle mellem forventede og forudsagte værdier.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Vores fejl ser ud til at være omkring 2 point, hvilket er ~17%. Ikke så godt. En anden indikator for modellens kvalitet er **determinationskoefficienten**, som kan opnås på denne måde:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
 Hvis værdien er 0, betyder det, at modellen ikke tager inputdata i betragtning, og fungerer som den *værste lineære forudsigelse*, som simpelthen er gennemsnitsværdien af resultatet. Værdien 1 betyder, at vi perfekt kan forudsige alle forventede output. I vores tilfælde er koefficienten omkring 0,06, hvilket er ganske lavt.

Vi kan også plotte testdata sammen med regressionslinjen for bedre at se, hvordan regressionen fungerer i vores tilfælde:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/da/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomiel Regression

En anden type lineær regression er polynomiel regression. Selvom der nogle gange er et lineært forhold mellem variabler - jo større græskar i volumen, desto højere pris - kan disse forhold til tider ikke plottes som et plan eller en lige linje.

✅ Her er [nogle flere eksempler](https://online.stat.psu.edu/stat501/lesson/9/9.8) på data, som kunne bruge polynomiel regression

Tag endnu et kig på forholdet mellem Dato og Pris. Ser dette scatterplot ud som om det nødvendigvis skal analyseres med en lige linje? Kan priser ikke svinge? I dette tilfælde kan du prøve polynomiel regression.

✅ Polynomier er matematiske udtryk, som kan bestå af en eller flere variable og koefficienter

Polynomiel regression skaber en buet linje for bedre at tilpasse ikke-lineære data. I vores tilfælde, hvis vi medtager et kvadreret `DayOfYear`-variabel i inputdata, bør vi kunne tilpasse vores data med en parabolsk kurve, som vil have et minimum på et bestemt tidspunkt i løbet af året.

Scikit-learn inkluderer en nyttig [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) til at kombinere forskellige trin i databehandlingen sammen. En **pipeline** er en kæde af **estimators**. I vores tilfælde vil vi skabe en pipeline, der først tilføjer polynomielle features til vores model, og derefter træner regressionen:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

At bruge `PolynomialFeatures(2)` betyder, at vi vil inkludere alle andengradspolynomier fra inputdata. I vores tilfælde vil det bare betyde `DayOfYear`<sup>2</sup>, men givet to inputvariabler X og Y, vil dette tilføje X<sup>2</sup>, XY og Y<sup>2</sup>. Vi kan også bruge polynomier med højere grad, hvis vi ønsker.

Pipelines kan bruges på samme måde som det oprindelige `LinearRegression`-objekt, dvs. vi kan `fit` pipelinen og derefter bruge `predict` til at få forudsigelsesresultater. Her er grafen, der viser testdata og tilpasningskurven:

<img alt="Polynomial regression" src="../../../../translated_images/da/poly-results.ee587348f0f1f60b.webp" width="50%" />

Med polynomiel regression kan vi opnå lidt lavere MSE og højere determination, men ikke signifikant. Vi skal tage andre features i betragtning!

> Du kan se, at de laveste græskarpriser observeres et sted omkring Halloween. Hvordan kan du forklare dette?

🎃 Tillykke, du har netop skabt en model, der kan hjælpe med at forudsige prisen på tærtegræskar. Du kan sikkert gentage den samme procedure for alle græskartyper, men det ville være kedeligt. Lad os nu lære, hvordan vi tager græskarvariant i betragtning i vores model!

## Kategoriske Features

I en ideel verden vil vi kunne forudsige priser for forskellige græskarvarianter ved hjælp af den samme model. Men `Variety`-kolonnen er lidt anderledes end kolonner som `Month`, fordi den indeholder ikke-numeriske værdier. Sådanne kolonner kaldes **kategoriske**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over brugen af kategoriske features.

Her kan du se, hvordan gennemsnitsprisen afhænger af variant:

<img alt="Average price by variety" src="../../../../translated_images/da/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

For at tage variant i betragtning, skal vi først konvertere den til numerisk form, eller **encode** den. Der er flere måder, vi kan gøre det på:

* Simpel **numerisk encoding** bygger en tabel over forskellige varianter og erstatter derefter variantnavnet med et indeks i tabellen. Dette er ikke den bedste ide for lineær regression, fordi lineær regression tager den faktiske numeriske værdi af indekset og lægger det til resultatet, multipliceret med en koefficient. I vores tilfælde er forholdet mellem indeksnummer og pris tydeligt ikke-lineært, selv hvis vi sørger for, at indeksene er ordnet på en bestemt måde.
* **One-hot encoding** vil erstatte `Variety`-kolonnen med 4 forskellige kolonner, en for hver variant. Hver kolonne vil indeholde `1`, hvis den tilsvarende række er af den givne variant, og `0` ellers. Det betyder, at der vil være fire koefficienter i lineær regression, én for hver græskarvariant, ansvarlig for "startpris" (eller snarere "ekstra pris") for netop denne variant.

Koden nedenfor viser, hvordan vi kan one-hot encode en variant:

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

For at træne lineær regression ved brug af one-hot encoded variant som input, skal vi blot initialisere `X` og `y` data korrekt:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Resten af koden er den samme, som vi brugte ovenfor til at træne lineær regression. Hvis du prøver det, vil du se, at mean squared error er omtrent det samme, men vi får en meget højere determinationskoefficient (~77%). For at få endnu mere præcise forudsigelser kan vi tage flere kategoriske features i betragtning, såvel som numeriske features, såsom `Month` eller `DayOfYear`. For at få et stort array af features kan vi bruge `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Her tager vi også hensyn til `City` og `Package`-type, hvilket giver os MSE 2,84 (10%) og determination 0,94!

## At samle det hele

For at lave den bedste model kan vi bruge kombinerede (one-hot encoded kategoriske + numeriske) data fra ovenstående eksempel sammen med polynomiel regression. Her er den komplette kode for din bekvemmelighed:

```python
# opsæt træningsdata
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# lav trænings- og testopdeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# opsæt og træn pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# forudsig resultater for testdata
pred = pipeline.predict(X_test)

# beregn MSE og bestemmelse
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Dette skulle give os den bedste determinationskoefficient på næsten 97%, og MSE=2,23 (~8% fejl i forudsigelsen).

| Model | MSE | Determination |
|-------|-----|---------------|
| `DayOfYear` Lineær | 2,77 (17,2%) | 0,07 |
| `DayOfYear` Polynomiel | 2,73 (17,0%) | 0,08 |
| `Variety` Lineær | 5,24 (19,7%) | 0,77 |
| Alle features Lineær | 2,84 (10,5%) | 0,94 |
| Alle features Polynomiel | 2,23 (8,25%) | 0,97 |

🏆 Godt klaret! Du har skabt fire regressionsmodeller i én lektion og forbedret modelkvaliteten til 97%. I den sidste sektion om regression vil du lære om logistisk regression til bestemmelse af kategorier.

---
## 🚀Udfordring

Test flere forskellige variable i denne notesbog for at se, hvordan korrelation svarer til modelnøjagtighed.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

I denne lektion lærte vi om lineær regression. Der findes andre vigtige typer regression. Læs om Stepwise, Ridge, Lasso og Elasticnet teknikker. Et godt kursus at studere for at lære mere er [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Opgave

[Byg en model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, bedes du være opmærksom på, at automatiske oversættelser kan indeholde fejl eller unøjagtigheder. Det oprindelige dokument på dets modersmål bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->