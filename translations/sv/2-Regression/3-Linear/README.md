# Bygg en regressionsmodell med Scikit-learn: regression på fyra sätt

## Nybörjarmeddelande

Linjär regression används när vi vill förutsäga ett **numeriskt värde** (till exempel huspris, temperatur eller försäljning).  
Det fungerar genom att hitta en rät linje som bäst representerar sambandet mellan indatafunktioner och utdata.

I denna lektion fokuserar vi på att förstå konceptet innan vi utforskar mer avancerade regressionstekniker.  
![Linear vs polynomial regression infographic](../../../../translated_images/sv/linear-polynomial.5523c7cb6576ccab.webp)  
> Informationsgrafik av [Dasani Madipalli](https://twitter.com/dasani_decoded)  
## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Den här lektionen finns tillgänglig i R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### Introduktion

Hittills har du utforskat vad regression är med hjälp av exempeldata från pumpaprisdatasetet som vi kommer använda i hela denna lektion. Du har också visualiserat det med Matplotlib.

Nu är du redo att fördjupa dig i regression för ML. Medan visualisering hjälper dig att förstå data, kommer den verkliga styrkan i maskininlärning från _tränade modeller_. Modeller tränas på historisk data för att automatiskt fånga databeroenden, och de låter dig förutsäga utfall för nya data som modellen inte har sett tidigare.

I denna lektion kommer du att lära dig mer om två typer av regression: _grundläggande linjär regression_ och _polynomregression_, tillsammans med en del av matematiken bakom dessa tekniker. Dessa modeller kommer att låta oss förutsäga pumpapris beroende på olika indata.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Klicka på bilden ovan för en kort videoöversikt av linjär regression.

> Genom hela detta utbildningsprogram antar vi minimala kunskaper i matematik och strävar efter att göra det tillgängligt för studenter från andra områden, så håll utkik efter anteckningar, 🧮 markeringar, diagram och andra inlärningsverktyg för att underlätta förståelsen.

### Förkunskaper

Du bör nu vara bekant med strukturen i pumpadata som vi undersöker. Den finns förladdad och förberedd rengjord i denna lektions _notebook.ipynb_-fil.  
I filen visas pumpapriset per bushel i en ny data frame. Se till att du kan köra dessa notebooks i kärnor i Visual Studio Code.

### Förberedelse

Som en påminnelse laddar du in denna data för att kunna ställa frågor om den.

- När är bästa tidpunkten att köpa pumpor?  
- Vilket pris kan jag förvänta mig för ett fall med miniatyrpumpor?  
- Bör jag köpa dem i halva bushel-korgar eller i 1 1/9 bushel-lådor?  
Låt oss fortsätta att gräva i denna data.

I föregående lektion skapade du en Pandas data frame och fyllde den med en del av den ursprungliga datasetet, standardiserande priser per bushel. Genom att göra detta kunde du dock bara samla in ungefär 400 datapunkter och endast för höstmånaderna.

Ta en titt på datan som vi förladdat i denna lektions tillhörande notebook. Data är förladdad och en initial spridningsdiagram är ritad för att visa månadsdata. Kanske kan vi få lite mer insikt om datan genom att rengöra den mer.

## En linjär regressionslinje

Som du lärde dig i Lektion 1 är målet med ett linjärt regressionsövning att kunna rita en linje för att:

- **Visa variabelrelationer**. Visa relationen mellan variabler  
- **Göra förutsägelser**. Göra noggranna förutsägelser om var en ny datapunkt skulle hamna i relation till den linjen.

Det är typiskt för **Minsta Kvadrat-regression** att rita denna typ av linje. Begreppet "Minsta Kvadrat" syftar på processen att minimera den totala felet i vår modell. För varje datapunkt mäter vi det vertikala avståndet (kallat residual) mellan den faktiska punkten och vår regressionslinje.

Vi kvadrerar dessa avstånd av två huvudsakliga skäl:

1. **Storlek över riktning:** Vi vill behandla ett fel på -5 lika som ett fel på +5. Kvadrering gör alla värden positiva.

2. **Straffa avvikare:** Kvadrering ger större vikt åt större fel och tvingar linjen att stanna närmare punkter som ligger långt ifrån.

Vi summerar sedan alla dessa kvadrerade värden. Vårt mål är att hitta den specifika linje där denna slutliga summa är som minst (det minsta möjliga värdet) – därav namnet "Minsta Kvadrat".

> **🧮 Visa mig matematiken**  
>  
> Denna linje, kallad _bästa anpassningslinje_ kan uttryckas med [en ekvation](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>  
> `X` är den 'förklarande variabeln'. `Y` är den 'beroende variabeln'. Linjens lutning är `b` och `a` är y-axelns intercept, vilket avser värdet på `Y` när `X = 0`.  
>  
>![calculate the slope](../../../../translated_images/sv/slope.f3c9d5910ddbfcf9.webp)  
>  
> Först beräkna lutningen `b`. Informationsgrafik av [Jen Looper](https://twitter.com/jenlooper)  
>  
> Med andra ord, och med hänvisning till vår pumpadata ursprungliga fråga: "förutsäg priset på en pumpa per bushel efter månad", skulle `X` avse priset och `Y` avse försäljningsmånaden.  
>  
>![complete the equation](../../../../translated_images/sv/calculation.a209813050a1ddb1.webp)  
>  
> Beräkna värdet på Y. Om du betalar runt 4 dollar måste det vara april! Informationsgrafik av [Jen Looper](https://twitter.com/jenlooper)  
>  
> Den matematik som beräknar linjen måste visa lutningen på linjen, som också beror på interceptet, eller var `Y` befinner sig när `X = 0`.  
>  
> Du kan observera beräkningsmetoden för dessa värden på [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) webbplats. Besök också [denna Minsta Kvadrat-kalkylator](https://www.mathsisfun.com/data/least-squares-calculator.html) för att se hur värdenas storlek påverkar linjen.

## Korrelationskoefficient

Ett till begrepp att förstå är **korrelationskoefficienten** mellan givna X- och Y-variabler. Med hjälp av ett spridningsdiagram kan du snabbt visualisera denna koefficient. Ett diagram med datapunkter utspridda i en prydlig linje har hög korrelation medan ett diagram med datapunkter utspridda överallt mellan X och Y har låg korrelation.

En bra linjär regressionsmodell är en som har hög (närmare 1 än 0) korrelationskoefficient med metoden Minsta Kvadrat Regression med regressionslinjen.

✅ Kör notebooken som hör till denna lektion och titta på spridningsdiagrammet Månad till Pris. Verkar datan som kopplar Månad till Pris för pumpaförsäljning ha hög eller låg korrelation enligt din visuella tolkning av spridningsdiagrammet? Ändras det om du använder en mer detaljerad mätning istället för `Month`, t.ex. *dag på året* (d.v.s. antal dagar sedan årets början)?

I koden nedan antar vi att vi har rengjort data och fått en dataframe kallad `new_pumpkins`, liknande följande:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price  
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  

> Koden för att rengöra data finns tillgänglig i [`notebook.ipynb`](notebook.ipynb). Vi har utfört samma rengöringssteg som i föregående lektion och beräknat kolumnen `DayOfYear` med följande uttryck:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Nu när du har en förståelse för matematiken bakom linjär regression, låt oss skapa en regressionsmodell för att se om vi kan förutsäga vilken pumpkinförpackning som kommer att ha de bästa pumpapriserna. Någon som köper pumpor för en högtidlig pumpaplats kanske vill ha denna information för att kunna optimera sina inköp av pumpkinpaket för sin plats.

## Letar efter korrelation

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Klicka på bilden ovan för en kort videoöversikt över korrelation.

Från föregående lektion har du förmodligen sett att genomsnittspriset för olika månader ser ut så här:

<img alt="Average price by month" src="../../../../translated_images/sv/barchart.a833ea9194346d76.webp" width="50%"/>

Detta antyder att det borde finnas någon korrelation, och vi kan försöka träna en linjär regressionsmodell för att förutsäga sambandet mellan `Month` och `Price`, eller mellan `DayOfYear` och `Price`. Här är spridningsdiagrammet som visar det senare förhållandet:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sv/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Låt oss se om det finns en korrelation med funktionen `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Det verkar som att korrelationen är ganska liten, -0,15 för `Month` och -0,17 för `DayOfYear`, men det kan finnas ett annat viktigt samband. Det ser ut som att det finns olika kluster av priser som motsvarar olika pumparter. För att bekräfta denna hypotes, låt oss plotta varje pumpkategori med en annan färg. Genom att skicka en `ax`-parameter till `scatter`-plotfunktionen kan vi rita alla punkter i samma diagram:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sv/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Vår undersökning tyder på att sorten har större effekt på priset än själva försäljningsdatumet. Vi kan se detta med ett stapeldiagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Bar graph of price vs variety" src="../../../../translated_images/sv/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Låt oss för tillfället fokusera endast på en pumpart, 'pie type', och se vilken effekt datumet har på priset:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sv/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Om vi nu beräknar korrelationen mellan `Price` och `DayOfYear` med funktionen `corr`, får vi något runt `-0.27` – vilket betyder att det är vettigt att träna en prediktiv modell.

> Innan du tränar en linjär regressionsmodell är det viktigt att se till att vår data är ren. Linjär regression fungerar inte bra med saknade värden, så det är vettigt att bli av med alla tomma celler:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
Ett annat tillvägagångssätt är att fylla de tomma värdena med medelvärden från motsvarande kolumn.

## Enkel linjär regression

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Klicka på bilden ovan för en kort videoöversikt över linjär och polynomisk regression.

För att träna vår linjära regressionsmodell kommer vi att använda **Scikit-learn**-biblioteket.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
Vi börjar med att separera indata (features) och förväntad utdata (label) i separata numpy-arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Observera att vi var tvungna att göra en `reshape` på indatan för att Linear Regression-paketet ska förstå den korrekt. Linjär regression förväntar sig en 2D-array som indata, där varje rad i arrayen motsvarar en vektor av indatafunktioner. I vårt fall, eftersom vi bara har en indata, behöver vi en array med formen N×1, där N är datasetets storlek.

Sedan måste vi dela datan i tränings- och testdatauppsättningar så att vi kan validera vår modell efter träning:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
Slutligen tar det bara två kodrader att träna den faktiska linjära regressionsmodellen. Vi definierar `LinearRegression`-objektet och anpassar det till vår data med metoden `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objektet efter att ha anpassats innehåller alla regressionskoefficienter, vilka kan nås via `.coef_`-egenskapen. I vårt fall finns det bara en koefficient, som bör vara runt `-0.017`. Det betyder att priserna verkar sjunka lite med tiden, men inte så mycket, ungefär 2 cent per dag. Vi kan också nå skärningspunkten för regressionen med Y-axeln genom `lin_reg.intercept_` - den kommer i vårt fall vara runt `21`, vilket indikerar priset i början av året.

För att se hur korrekt vår modell är kan vi förutsäga priser på en testdatamängd och sedan mäta hur nära våra förutsägelser är de förväntade värdena. Detta kan göras med hjälp av medelkvadratfel (MSE), vilket är medelvärdet av alla kvadrerade skillnader mellan förväntat och förutsagt värde.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Vårt fel verkar vara runt 2 poäng, vilket är ~17%. Inte så bra. En annan indikator på modellens kvalitet är **bestämningskoefficienten**, som kan erhållas så här:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Om värdet är 0 betyder det att modellen inte tar inputdata i beaktande, och agerar som den *sämsta linjära prediktorn*, vilket helt enkelt är medelvärdet av resultatet. Värdet 1 innebär att vi perfekt kan förutsäga alla förväntade utgångar. I vårt fall är koefficienten runt 0,06, vilket är ganska lågt.

Vi kan också plotta testdata tillsammans med regressionslinjen för att bättre se hur regressionen fungerar i vårt fall:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/sv/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomregression

En annan typ av linjär regression är polynomregression. Medan det ibland finns ett linjärt samband mellan variabler – ju större pumpan i volym, desto högre pris – kan dessa samband ibland inte plottas som ett plan eller rakt linje.

✅ Här är [fler exempel](https://online.stat.psu.edu/stat501/lesson/9/9.8) på data som kan använda polynomregression

Titta igen på sambandet mellan Datum och Pris. Verkar detta spridningsdiagram som något som nödvändigtvis bör analyseras med en rät linje? Kan inte priserna fluktuera? I det här fallet kan du prova polynomregression.

✅ Polynom är matematiska uttryck som kan bestå av en eller flera variabler och koefficienter

Polynomregression skapar en kurvad linje för att bättre passa icke-linjära data. I vårt fall, om vi inkluderar en kvadratisk `DayOfYear`-variabel i indata, bör vi kunna anpassa våra data med en parabolisk kurva, som kommer att ha ett minimum vid en viss punkt under året.

Scikit-learn innehåller ett användbart [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) för att kombinera olika steg i databehandlingen. En **pipeline** är en kedja av **estimators**. I vårt fall skapar vi en pipeline som först lägger till polynomfunktioner till vår modell och sedan tränar regressionen:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Att använda `PolynomialFeatures(2)` innebär att vi kommer att inkludera alla andragradspolynom från inputdata. I vårt fall betyder det bara `DayOfYear`<sup>2</sup>, men med två inputvariabler X och Y skulle detta lägga till X<sup>2</sup>, XY och Y<sup>2</sup>. Vi kan också använda högre gradens polynom om vi vill.

Pipelines kan användas på samma sätt som det ursprungliga `LinearRegression`-objektet, dvs vi kan `fit` pipelinen, och sedan använda `predict` för att få förutsägelser. Här är grafen som visar testdata och approximationskurvan:

<img alt="Polynomial regression" src="../../../../translated_images/sv/poly-results.ee587348f0f1f60b.webp" width="50%" />

Med polynomregression kan vi få något lägre MSE och högre bestämning, men inte signifikant. Vi behöver ta hänsyn till andra egenskaper!

> Du kan se att de lägsta pumpapriserna observeras runt Halloween. Hur kan du förklara detta?

🎃 Grattis, du skapade precis en modell som kan hjälpa till att förutsäga priset på pajpumpor. Du kan förmodligen upprepa samma procedur för alla pumpatyper, men det vore tråkigt. Låt oss nu lära oss hur vi tar pumpasort i beaktning i vår modell!

## Kategoriska egenskaper

I en perfekt värld vill vi kunna förutsäga priser för olika pumpasorter med samma modell. Dock skiljer sig kolumnen `Variety` något från kolumner som `Month`, eftersom den innehåller icke-numeriska värden. Sådana kolumner kallas **kategoriska**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Klicka på bilden ovan för en kort videoöversikt om användning av kategoriska egenskaper.

Här kan du se hur medelpriset beror på sort:

<img alt="Average price by variety" src="../../../../translated_images/sv/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

För att ta sort i beaktning måste vi först konvertera den till ett numeriskt format, eller **koda** den. Det finns flera sätt att göra det:

* Enkel **numerisk kodning** skapar en tabell med olika sorter och ersätter sedan sortnamnet med ett index i den tabellen. Detta är inte bästa idé för linjär regression, eftersom linjär regression tar det faktiska numeriska värdet på indexet och adderar det till resultatet, multiplicerat med någon koefficient. I vårt fall är sambandet mellan indexnummer och pris uppenbart icke-linjärt, även om vi ser till att indexen ordnas på något specifikt sätt.
* **One-hot-encoding** ersätter `Variety`-kolumnen med 4 olika kolumner, en för varje sort. Varje kolumn innehåller `1` om motsvarande rad är av en viss sort, och `0` annars. Det innebär att det kommer att finnas fyra koefficienter i linjär regression, en för varje pumpsort, ansvarig för "startpris" (eller snarare "tilläggspris") för just den sorten.

Koden nedan visar hur vi kan one-hot-koda en sort:

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

För att träna linjär regression med one-hot-kodad sort som input behöver vi bara initiera `X` och `y` data korrekt:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Resten av koden är densamma som vi använde ovan för att träna linjär regression. Om du provar detta kommer du se att medelkvadratfelet blir ungefär detsamma, men vi får en mycket högre bestämningskoefficient (~77%). För att få ännu mer exakta förutsägelser kan vi ta hänsyn till fler kategoriska egenskaper, samt numeriska egenskaper som `Month` eller `DayOfYear`. För att få en stor array med egenskaper kan vi använda `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Här tar vi också hänsyn till `City` och `Package`-typ, vilket ger oss MSE 2.84 (10%) och bestämning 0.94!

## Att sätta ihop allt

För att göra den bästa modellen kan vi använda kombinerade (one-hot-kodade kategoriska + numeriska) data från föregående exempel tillsammans med polynomregression. Här är den kompletta koden för din bekvämlighet:

```python
# skapa träningsdata
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# gör fördelning av tränings- och testdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# konfigurera och träna pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# förutsäg resultat för testdata
pred = pipeline.predict(X_test)

# beräkna MSE och förklaringsgrad
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Detta bör ge oss den bästa bestämningskoefficienten på nästan 97% och MSE=2.23 (~8% förutsägelsefel).

| Modell | MSE | Bestämning |
|-------|-----|------------|
| `DayOfYear` Linjär | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynom | 2.73 (17.0%) | 0.08 |
| `Variety` Linjär | 5.24 (19.7%) | 0.77 |
| Alla egenskaper Linjär | 2.84 (10.5%) | 0.94 |
| Alla egenskaper Polynom | 2.23 (8.25%) | 0.97 |

🏆 Bra jobbat! Du skapade fyra regressionsmodeller på en lektion och förbättrade modellens kvalitet till 97%. I det sista avsnittet om regression får du lära dig om logistisk regression för att bestämma kategorier.

---
## 🚀Utmaning

Testa flera olika variabler i den här anteckningsboken för att se hur korrelationen motsvarar modellens noggrannhet.

## [Quiz efter föreläsning](https://ff-quizzes.netlify.app/en/ml/)

## Genomgång & Självstudier

I den här lektionen lärde vi oss om linjär regression. Det finns andra viktiga typer av regression. Läs om Stepwise, Ridge, Lasso och Elasticnet-tekniker. En bra kurs att studera för att lära sig mer är [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Uppgift

[Bygg en modell](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Trots att vi strävar efter noggrannhet, bör du vara medveten om att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål ska betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår från användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->