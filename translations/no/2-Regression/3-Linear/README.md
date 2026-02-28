# Bygg en regresjonsmodell med Scikit-learn: regresjon på fire måter

## Nybegynnermerknad

Lineær regresjon brukes når vi ønsker å predikere en **numerisk verdi** (for eksempel huspris, temperatur eller salg).
Den fungerer ved å finne en rett linje som best representerer forholdet mellom inngangsfunksjoner og utgangen.

I denne leksjonen fokuserer vi på å forstå konseptet før vi utforsker mer avanserte regresjonsteknikker.
![Lineær vs polynom regresjon infografikk](../../../../translated_images/no/linear-polynomial.5523c7cb6576ccab.webp)
> Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Pre-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er tilgjengelig i R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduksjon 

Så langt har du utforsket hva regresjon er med prøveutvalg samlet fra gresskarprisdatasettet som vi skal bruke gjennom denne leksjonen. Du har også visualisert det med Matplotlib.

Nå er du klar til å dykke dypere inn i regresjon for ML. Mens visualisering lar deg forstå data, kommer den virkelige kraften i Maskinlæring fra _trening av modeller_. Modeller trenes på historiske data for automatisk å fange opp dataavhengigheter, og de tillater deg å forutsi utfall for nye data, som modellen ikke har sett før.

I denne leksjonen vil du lære mer om to typer regresjon: _grunnleggende lineær regresjon_ og _polynomisk regresjon_, sammen med noe av matematikken som ligger til grunn for disse teknikkene. Disse modellene vil tillate oss å forutsi gresskarpriser basert på forskjellige inngangsdata. 

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over lineær regresjon.

> Gjennom dette pensumet forutsetter vi minimal kunnskap i matematikk, og søker å gjøre det tilgjengelig for studenter fra andre fagfelt, så hold øye med notater, 🧮 markeringer, diagrammer og andre læringsverktøy for å hjelpe forståelsen.

### Forutsetninger

Du bør nå være kjent med strukturen på gresskardataene som vi undersøker. Du kan finne det forhåndslastet og forhåndsrenset i denne leksjonens _notebook.ipynb_-fil. I filen vises gresskarpris per bushel i en ny dataramme. Sørg for at du kan kjøre disse notatbøkene i kjerne i Visual Studio Code.

### Forberedelse

Som en påminnelse laster du inn disse dataene for å stille spørsmål til dem. 

- Når er det beste tidspunktet å kjøpe gresskar?
- Hvilken pris kan jeg forvente på en kasse med miniatyrgresskar?
- Bør jeg kjøpe dem i halvbukettkurver eller i 1 1/9 bushel-boksen?
La oss fortsette å grave i disse dataene.

I forrige leksjon opprettet du en Pandas-dataramme og fylte den med deler av det originale datasettet, og standardiserte prisene per bushel. Ved å gjøre det, klarte du imidlertid bare å samle inn omtrent 400 datapunkter og kun for høstmånedene. 

Ta en titt på dataene vi forhåndslastet i denne leksjonens medfølgende notatbok. Dataene er forhåndslastet, og en innledende spredningsdiagram er tegnet for å vise månedsdata. Kanskje vi kan få litt mer detalj om datanaturen ved å rense den mer.

## En lineær regresjonslinje

Som du lærte i leksjon 1, er målet med en lineær regresjonsøvelse å kunne plotte en linje for å:

- **Vis variable relasjoner**. Vis forholdet mellom variabler
- **Gjøre prediksjoner**. Gjør nøyaktige prediksjoner om hvor et nytt datapunkt vil falle i forhold til den linjen. 
 
Det er vanlig med **minste kvadraters regresjon** å tegne denne typen linje. Begrepet "minste kvadraters" refererer til prosessen med å minimere den totale feilen i modellen vår. For hvert datapunkt måler vi den vertikale avstanden (kalt et residual) mellom det faktiske punktet og regresjonslinjen vår.  

Vi kvadrerer disse avstandene av to hovedgrunner:  

1. **Størrelse over retning:** Vi ønsker å behandle en feil på -5 det samme som en feil på +5. Kvadrering gjør alle verdier positive.  

2. **Straff for uteliggere:** Kvadrering gir mer vekt til større feil, og tvinger linjen til å holde seg nærmere punkter som ligger langt unna.  

Deretter legger vi sammen alle disse kvadrerte verdiene. Målet vårt er å finne den spesifikke linjen hvor denne endelige summen er på sitt laveste (den minste mulige verdien) — derav navnet "minste kvadraters".  

> **🧮 Vis meg matematikken** 
> 
> Denne linjen, kalt _best fit-linjen_ kan uttrykkes ved [en ligning](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` er den 'forklarende variabelen'. `Y` er den 'avhengige variabelen'. Stigningstallet til linjen er `b` og `a` er y-avskjæringen, som refererer til verdien av `Y` når `X = 0`. 
>
>![beregn stigningen](../../../../translated_images/no/slope.f3c9d5910ddbfcf9.webp)
>
> Først beregnes stigningen `b`. Infografikk av [Jen Looper](https://twitter.com/jenlooper)
>
> Med andre ord, og med henvisning til vårt gresskardatas opprinnelige spørsmål: "forutsi prisen på et gresskar per bushel etter måned", vil `X` referere til prisen og `Y` til salgsmåneden. 
>
>![fullfør ligningen](../../../../translated_images/no/calculation.a209813050a1ddb1.webp)
>
> Beregn verdien av Y. Hvis du betaler rundt 4 dollar, må det være april! Infografikk av [Jen Looper](https://twitter.com/jenlooper)
>
> Matematikk som beregner linjen må demonstrere linjens stigning, som også avhenger av avskjæringen, eller hvor `Y` er plassert når `X = 0`.
>
> Du kan observere metoden for beregning av disse verdiene på [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) nettstedet. Besøk også [denne minste kvadraters kalkulator](https://www.mathsisfun.com/data/least-squares-calculator.html) for å se hvordan tallverdiene påvirker linjen.

## Korrelasjon

Et annet begrep å forstå er **korrelasjonskoeffisienten** mellom gitte X og Y variabler. Ved å bruke et spredningsdiagram kan du raskt visualisere denne koeffisienten. Et plott med datapunkter fordelt i en pen linje har høy korrelasjon, men et plott med datapunkter spredt overalt mellom X og Y har lav korrelasjon.

En god lineær regresjonsmodell vil være en som har høy (nærmere 1 enn 0) korrelasjonskoeffisient ved bruk av minste kvadraters regresjonsmetode med en regresjonslinje.

✅ Kjør notatboken som følger med denne leksjonen og se på spredningsplottet for Måned mot Pris. Ser dataene som kobler Måned til Pris for gresskarsalg ut til å ha høy eller lav korrelasjon, i henhold til din visuelle tolkning av spredningsplottet? Endres det hvis du bruker et mer detaljert mål i stedet for `Month`, f.eks. *dag i året* (dvs. antall dager siden begynnelsen av året)?

I koden nedenfor antar vi at vi har renset dataene, og oppnådd en dataramme kalt `new_pumpkins`, lik følgende:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Koden for å rense dataene er tilgjengelig i [`notebook.ipynb`](notebook.ipynb). Vi har utført de samme renseprosedyrene som i forrige leksjon, og har beregnet kolonnen `DayOfYear` ved hjelp av følgende uttrykk: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nå som du har forståelse for matematikken bak lineær regresjon, la oss lage en regresjonsmodell for å se om vi kan forutsi hvilken pakke med gresskar som vil ha de beste gresskarprisene. Noen som kjøper gresskar til en høstgresskarhage kan ønske denne informasjonen for å kunne optimalisere sine kjøp av gresskarpakker til hagen.

## Ser etter korrelasjon

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over korrelasjon.

Fra forrige leksjon har du sannsynligvis sett at gjennomsnittsprisen for ulike måneder ser slik ut:

<img alt="Gjennomsnittspris per måned" src="../../../../translated_images/no/barchart.a833ea9194346d76.webp" width="50%"/>

Dette antyder at det bør være en viss korrelasjon, og vi kan prøve å trene en lineær regresjonsmodell for å predikere forholdet mellom `Month` og `Price`, eller mellom `DayOfYear` og `Price`. Her er spredningsplottet som viser det sistnevnte forholdet:

<img alt="Spredningsplott av Pris vs. Dag i året" src="../../../../translated_images/no/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

La oss se om det er en korrelasjon ved å bruke `corr`-funksjonen:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Det ser ut som korrelasjonen er ganske liten, -0.15 basert på `Month` og -0.17 basert på `DayOfYear`, men det kan være et annet viktig forhold. Det ser ut til å være forskjellige klynger av priser som tilsvarer forskjellige gresskarsorter. For å bekrefte denne hypotesen, la oss tegne hver gresskarkategori med en annen farge. Ved å sende inn en `ax`-parameter til `scatter`-plottfunksjonen kan vi tegne alle punktene på samme graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Spredningsplott av Pris vs. Dag i året" src="../../../../translated_images/no/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Vår undersøkelse antyder at sort har større effekt på den totale prisen enn den faktiske salgsdatoen. Vi kan se dette med et stolpediagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stolpediagram av pris vs sort" src="../../../../translated_images/no/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

La oss fokusere for øyeblikket bare på én gresskarsort, 'pie type', og se hvilken effekt datoen har på prisen:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Spredningsplott av Pris vs. Dag i året" src="../../../../translated_images/no/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Hvis vi nå beregner korrelasjonen mellom `Price` og `DayOfYear` ved bruk av `corr`-funksjonen, får vi noe som `-0.27` – noe som betyr at det gir mening å trene en prediksjonsmodell.

> Før vi trener en lineær regresjonsmodell, er det viktig å sørge for at dataene våre er rene. Lineær regresjon fungerer ikke godt med manglende verdier, så det gir mening å kvitte seg med alle tomme celler:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

En annen tilnærming kan være å fylle de tomme verdiene med gjennomsnittsverdier fra den tilsvarende kolonnen.

## Enkel lineær regresjon

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Klikk på bildet ovenfor for en kort videooversikt over lineær og polynomisk regresjon.

For å trene vår lineære regresjonsmodell, vil vi bruke **Scikit-learn**-biblioteket.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Vi starter med å separere inngangsverdier (egenskaper) og forventet utgang (etikett) i separate numpy-arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Merk at vi måtte utføre `reshape` på inngangsdataene for at Linear Regression-pakken skulle forstå det korrekt. Lineær regresjon forventer et 2D-array som input, der hver rad av arrayet tilsvarer en vektor med inngangsegenskaper. I vårt tilfelle, siden vi bare har én input, trenger vi et array med formen N&times;1, der N er datastørrelsen.

Deretter må vi splitte dataene i trenings- og testdatasett, slik at vi kan validere modellen vår etter trening:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Til slutt tar trening av den faktiske lineære regresjonsmodellen bare to kodelinjer. Vi definerer `LinearRegression`-objektet, og tilpasser det til dataene våre med `fit`-metoden:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objektet `LinearRegression` etter at det er blitt `fit`-tet inneholder alle koeffisientene til regresjonen, som kan aksesseres ved hjelp av `.coef_`-egenskapen. I vårt tilfelle er det bare én koeffisient, som bør være rundt `-0.017`. Det betyr at prisene ser ut til å synke litt med tiden, men ikke så mye, omtrent 2 cent per dag. Vi kan også aksessere skjæringspunktet til regresjonen med Y-aksen ved å bruke `lin_reg.intercept_` - det vil være rundt `21` i vårt tilfelle, noe som indikerer prisen ved begynnelsen av året.

For å se hvor nøyaktig modellen vår er, kan vi predikere priser på et testdatasett, og deretter måle hvor nær våre prediksjoner er til de forventede verdiene. Dette kan gjøres ved hjelp av gjennomsnittlig kvadratfeil (MSE) metrikken, som er gjennomsnittet av alle kvadrerte forskjeller mellom forventet og predikert verdi.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Feilen vår ser ut til å være rundt 2 poeng, som er ~17%. Ikke så bra. En annen indikator på modellkvalitet er **determinasjonskoeffisienten**, som kan oppnås på denne måten:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
 Hvis verdien er 0, betyr det at modellen ikke tar hensyn til innndataene, og fungerer som den *verste lineære prediktoren*, som rett og slett er et gjennomsnitt av resultatet. Verdien 1 betyr at vi kan perfekt predikere alle forventede utganger. I vårt tilfelle er koeffisienten rundt 0.06, som er ganske lav.

Vi kan også plotte testdata sammen med regresjonslinjen for bedre å se hvordan regresjon fungerer i vårt tilfelle:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/no/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomregresjon

En annen type lineær regresjon er polynomregresjon. Selv om det noen ganger er et lineært forhold mellom variabler – jo større gresskaret er i volum, desto høyere er prisen – kan disse forholdene noen ganger ikke plottes som et plan eller en rett linje.

✅ Her er [noen flere eksempler](https://online.stat.psu.edu/stat501/lesson/9/9.8) på data som kan bruke polynomregresjon

Ta en ny titt på forholdet mellom dato og pris. Virker ikke dette scatterplottet som om det nødvendigvis bør analyseres med en rett linje? Kan ikke prisene svinge? I så fall kan du prøve polynomregresjon.

✅ Polynomier er matematiske uttrykk som kan bestå av en eller flere variabler og koeffisienter

Polynomregresjon lager en buet linje for bedre å tilpasse ikke-lineære data. I vårt tilfelle, hvis vi inkluderer en kvadratisk `DayOfYear`-variabel i inndataene, bør vi kunne tilpasse våre data med en parabolsk kurve, som vil ha et minimum på et visst punkt i løpet av året.

Scikit-learn inkluderer en nyttig [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) for å kombinere forskjellige trinn i databehandling. En **pipeline** er en kjede av **estimatorer**. I vårt tilfelle vil vi lage en pipeline som først legger til polynomfunksjoner til modellen vår, og deretter trener regresjonen:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Å bruke `PolynomialFeatures(2)` betyr at vi inkluderer alle andregradspolynomer fra inndataene. I vårt tilfelle betyr det bare `DayOfYear`<sup>2</sup>, men gitt to inngangsvariabler X og Y, vil dette legge til X<sup>2</sup>, XY og Y<sup>2</sup>. Vi kan også bruke polynomer med høyere grad hvis vi ønsker det.

Pipelines kan brukes på samme måte som det originale `LinearRegression`-objektet, dvs. vi kan `fit`-te pipelinen, og deretter bruke `predict` for å få prediksjonsresultater. Her er grafen som viser testdataene og tilnærmingskurven:

<img alt="Polynomial regression" src="../../../../translated_images/no/poly-results.ee587348f0f1f60b.webp" width="50%" />

Ved å bruke polynomregresjon kan vi få litt lavere MSE og høyere determinering, men ikke signifikant. Vi må ta hensyn til andre funksjoner!

> Du kan se at de laveste gresskarprisene observeres et sted rundt Halloween. Hvordan kan du forklare dette?

🎃 Gratulerer, du har nettopp laget en modell som kan hjelpe til med å forutsi prisen på paigresskar. Du kan sannsynligvis gjenta samme prosedyre for alle gresskartyper, men det ville være tidkrevende. La oss nå lære hvordan vi tar hensyn til gresskarvariant i modellen vår!

## Kategoriske variabler

I en ideell verden ønsker vi å kunne forutsi priser for forskjellige gresskarvarianter ved å bruke den samme modellen. Men kolonnen `Variety` er litt annerledes enn kolonner som `Month`, fordi den inneholder ikke-numeriske verdier. Slike kolonner kalles **kategoriske**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Klikk på bildet over for en kort videooversikt over bruk av kategoriske variabler.

Her kan du se hvordan gjennomsnittsprisen avhenger av variasjon:

<img alt="Average price by variety" src="../../../../translated_images/no/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

For å ta varianter i betraktning, må vi først konvertere det til numerisk form, eller **kode** det. Det finnes flere måter vi kan gjøre det på:

* Enkel **numerisk koding** bygger en tabell med forskjellige varianter, og erstatter så variantnavnet med en indeks i tabellen. Dette er ikke den beste idéen for lineær regresjon, fordi lineær regresjon tar den faktiske numeriske verdien av indeksen, og legger det til resultatet multiplisert med en koeffisient. I vårt tilfelle er forholdet mellom indeksnummer og pris klart ikke-lineært, selv om vi sørger for at indeksene er ordnet på en bestemt måte.
* **One-hot-koding** erstatter kolonnen `Variety` med 4 forskjellige kolonner, én for hver variant. Hver kolonne inneholder `1` hvis raden tilsvarer den aktuelle varianten, og `0` ellers. Det betyr at det vil være fire koeffisienter i den lineære regresjonen, én for hver gresskarvariant, ansvarlig for "startpris" (eller heller "tillegg" pris) for den aktuelle varianten.

Koden nedenfor viser hvordan vi kan en-til-en kode en variant:

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

For å trene lineær regresjon med one-hot kodet variant som input, må vi bare initialisere `X` og `y` data korrekt:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Resten av koden er den samme som vi brukte ovenfor for å trene lineær regresjon. Hvis du prøver det, vil du se at gjennomsnittlig kvadratfeil er omtrent den samme, men vi får mye høyere determinering (~77%). For å få enda mer nøyaktige prediksjoner kan vi ta med flere kategoriske variabler, samt numeriske funksjoner, som `Month` eller `DayOfYear`. For å få et stort array med funksjoner kan vi bruke `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Her tar vi også hensyn til `City` og `Package`-type, noe som gir oss MSE 2.84 (10%), og determinering 0.94!

## Sette det hele sammen

For å lage den beste modellen kan vi bruke kombinert (one-hot kodet kategorisk + numeriske) data fra eksempelet ovenfor sammen med polynomregresjon. Her er den komplette koden for din bekvemmelighet:

```python
# sett opp treningsdata
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# lag trenings-test deling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# sett opp og tren pipelinen
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prediker resultater for testdata
pred = pipeline.predict(X_test)

# beregn MSE og determinering
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Dette bør gi oss den beste determinering-koeffisienten på nesten 97%, og MSE=2.23 (~8% prediksjonsfeil).

| Modell | MSE | Determinering |
|--------|-----|---------------|
| `DayOfYear` Lineær | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynom | 2.73 (17.0%) | 0.08 |
| `Variety` Lineær | 5.24 (19.7%) | 0.77 |
| Alle funksjoner Lineær | 2.84 (10.5%) | 0.94 |
| Alle funksjoner Polynom | 2.23 (8.25%) | 0.97 |

🏆 Bra jobbet! Du laget fire regresjonsmodeller i én leksjon, og forbedret modellkvaliteten til 97%. I den siste delen om regresjon vil du lære om logistisk regresjon for å bestemme kategorier.

---
## 🚀Utfordring

Test flere forskjellige variable i denne notebooken for å se hvordan korrelasjon samsvarer med modellnøyaktighet.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudie

I denne leksjonen lærte vi om lineær regresjon. Det finnes andre viktige typer regresjon. Les om Stepwise, Ridge, Lasso og Elasticnet teknikker. Et godt kurs å studere for å lære mer er [Stanford Statistical Learning-kurset](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Oppgave

[Bygg en modell](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vennligst vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det opprinnelige dokumentet på originalspråket bør betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår fra bruken av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->