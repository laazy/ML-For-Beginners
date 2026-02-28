# Rakenna regressiomalli Scikit-learnillä: regressio neljällä tavalla

## Aloittelijan huomautus

Lineaarista regressiota käytetään, kun haluamme ennustaa **numeraalisen arvon** (esimerkiksi talon hinta, lämpötila tai myynti).
Se toimii löytämällä suoran viivan, joka parhaiten kuvaa syötteen ominaisuuksien ja tulosteen välisen suhteen.

Tässä oppitunnissa keskitymme käsitteen ymmärtämiseen ennen kuin tutustumme edistyneempiin regressiotekniikoihin.
![Lineaarinen vs polynominen regressio infograafi](../../../../translated_images/fi/linear-polynomial.5523c7cb6576ccab.webp)
> Infograafi tekijältä [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Esiluentokysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla R:llä!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Johdanto 

Tähän mennessä olet tutustunut regressioon näyteaineiston avulla, joka on poimittu kurpitsan hinnoitteluaineistosta ja jota käytämme koko oppitunnin ajan. Olet myös visualisoinut aineistoa Matplotlibillä.

Nyt olet valmis sukeltamaan syvemmälle ML:n regressioon. Visualisointi auttaa sinua ymmärtämään dataa, mutta todellinen koneoppimisen voima tulee _mallien opettamisesta_. Mallit opetetaan historiatiedolla, jotta ne automaattisesti tunnistavat datariippuvuuksia, ja ne mahdollistavat uusien, aiemmin näkemättömien tietojen tulosten ennustamisen.

Tässä oppitunnissa opit lisää kahdesta regressiotyypistä: _perus lineaarisesta regressiosta_ ja _polynomiregressiosta_, sekä joistakin näiden tekniikoiden taustalla olevasta matematiikasta. Nämä mallit mahdollistavat kurpitsahintojen ennustamisen eri syötetietojen perusteella.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videokatsauksen lineaarisesta regressiosta.

> Tämän opetussuunnitelman aikana oletamme hyvin vähäisiä matemaattisia ennakkotietoja ja pyrimme tekemään sisällöstä saavutettavaa eri alojen opiskelijoille, joten kiinnitä huomiota huomautuksiin, 🧮 kutsuihin, kaavioihin ja muihin oppimistyökaluihin, jotka tukevat ymmärrystä.

### Esivaatimus

Sinun tulisi nyt olla perehtynyt tutkimamme kurpitsadatan rakenteeseen. Sen löydät ennakkoladattuna ja esipuhdistettuna tämän oppitunnin _notebook.ipynb_-tiedostosta. Tiedostossa kurpitsan hinta esitetään busheliltä uudessa dataframessa. Varmista, että voit ajaa nämä notebookit Visual Studio Coden kernoissa.

### Valmistelu

Muistutuksena, lataat tätä dataa voidaksesi esittää sille kysymyksiä.

- Milloin on paras aika ostaa kurpitsoja? 
- Minkä hintaisen laatikon minikurpitsoja voin odottaa?
- Kannattaako ostaa puoli-bushelin koreissa vai 1 1/9 bushelin laatikoissa?
Tutkitaanpa dataa lisää.

Edellisessä oppitunnissa loit Pandas-dataframen ja täytit sen osalla alkuperäisestä aineistosta, standardoiden hinnat bushelin mukaan. Näin teit, mutta keräsit vain noin 400 datapistettä ja vain syys- ja loka-kuukausilta.

Katso tässä oppitunnissa mukana olevasta notebookista esiladattu data. Data on esiladattu ja alkuperäinen hajontakaavio on piirretty, joka näyttää kuukaustiedot. Voimme ehkä saada tarkemman kuvan datan luonteesta puhdistamalla sitä lisää.

## Lineaarinen regressioviiva

Kuten opit Oppitunnissa 1, lineaarisen regression tavoitteena on piirtää viiva, joka:

- **Näyttää muuttujien suhteet**. Näyttää muuttujien välisen suhteen
- **Tekee ennusteita**. Tekee tarkkoja ennusteita siitä, mihin uusi datapiste asettuu suhteessa viivaan.

On tyypillistä, että **vähintään neliöiden menetelmää** käytetään tämän tyyppisen viivan piirtämiseen. Termi "vähintään neliöiden menetelmä" viittaa mallin virheen kokonaismäärän minimointiin. Jokaisen datapisteen pystysuora etäisyys (jota kutsutaan residuaaliksi) mitataan todellisen pisteen ja regressioviivan välillä.

Neliöimme nämä etäisyydet kahdesta pääsyystä:

1. **Suuruus ilman suuntaa:** Haluamme käsitellä -5 virheen kuten +5 virhettä. Neliöiminen tekee kaikista arvoista positiivisia.

2. **Poikkeamien rankaisu:** Neliöiminen antaa suuremman painon suurille virheille, pakottaen viivan pysymään lähempänä kaukana olevia pisteitä.

Lisäämme sitten kaikki nämä neliöidyt arvot yhteen. Tavoitteemme on löytää tarkka viiva, jossa tämä summa on pienin mahdollinen – siksi nimi "vähintään neliöiden menetelmä".

> **🧮 Näytä matematiikka** 
> 
> Tämä viiva, jota kutsutaan _paras sovitusviivaksi_, voidaan esittää [yhtälöllä](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` on 'selittävä muuttuja'. `Y` on 'riippuva muuttuja'. Viivan kulmakerroin on `b` ja `a` on y-akselin leikkauspiste, joka kuvaa `Y`:n arvoa kun `X = 0`.
>
>![laske kulmakerroin](../../../../translated_images/fi/slope.f3c9d5910ddbfcf9.webp)
>
> Lasketaan ensin kulmakerroin `b`. Infograafi tekijältä [Jen Looper](https://twitter.com/jenlooper)
>
> Toisin sanoen ja viitaten kurpitsadatan alkuperäiseen kysymykseen: "ennustetaan kurpitsan hinta per bushel kuukauden mukaan", `X` viittaa hintaan ja `Y` myyntikuukauteen.
>
>![täydennä yhtälö](../../../../translated_images/fi/calculation.a209813050a1ddb1.webp)
>
> Lasketaan Y:n arvo. Jos maksat noin 4 dollaria, täytyy olla huhtikuu! Infograafi tekijältä [Jen Looper](https://twitter.com/jenlooper)
>
> Viivan laskennassa on näytettävä sen kaltevuus, joka riippuu myös leikkauskohdasta, eli siitä missä `Y` sijaitsee, kun `X = 0`.
>
> Voit tarkastella näiden arvojen laskennan menetelmää verkkosivulta [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Käy myös [tällä vähintään neliöiden laskimella](https://www.mathsisfun.com/data/least-squares-calculator.html) nähdäksesi kuinka lukuarvot vaikuttavat viivaan.

## Korrelatiivisuus

Yksi termi, joka on syytä ymmärtää, on **korrelaatiokerroin** tietyille X- ja Y-muuttujille. Hajontakuvion avulla voit nopeasti havaita tämän kertoimen. Kuvio, jossa datapisteet ovat siistissä viivassa, on korkea korrelaatio, mutta kun pisteet ovat hajallaan kaikkialla X:n ja Y:n välillä, korrelaatio on matala.

Hyvä lineaarinen regressiomalli on sellainen, jolla on korkea (lähellä 1:tä eikä 0:aa) korrelaatiokerroin, käytettäessä vähintään neliöiden menetelmää ja regressioviivaa.

✅ Suorita tämän oppitunnin mukana oleva notebook ja tarkastele Kuukausi vs. Hinta hajontakaaviota. Vaikuttaako kurpitsamyynnin Kuukauden ja Hinnan yhdistelmä korkealta vai matalalta korrelaatiolta, silmämääräisen tulkintasi perusteella? Muuttuuko tulos, jos käytät kuukauden sijaan yksityiskohtaisempaa mittaria, esim. *vuorokauden numeroa* (eli kuinka mones päivä vuodesta)?

Alla koodissa oletamme, että olemme puhdistaneet datan ja saaneet dataframen nimeltä `new_pumpkins`, joka muistuttaa seuraavaa:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Koodin dataa puhdistavista toimenpiteistä löydät tiedostosta [`notebook.ipynb`](notebook.ipynb). Olemme suorittaneet samat puhdistusvaiheet kuin edellisessä oppitunnissa, ja laskeneet `DayOfYear`-sarakkeen seuraavalla lausekkeella:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nyt kun ymmärrät lineaarisen regression matematiikan taustalla, luokaamme regressiomalli nähdäksesi, voimmeko ennustaa, mikä kurpitsapakkaus tarjoaa parhaat kurpitsahinnat. Joku, joka ostaa kurpitsoja juhlapuutarhaa varten, haluaa tämän tiedon optimoidakseen kurpitsapakkaustensa ostot.

## Etsimässä korrelaatiota

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videokatsauksen korrelaatiosta.

Edellisestä oppitunnista olet todennäköisesti nähnyt, että eri kuukausien keskimääräinen hinta näyttää tältä:

<img alt="Keskiarvo hinta kuukauden mukaan" src="../../../../translated_images/fi/barchart.a833ea9194346d76.webp" width="50%"/>

Tämä viittaa siihen, että jonkinlaista korrelaatiota pitäisi olla, ja voimme yrittää opettaa lineaarisen regressiomallin ennustamaan suhdetta `Month` ja `Price` välillä, tai `DayOfYear` ja `Price` välillä. Tässä on hajontakaavio jälkimmäisestä suhteesta:

<img alt="Hajontakaavio hinta vs. vuorokauden numero" src="../../../../translated_images/fi/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Katsotaan onko korrelaatiota käyttäen `corr`-funktiota:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Korrelaation näyttää olevan melko pieni, -0.15 kuukauden mukaan ja -0.17 vuorokauden numeron mukaan, mutta voi olla toinen tärkeä suhde. Näyttää siltä, että hintoja on eri ryhmiä eri kurpitsalajikkeille. Vahvistaaksemme tätä hypoteesia piirretään kukin kurpitsakategoria eri värillä. Käyttämällä `ax` parametria `scatter`-funktiossa voimme piirtää kaikki pisteet samaan kuvaan:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Hajontakaavio hinta vs. vuorokauden numero värikoodattu" src="../../../../translated_images/fi/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Tutkimuksemme viittaa siihen, että lajikkeella on suurempi vaikutus kokonaishintaan kuin varsinaisella myyntipäivällä. Näemme tämän pylväskaaviosta:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Pylväskaavio hinta vs. lajike" src="../../../../translated_images/fi/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Keskitytään hetkeksi vain yhteen kurpitsalajikkeeseen, 'pie type', ja katsotaan mikä vaikutus päivällä on hintaan:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Hajontakaavio Hinta vs Vuorokauden Numero" src="../../../../translated_images/fi/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Jos nyt lasketaan korrelaatio `Price` ja `DayOfYear` välillä `corr`-funktiolla, saamme noin `-0.27`, mikä tarkoittaa, että ennustemallin opettaminen on järkevää.

> Ennen lineaarisen regressiomallin opettamista on tärkeää varmistaa, että datamme on puhdasta. Lineaarinen regressio ei toimi hyvin puuttuvien arvojen kanssa, joten on järkevää poistaa kaikki tyhjät solut:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Toinen lähestymistapa voisi olla täyttää tyhjät arvot vastaavien sarakkeiden keskiarvoilla.

## Yksinkertainen lineaarinen regressio

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videokatsauksen lineaarisesta ja polynomisesta regressiosta.

Lineaarisen regressiomallin opettamiseen käytämme **Scikit-learn** kirjastoa.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Aloitamme erottamalla syötearvot (ominaisuudet) ja odotetun tuloksen (label) eri numpy-taulukoihin:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Huomaa, että jouduimme käyttämään `reshape`-toimintoa syötteessä, jotta LinearRegression kirjasto ymmärtää sen oikein. Lineaarinen regressio odottaa 2-ulotteista taulukkoa syötteenä, jossa jokainen rivi on yksi ominaisuusvektori. Koska meillä on vain yksi syöte, tarvitsemme N×1 muotoisen taulukon, jossa N on aineiston koko.

Seuraavaksi jaamme aineiston harjoitus- ja testidatoihin, jotta voimme validoida mallimme opettamisen jälkeen:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Viimein, mallin opettaminen tapahtuu kahdella rivillä koodia. Määrittelemme `LinearRegression`-olion ja sovitamme sen dataamme metodilla `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objekti sovittamisen (`fit`) jälkeen sisältää kaikki regressiokertoimet, joihin pääsee käsiksi `.coef_`-ominaisuudella. Tapauksessamme on vain yksi kerroin, joka on noin `-0.017`. Tämä tarkoittaa, että hinnat näyttävät laskevan hieman ajan myötä, mutta ei liikaa, noin 2 senttiä päivässä. Voimme myös päästä käsiksi regressioviivan leikkauspisteeseen Y-akselilla käyttämällä `lin_reg.intercept_` -arvoa – se on tapauksessamme noin `21`, mikä osoittaa hinnan vuoden alussa.

Nähdäksemme, kuinka tarkka mallimme on, voimme ennustaa hintoja testiaineistolla ja mitata, kuinka lähellä ennusteet ovat odotettuja arvoja. Tämä voidaan tehdä käyttäen keskineliövirheen (MSE) metriikkaa, joka on kaikkien odotettujen ja ennustettujen arvojen neliöllisten erotusten keskiarvo.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Virheemme näyttää olevan noin 2 pistettä, mikä on ~17%. Ei kovin hyvä. Toinen mallin laadun mittari on **selitysaste** (coefficient of determination), joka saadaan seuraavasti:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Jos arvo on 0, se tarkoittaa, että malli ei ota syötteitä huomioon ja toimii *huonoimpana lineaarisena ennustajana*, joka on yksinkertaisesti tuloksen keskiarvo. Arvo 1 tarkoittaa, että voimme täydellisesti ennustaa kaikki odotetut tulokset. Meidän tapauksessamme kerroin on noin 0.06, mikä on melko matala.

Voimme myös piirtää testidatan yhdessä regressioviivan kanssa nähdäksesi paremmin, miten regressio toimii tapauksessamme:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineaarinen regressio" src="../../../../translated_images/fi/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomiregressio

Toinen lineaarisen regression muoto on polynomiregressio. Vaikka joskus muuttujien välillä on lineaarinen suhde – mitä suurempi kurpitsan tilavuus, sitä korkeampi hinta – joskus näitä suhteita ei voi kuvata tasaisena tasona tai suorana viivana.

✅ Tässä on [lisää esimerkkejä](https://online.stat.psu.edu/stat501/lesson/9/9.8) aineistosta, johon voisi soveltaa polynomiregressiota

Katso uudelleen suhdetta Päivämäärän ja Hinnan välillä. Näyttääkö tämä hajontakuvio siltä, että sitä tulisi välttämättä analysoida suoralla viivalla? Eikö hinnat voi heilahdella? Tässä tapauksessa voit kokeilla polynomiregressiota.

✅ Polynomit ovat matemaattisia lausekkeita, jotka voivat sisältää yhden tai useamman muuttujan ja kertoimia

Polynomiregressio luo kaarevan käyrän sovittamaan paremmin epälineaarista dataa. Jos mukaan otetaan toisen asteen `DayOfYear`-muuttuja syöttötietoihin, pystymme sovittamaan dataamme paraabelin muotoisen käyrän, jolla on minimi tietyn pisteen kohdalla vuoden aikana.

Scikit-learn sisältää hyödyllisen [pipeline-rajapinnan](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) yhdistämään erilaiset data-analyysin vaiheet. **Pipeline** on ketju **estimaattoreita**. Tapauksessamme luomme pipeline:n, joka ensin lisää polynomiset ominaisuudet malliin ja sitten kouluttaa regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)` tarkoittaa, että mukaan otetaan kaikki toisen asteen polynomit syötedatasta. Meidän tapauksessamme se tarkoittaa vain `DayOfYear`<sup>2</sup>, mutta jos meillä olisi kaksi syöte muuttujaa X ja Y, mukaan tulisi X<sup>2</sup>, XY ja Y<sup>2</sup>. Voimme myös käyttää korkeampiasteisia polynomeja, jos haluamme.

Pipelinetä voi käyttää samalla tavalla kuin alkuperäistä `LinearRegression`-objektia, eli voimme `fit`-metodilla sovittaa ja `predict`-metodilla ennustaa tuloksia. Tässä on kuvaaja, joka näyttää testidatan ja approksimaatiokäyrän:

<img alt="Polynomiregressio" src="../../../../translated_images/fi/poly-results.ee587348f0f1f60b.webp" width="50%" />

Polynomiregressiolla saamme hieman pienemmän MSE:n ja korkeamman selitysasteen, mutta ei merkittävästi. Meidän täytyy ottaa mukaan muitakin piirteitä!

> Voit nähdä, että minimihinnat kurpitsoille ovat jossain halloweenin tienoilla. Miten tätä voisi selittää?

🎃 Onneksi olkoon, loit juuri mallin, joka auttaa ennustamaan kurpitsapiirakan hintoja. Voisit luultavasti toistaa saman prosessin kaikille kurpitsatyypeille, mutta se olisi työlästä. Opitaan nyt, miten ottaa kurpitsan lajike huomioon mallissa!

## Kategoriset ominaisuudet

Ihannetilanteessa haluamme pystyä ennustamaan eri kurpitsalajikkeiden hintoja samalla mallilla. Kuitenkin `Variety`-sarake eroaa esimerkiksi `Month`-sarakkeesta, koska se sisältää ei-numeerisia arvoja. Tällaisia sarakkeita kutsutaan **kategorisiksi**.

[![ML aloittelijoille - Kategoriset ominaisuudet ja lineaarinen regressio](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML aloittelijoille - Kategoriset ominaisuudet ja lineaarinen regressio")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videokatsauksen kategoristen ominaisuuksien käytöstä.

Tässä näet, miten keskimääräinen hinta riippuu lajikkeesta:

<img alt="Keskiarvohinta lajikkeittain" src="../../../../translated_images/fi/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Ottaaksemme lajikkeen huomioon, meidän täytyy ensin muuntaa se numeromuotoon eli **koodata** se. On olemassa muutamia tapoja tehdä tämä:

* Yksinkertainen **numeroarvokoodaus** luo taulukon eri lajikkeista ja korvaa lajikenimen kyseisen taulukon indeksillä. Tämä ei ole paras idea lineaarisessa regressiossa, koska lineaariregressio ottaa indeksin numeerisen arvon eikä huomioi, että indeksin numeerinen arvo ei välttämättä ole lineaarisesti yhteydessä hintaan. Toisin sanoen, lineaarisessa mallissa kerroin kerrotaan indeksiluvulla, mikä ei vastaa todellista hintasuhdetta.
* **One-hot-koodaus** korvaa `Variety`-sarakkeen neljällä eri sarakkeella, yhden kullekin lajikkeelle. Kukin sarake sisältää `1`, jos rivi vastaa kyseistä lajiketta, ja `0` muuten. Tämä tarkoittaa, että lineaariregressiossa on neljä kerrointa, yksi jokaiselle kurpitsalajikkeelle, jotka kuvaavat kunkin lajikkeen "aloitushintaa" (tai pikemminkin "lisähintaa").

Alla oleva koodi näyttää, miten lajike voidaan one-hot-koodata:

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

Jotta voimme kouluttaa lineaariregression käyttäen one-hot-koodattua lajiketta syötteenä, meidän pitää vain alustaa `X` ja `y` oikein:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Muu koodi on sama kuin aiemmin käyttämämme lineaarisen regression koulutuksessa. Kun kokeilet, huomaat, että keskineliövirhe on suunnilleen sama, mutta selitysaste nousee huomattavasti (~77%). Tarkempia ennusteita saamme ottamalla huomioon lisää kategorisia piirteitä sekä numeerisia ominaisuuksia, kuten `Month` tai `DayOfYear`. Yhden suuren ominaisuustaulukon saamiseksi voimme käyttää `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tässä otamme myös huomioon `City`- ja `Package`-tyypin, mikä antaa meille MSE-arvoksi 2.84 (10%) ja selitysasteeksi 0.94!

## Kaiken yhdistäminen

Parhaan mallin saamiseksi voimme käyttää yhdistettyä (one-hot-koodattua kategorista + numeerista) dataa ylläolevasta esimerkistä yhdessä polynomiregression kanssa. Tässä on koko koodi vaivattomuutta varten:

```python
# aseta harjoitusdata
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# tee opetus- ja testijako
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# määritä ja kouluta putkisto
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# ennusta tulokset testidatalle
pred = pipeline.predict(X_test)

# laske MSE ja selitysaste
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Tämä antaa meille parhaan selitysasteen lähes 97 % ja MSE:n 2.23 (~8 % ennustevirhe).

| Malli | MSE | Selitysaste |
|-------|-----|-------------|
| `DayOfYear` lineaarinen | 2.77 (17.2%) | 0.07 |
| `DayOfYear` polynomi | 2.73 (17.0%) | 0.08 |
| `Variety` lineaarinen | 5.24 (19.7%) | 0.77 |
| Kaikki ominaisuudet lineaarinen | 2.84 (10.5%) | 0.94 |
| Kaikki ominaisuudet polynomi | 2.23 (8.25%) | 0.97 |

🏆 Hienoa työtä! Loit neljä regressiomallia yhdessä oppitunnissa ja paransit mallin laatua 97 prosenttiin. Regressio-aiheen lopussa opit logistisesta regressiosta luokitteluja varten.

---
## 🚀Haaste

Testaa eri muuttujia tässä muistikirjassa nähdäksesi, miten korrelaatio vastaa mallin tarkkuutta.

## [Luentojälkeinen tietovisa](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itsenäinen opiskelu

Tässä oppitunnissa opimme lineaarisesta regressiosta. On olemassa myös muita tärkeitä regressiotyyppejä. Lue lisää Stepwise-, Ridge-, Lasso- ja Elasticnet-menetelmistä. Hyvä kurssi opinnoille on [Stanfordin tilastollisen oppimisen kurssi](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tehtävä

[Rakenna malli](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Pyrimme tarkkuuteen, mutta huomioithan, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suosittelemme ammattimaista ihmiskäännöstä. Emme ota vastuuta tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai virheellisistä tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->