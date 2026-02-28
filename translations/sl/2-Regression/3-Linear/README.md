# Ustvarjanje regresijskega modela z uporabo Scikit-learn: regresija na štiri načine

## Opomba za začetnike

Linearna regresija se uporablja, kadar želimo napovedati **številsko vrednost** (na primer cena hiše, temperatura ali prodaja).
Deluje tako, da najde ravno črto, ki najbolje predstavlja razmerje med vhodnimi značilnostmi in izhodom.

V tej lekciji se osredotočamo na razumevanje koncepta, preden raziskujemo bolj napredne regresijske tehnike.
![Linearna proti polinomska regresija infografika](../../../../translated_images/sl/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Predpredavalni kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcija je na voljo v R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Uvod

Do zdaj ste raziskali, kaj je regresija na podlagi vzorčnih podatkov iz nabora podatkov o cenah buč, ki jih bomo uporabljali skozi to lekcijo. Prav tako ste jih vizualizirali z uporabo Matplotlib.

Zdaj ste pripravljeni, da se poglobite v regresijo za ML. Medtem ko vizualizacija omogoča razumevanje podatkov, prava moč strojnega učenja izvira iz _usposabljanja modelov_. Modele usposabljamo na zgodovinskih podatkih, da samodejno zajamejo odvisnosti podatkov, in omogočajo napovedovanje izidov za nove podatke, ki jih model še ni videl.

V tej lekciji boste izvedeli več o dveh vrstah regresije: _osnovni linearni regresiji_ in _polinomski regresiji_, skupaj z nekaj matematike, ki je osnova teh tehnik. Ti modeli nam bodo omogočili napovedovanje cen buč glede na različne vhodne podatke.

[![ML za začetnike - Razumevanje linearne regresije](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML za začetnike - Razumevanje linearne regresije")

> 🎥 Kliknite na sliko zgoraj za kratek video pregled linearne regresije.

> V celotnem učnem načrtu predvidevamo minimalno znanje matematike in želimo, da je dostopen študentom iz drugih področij, zato bodite pozorni na opombe, 🧮 klice, diagrame in druge učne pripomočke, ki pomagajo pri razumevanju.

### Predpogoj

Ugotoviti bi morali, da se seznanjate s strukturo podatkov o bučah, ki jih preučujemo. Najdete jih naložene in očiščene v datoteki _notebook.ipynb_ te lekcije. V tej datoteki je cena buče prikazana na mero bushela v novi podatkovni tabeli. Prepričajte se, da lahko te zvezke zaženete v okolju Visual Studio Code.

### Priprava

Za opomnik, ta podatke nalagate zato, da bi lahko zastavljali vprašanja o njih.

- Kdaj je najboljši čas za nakup buč?
- Kakšno ceno lahko pričakujem za škatlo mini buč?
- Naj jih kupim v košarah za pol bushela ali v škatlah za 1 1/9 bushel?
Poglejmo torej podrobneje v te podatke.

V prejšnji lekciji ste ustvarili Pandasovo podatkovno tabelo in jo napolnili z delom izvirnega nabora podatkov ter standardizirali cene glede na mero bushela. S tem pa ste pridobili le približno 400 podatkovnih točk in samo za jesenske mesece.

Poglejte si podatke, ki smo jih prednaložili v spremljajočem zvezku za to lekcijo. Podatki so prednaloženi, začetni razpršeni graf pa prikazuje mesečne podatke. Morda lahko z dodatnim čiščenjem pridobimo več podrobnosti o naravi podatkov.

## Linearna regresijska črta

Kot ste se naučili v Lekciji 1, je cilj linearne regresije narisati črto, ki:

- **Prikazuje odnose med spremenljivkami**. Prikaže povezavo med spremenljivkami
- **Naredi napovedi**. Accurate napove, kje bo nova podatkovna točka glede na to črto.

Običajno za **najmanjše kvadrate regresije** narišemo tovrstno črto. Izraz "Najmanjši kvadrati" se nanaša na proces minimizacije skupne napake v našem modelu. Za vsako podatkovno točko izmerimo vertikalno razdaljo (katerakoli ostanek) med dejansko točko in regresijsko črto.

Te razdalje kvadriramo iz dveh glavnih razlogov:

1. **Velikost nad smerjo:** Želimo, da je napaka -5 enaka napaki +5. Kvadriranje pretvori vse vrednosti v pozitivne.

2. **Kaznovanje odstopanj:** Kvadriranje daje večjo težo večjim napakam, zaradi česar mora črta ostati bližje točkam, ki so daleč stran.

Nato seštejemo vse te kvadrirane vrednosti. Naš cilj je najti točno tisto linijo, kjer je ta končni vsota najmanjša (najmanjša možna vrednost)—od tod ime "najmanjši kvadrati".

> **🧮 Pokaži mi matematiko**
> 
> Ta črta, imenovana _črta najboljšega prileganja_, je izražena z [enačbo](https://en.wikipedia.org/wiki/Simple_linear_regression):
>
> ```
> Y = a + bX
> ```
>
> `X` je 'razlagaška spremenljivka'. `Y` je 'odvisna spremenljivka'. Naklon črte je `b`, `a` pa je y-presek, ki označuje vrednost `Y`, ko je `X = 0`.
>
>![izračun naklona](../../../../translated_images/sl/slope.f3c9d5910ddbfcf9.webp)
>
> Najprej izračunaj naklon `b`. Infografika avtorja [Jen Looper](https://twitter.com/jenlooper)
>
> Z drugimi besedami, in se sklicujoč na prvotno vprašanje glede naših podatkov o bučah: "napovedati ceno buče na bushel glede na mesec", bi `X` predstavljal ceno, `Y` pa mesec prodaje.
>
>![dopolni enačbo](../../../../translated_images/sl/calculation.a209813050a1ddb1.webp)
>
> Izračunaj vrednost Y. Če plačuješ okoli 4 $, mora biti april! Infografika avtorja [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika, ki izračuna črto, mora upoštevati naklon črte, ki je odvisen tudi od preseka, torej kje je `Y`, ko je `X = 0`.
>
> Metode za izračun teh vrednosti si lahko ogledate na spletni strani [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Prav tako obiščite [ta kalkulator najmanjših kvadratov](https://www.mathsisfun.com/data/least-squares-calculator.html), kjer lahko vidite, kako vrednosti števil vplivajo na črto.

## Korelacija

Še en pojem, ki ga je treba razumeti, je **koeficient korelacije** med danima spremenljivkama X in Y. Z uporabo razpršenega grafa lahko hitro vizualizirate ta koeficient. Graf s podatkovnimi točkami, razporejenimi v lepo črto, ima visoko korelacijo, medtem ko graf s podatkovnimi točkami, raztresenimi povsod med X in Y, ima nizko korelacijo.

Dober linearni regresijski model je tisti, ki ima visok (bližji 1 kot 0) koeficient korelacije, izračunan z metodo najmanjših kvadratov s črto regresije.

✅ Zaženite zvezek, ki spremlja to lekcijo, in si oglejte razpršeni graf Mesecev proti Ceni. Ali se zdi, da ima podatek, ki povezuje mesec s ceno pri prodaji buč, visoko ali nizko korelacijo glede na vašo vizualno interpretacijo razpršenega grafa? Ali se kaj spremeni, če uporabite bolj natančno meritev namesto `Month`, npr. *dan v letu* (to je število dni od začetka leta)?

V spodnji kodi bomo predpostavili, da smo podatke očistili in pridobili podatkovno tabelo z imenom `new_pumpkins`, podobno spodnjemu:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Koda za čiščenje podatkov je na voljo v [`notebook.ipynb`](notebook.ipynb). Izvedli smo enake korake čiščenja kot v prejšnji lekciji in izračunali stolpec `DayOfYear` z naslednjim izrazom:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Zdaj, ko imate razumevanje matematike za linearno regresijo, ustvarimo regresijski model, da vidimo, ali lahko napovemo, katera embalaža buč bo imela najboljše cene. Nekdo, ki kupuje buče za praznični bučni vrt, bi želel te informacije, da bi lahko optimiziral svoj nakup.

## Iskanje korelacije

[![ML za začetnike - Iskanje korelacije: ključ linearne regresije](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML za začetnike - Iskanje korelacije: ključ linearne regresije")

> 🎥 Kliknite na sliko zgoraj za kratek video pregled korelacije.

Iz prejšnje lekcije ste verjetno videli, da povprečna cena v različnih mesecih izgleda tako:

<img alt="Povprečna cena po mesecih" src="../../../../translated_images/sl/barchart.a833ea9194346d76.webp" width="50%"/>

To kaže, da mora obstajati neka korelacija in lahko poskusimo usposobiti linearni regresijski model za napovedovanje odnosa med `Month` in `Price` ali med `DayOfYear` in `Price`. Tukaj je razpršeni graf, ki prikazuje slednji odnos:

<img alt="Razpršeni graf Price proti Day of Year" src="../../../../translated_images/sl/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Poglejmo, ali obstaja korelacija z uporabo funkcije `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Zdi se, da je korelacija precej majhna, -0,15 glede na `Month` in -0,17 glede na `DayOfMonth`, lahko pa obstaja druga pomembna povezava. Zdi se, da so različni grozdi cen, ki ustrezajo različnim sortam buč. Da potrdimo to hipotezo, narišimo vsako kategorijo buč z drugo barvo. Tako da podamo parameter `ax` funkciji `scatter` lahko vse točke narišemo v isti graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Razpršeni graf Price proti Day of Year s barvami" src="../../../../translated_images/sl/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Naše raziskave nakazujejo, da ima sorta večji vpliv na skupno ceno kot dejanski datum prodaje. To lahko vidimo z stolpičnim grafom:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stolpični graf cen glede na sorto buče" src="../../../../translated_images/sl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Za zdaj se osredotočimo samo na eno sorto buč, 'pie type', in poglejmo, kakšen vpliv ima datum na ceno:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Razpršeni graf Price proti Day of Year za sorto pie type" src="../../../../translated_images/sl/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Če zdaj izračunamo korelacijo med `Price` in `DayOfYear` z uporabo funkcije `corr`, bomo dobili nekaj takega kot `-0.27` – kar pomeni, da smiselno trenirati napovedni model.

> Preden usposobimo linearen regresijski model, je pomembno zagotoviti, da so naši podatki čisti. Linearna regresija ne deluje dobro z manjkajočimi vrednostmi, zato je smiselno odstraniti vse prazne celice:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Druga možnost je, da te prazne vrednosti zapolnimo s povprečnimi vrednostmi iz ustreznega stolpca.

## Enostavna linearna regresija

[![ML za začetnike - Linearna in polinomska regresija z uporabo Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML za začetnike - Linearna in polinomska regresija z uporabo Scikit-learn")

> 🎥 Kliknite na sliko zgoraj za kratek video pregled linearne in polinomske regresije.

Za usposabljanje našega modela linearne regresije bomo uporabili knjižnico **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Začnemo tako, da ločimo vhodne vrednosti (značilnosti) in pričakovani izhod (oznako) v ločene numpy polja:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Upoštevajte, da smo morali na vhodne podatke uporabiti `reshape`, da jih je paket LinearRegression pravilno razumel. Linearna regresija namreč pričakuje vhod v obliki 2D polja, kjer vsak vrstični vektor predstavlja vektor vhodnih značilnosti. V našem primeru, ker imamo samo en vhod, potrebujemo polje oblike N&times;1, kjer je N velikost nabora podatkov.

Nato moramo podatke razdeliti na učni in testni sklop, da lahko model preverimo po usposabljanju:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Na koncu usposabljanje modela linearne regresije traja le dve vrstici kode. Definiramo objekt `LinearRegression` in ga prilagodimo našim podatkom z metodo `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` po prilagoditvi (`fit`) vsebuje vse koeficiente regresije, do katerih lahko dostopamo z lastnostjo `.coef_`. V našem primeru je samo en koeficient, ki bi moral biti okoli `-0.017`. To pomeni, da se cene zdi, da nekoliko padajo s časom, vendar ne preveč, približno 2 centa na dan. Do presečišča regresije z Y-osjo lahko dostopamo tudi z `lin_reg.intercept_` - v našem primeru bo okoli `21`, kar daje informacijo o ceni na začetku leta.

Za preverjanje točnosti našega modela lahko napovemo cene na testnem naboru podatkov in nato merimo, kako blizu so naše napovedi pričakovanim vrednostim. To lahko storimo z merjenjem srednje kvadratne napake (MSE), kar je srednja vrednost vseh kvadratov razlik med pričakovano in napovedano vrednostjo.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Naša napaka je okoli 2 točki, kar je ~17 %. Ni ravno dobro. Drug indikator kakovosti modela je **koeficient determinacije**, ki ga lahko pridobimo na naslednji način:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Če je vrednost 0, pomeni, da model ne upošteva vhodnih podatkov in deluje kot *najslabši linearni napovedovalec*, ki je preprosto povprečna vrednost rezultata. Vrednost 1 pomeni, da lahko popolnoma napovemo vse pričakovane izide. V našem primeru je koeficient okoli 0.06, kar je precej nizko.

Testne podatke lahko tudi narišemo skupaj z regresijsko premico, da bolje vidimo, kako regresija deluje v našem primeru:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/sl/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polinomska regresija

Druga vrsta linearne regresije je polinomska regresija. Čeprav obstaja včasih linearna povezava med spremenljivkami - večja kot je buča po prostornini, višja je cena - včasih teh povezav ni mogoče prikazati kot ravnino ali ravno črto.

✅ Tu je [še nekaj primerov](https://online.stat.psu.edu/stat501/lesson/9/9.8) podatkov, kjer bi lahko uporabili polinomsko regresijo.

Oglejmo si še enkrat odnos med Datumom in Ceno. Ali ta diagram razpršitve nujno kaže, da se mora analizirati z ravno črto? Ali cene ne morejo nihati? V tem primeru lahko poskusimo polinomsko regresijo.

✅ Polinomi so matematični izrazi, ki so lahko sestavljeni iz ene ali več spremenljivk in koeficientov.

Polinomska regresija ustvari ukrivljeno črto, da bolje ustreza nelinearnim podatkom. V našem primeru, če vključimo kvadratno variablo `DayOfYear`, bi morali podatke prilagoditi paraboli, ki bo imela minimum v določenem trenutku v letu.

Scikit-learn vključuje uporabno [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) za združevanje različnih korakov obdelave podatkov skupaj. **Pipeline** je veriga **ocenjevalcev**. V našem primeru bomo ustvarili pipeline, ki najprej doda polinomske značilnosti modelu, nato pa izvede učenje regresije:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Uporaba `PolynomialFeatures(2)` pomeni, da bomo vključili vse polinome drugega reda iz vhodnih podatkov. V našem primeru to pomeni samo `DayOfYear`<sup>2</sup>, vendar če imamo dve vhodni spremenljivki X in Y, bo to dodalo X<sup>2</sup>, XY in Y<sup>2</sup>. Uporabimo lahko tudi poljuben višji red polinomov, če želimo.

Pipepline lahko uporabljamo na enak način kot izvirni objekt `LinearRegression`, torej lahko na pipeline uporabimo `fit`, nato pa uporabimo `predict` za pridobitev rezultatov napovedi. Tu je graf, ki prikazuje testne podatke in približno krivuljo:

<img alt="Polynomial regression" src="../../../../translated_images/sl/poly-results.ee587348f0f1f60b.webp" width="50%" />

Z uporabo polinomske regresije lahko dosežemo nekoliko nižjo vrednost MSE in višji koeficient determinacije, vendar ne bistveno. Potrebno je upoštevati tudi druge značilnosti!

> Vidite, da so minimalne cene buč opazne nekje okoli noči čarovnic. Kako bi to razložili? 

🎃 Čestitamo, ustvarili ste model, ki lahko napove ceno buč za pito. Verjetno lahko isto proceduro ponovite za vse vrste buč, ampak to bi bilo zamudno. Naučimo se zdaj, kako upoštevati vrsto buče v našem modelu!

## Kategorikalne značilnosti

V idealnem svetu želimo napovedovati cene za različne vrste buč z uporabo istega modela. Vendar pa je stolpec `Variety` nekoliko drugačen od stolpcev, kot je `Month`, ker vsebuje nenumerične vrednosti. Takim stolpcem pravimo **kategorikalni**.

[![ML za začetnike - Napovedi kategorialnih značilnosti z linearno regresijo](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML za začetnike - Napovedi kategorialnih značilnosti z linearno regresijo")

> 🎥 Kliknite sliko zgoraj za kratek video pregled uporabe kategorikalnih značilnosti.

Tukaj si lahko ogledate, kako povprečna cena zavisi od vrste:

<img alt="Average price by variety" src="../../../../translated_images/sl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Za upoštevanje vrste moramo najprej pretvoriti podatke v številčno obliko, oziroma jih **kodirati**. Obstaja več načinov za to:

* Preprosta **numerična koda** bo sestavila tabelo različnih vrst, nato pa zamenjala ime vrstice z indeksom v tej tabeli. To ni najboljša ideja za linearno regresijo, ker linearna regresija vzame dejansko numerično vrednost indeksa in jo doda rezultatu, pomnoženo z nekim koeficientom. V našem primeru je odnos med številom indeksa in ceno očitno nelinearen, tudi če zagotovimo, da so indeksi urejeni na določen način.
* **One-hot kodiranje** bo stolpec `Variety` nadomestilo s 4 različnimi stolpci, po enim za vsako vrsto. Vsak stolpec bo vseboval `1`, če je ustrezni vrstici določena vrsta, in `0` sicer. To pomeni, da bo v linearni regresiji štiri koeficiente, po enega za vsako vrsto buče, ki bo odgovoren za "začetno ceno" (ali bolj "dodatno ceno") za določeno vrsto.

Spodnja koda prikazuje, kako lahko one-hot kodiramo vrsto:

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

Za učenje linearne regresije z one-hot kodirano vrsto kot vhodom moramo samo pravilno pripraviti podatka `X` in `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Preostanek kode je enak tisti, ki smo jo uporabili prej za učenje linearne regresije. Če to poskusite, boste videli, da je srednja kvadratna napaka približno enaka, vendar bomo dobili precej višji koeficient determinacije (~77 %). Za še natančnejše napovedi lahko upoštevamo več kategorikalnih značilnosti ter tudi numerične, kot sta `Month` ali `DayOfYear`. Za združitev vseh značilnosti v eno veliko tabelo lahko uporabimo `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tukaj upoštevamo tudi `City` in tip `Package`, kar nam da MSE 2.84 (10 %) in koeficient determinacije 0.94!

## Vse skupaj

Za najboljši model lahko uporabimo združene (one-hot kodirane kategorikalne in numerične) podatke iz zgornjega primera skupaj s polinomsko regresijo. Tukaj je celotna koda za vašo udobje:

```python
# nastavi učne podatke
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# naredi razdelitev na učno in testno množico
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# nastavi in izuči cevovod
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# napovej rezultate za testne podatke
pred = pipeline.predict(X_test)

# izračunaj MSE in koeficient determinacije
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

To nam mora dati najboljši koeficient determinacije skoraj 97 % in MSE=2.23 (~8 % napake napovedi).

| Model | MSE | Koeficient determinacije |
|-------|-----|--------------------------|
| Linearni `DayOfYear` | 2.77 (17,2 %) | 0.07 |
| Polinomski `DayOfYear` | 2.73 (17,0 %) | 0.08 |
| Linearni `Variety` | 5.24 (19,7 %) | 0.77 |
| Linearni - vse značilnosti | 2.84 (10,5 %) | 0.94 |
| Polinomski - vse značilnosti | 2.23 (8,25 %) | 0.97 |

🏆 Odlično! V eni lekciji ste ustvarili štiri regresijske modele in izboljšali kakovost modela na 97 %. V zadnjem delu o regresiji se boste naučili o logistični regresiji za določanje kategorij.

---
## 🚀Izazov

Preizkusite več različnih spremenljivk v tej zvezki, da vidite, kako korelacija ustreza natančnosti modela.

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

V tej lekciji smo se naučili o linearni regresiji. Obstajajo še druge pomembne vrste regresije. Preberite o tehnikah Stepwise, Ridge, Lasso in Elasticnet. Dober tečaj za nadaljnje učenje je [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Naloga

[Ustvari model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Omejitev odgovornosti**:
Ta dokument je bil preveden z uporabo AI prevajalske storitve [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da avtomatizirani prevodi lahko vsebujejo napake ali netočnosti. Izvirni dokument v njegovem maternem jeziku velja za avtoritativni vir. Za ključne informacije priporočamo strokovni človeški prevod. Nismo odgovorni za kakršnekoli nesporazume ali napačne interpretacije, ki izhajajo iz uporabe tega prevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->