# Vytvorte regresný model pomocou Scikit-learn: štyri spôsoby regresie

## Poznámka pre začiatočníkov

Lineárna regresia sa používa, keď chceme predpovedať **číselnú hodnotu** (napríklad cenu domu, teplotu alebo predaj).
Funguje tak, že nájde priamku, ktorá najlepšie reprezentuje vzťah medzi vstupnými premennými a výstupom.

V tejto lekcii sa zameriavame na pochopenie konceptu predtým, než preskúmame pokročilejšie regresné techniky.
![Lineárna vs polynomiálna regresia infografika](../../../../translated_images/sk/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná aj v R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Úvod

Doteraz ste preskúmali, čo je regresia, pomocou vzorových údajov zo súboru údajov o cene tekvíc, ktorý budeme používať počas celej tejto lekcie. Tiež ste ich vizualizovali pomocou Matplotlib.

Teraz ste pripravení ponoriť sa hlbšie do regresie pre ML. Zatiaľ čo vizualizácia umožňuje lepšie pochopiť údaje, skutočná sila strojového učenia pochádza z _trénovania modelov_. Modely sa trénujú na historických dátach, aby automaticky zachytili závislosti v dátach, a umožňujú vám predpovedať výsledky pre nové dáta, ktoré model predtým nevidel.

V tejto lekcii sa dozviete viac o dvoch typoch regresie: _základnej lineárnej regresii_ a _polynomiálnej regresii_ spolu s niektorou z matematiky, ktorá stojí za týmito technikami. Tieto modely nám umožnia predpovedať ceny tekvíc v závislosti od rôznych vstupných údajov.

[![ML pre začiatočníkov - Pochopenie lineárnej regresie](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pre začiatočníkov - Pochopenie lineárnej regresie")

> 🎥 Kliknite na obrázok vyššie pre krátke video o lineárnej regresii.

> Počas celého kurikula predpokladáme minimálne znalosti matematiky a snažíme sa ich sprístupniť študentom z iných odborov, tak sledujte poznámky, 🧮 upozornenia, diagramy a ďalšie výučbové pomôcky na lepšie pochopenie.

### Predpoklady

Teraz by ste už mali byť oboznámení so štruktúrou údajov o tekviciach, ktoré skúmame. Nájdete ich prednačítané a predvyčistené v súbore _notebook.ipynb_ tejto lekcie. V súbore je cena tekvíc uvedená za košík v novom dátovom rámci. Uistite sa, že viete spustiť tieto notebooky v kerneloch vo Visual Studio Code.

### Príprava

Ako pripomienku, načítavate tieto údaje, aby ste na nich mohli klásť otázky.

- Kedy je najlepší čas na nákup tekvíc? 
- Akú cenu môžem očakávať za balenie mini tekvíc?
- Mám ich kúpiť v polkošíkoch alebo v krabici 1 1/9 košíka?
Poďme sa ďalej ponoriť do tohto dátového súboru.

V predchádzajúcej lekcii ste vytvorili dátový rámec Pandas a naplnili ho časťou pôvodného datasetu, štandardizujúc ceny podľa košíka. Týmto ste však získali iba asi 400 dátových bodov a len pre jesenné mesiace.

Pozrite si údaje, ktoré sme prednačítali v sprievodnom notebooku tejto lekcie. Údaje sú predpripravené a prvý rozptýlený graf ukazuje mesiac predaja. Možno pôjdeme ďalej a vyčistíme dáta podrobnejšie.

## Lineárna regresná čiara

Ako ste sa naučili v Lekcii 1, cieľom lineárnej regresie je nakresliť čiaru, ktorá:

- **Ukazuje vzťah medzi premennými**. Ukáže vzťah medzi premennými
- **Predpovedá**. Umožní presne predpovedať, kde by nový dátový bod ležal vzhľadom na túto čiaru.

Typické na **regresii metódou najmenších štvorcov** je kreslenie takéhoto druhu čiary. Termín "najmenšie štvorce" označuje proces minimalizácie celkovej chyby v našom modeli. Pre každý dátový bod meriame vertikálnu vzdialenosť (nazývanú rezíduum) medzi skutočným bodom a regresnou čiarou.

Tieto vzdialenosti umocňujeme na druhú pre dva hlavné dôvody:

1. **Veľkosť pred smerom:** Chceme, aby chyba -5 bola rovnocenná chybe +5. Umocnenie na druhú zmení všetky hodnoty na kladné.

2. **Trestenie odľahlých hodnôt:** Umocnenie na druhú dáva väčšiu váhu väčším chybám, núti čiaru zostať bližšie k bodom, ktoré sú vzdialené.

Tieto umocnené hodnoty potom sčítame. Naším cieľom je nájsť čiaru, kde výsledný súčet bude čo najmenší (najmenšia možná hodnota) — odtiaľ názov "najmenšie štvorce".

> **🧮 Ukáž mi matematiku** 
> 
> Táto čiara, nazývaná _čiara najlepšieho prispôsobenia_, môže byť vyjadrená [rovnicou](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'vysvetľujúca premenná'. `Y` je 'závislá premenná'. Sklon čiary je `b` a `a` je y-priesečník, ktorý predstavuje hodnotu `Y` pre `X = 0`. 
>
>![vypočítajte sklon](../../../../translated_images/sk/slope.f3c9d5910ddbfcf9.webp)
>
> Najprv vypočítajte sklon `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Inými slovami a odkazujúc na pôvodnú otázku našich údajov o tekviciach: "predpovedať cenu tekvice za košík podľa mesiaca", `X` by predstavoval cenu a `Y` by označoval mesiac predaja.
>
>![dokončite rovnicu](../../../../translated_images/sk/calculation.a209813050a1ddb1.webp)
>
> Vypočítajte hodnotu Y. Ak platíte okolo 4 dolárov, musí to byť apríl! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika, ktorá počíta čiaru, musí ukázať sklon čiary, ktorý závisí aj od priesečníka, teda kde sa `Y` nachádza, keď `X = 0`.
>
> Metódu výpočtu týchto hodnôt môžete vidieť na webovej stránke [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Navštívte tiež [tento Least-squares kalkulátor](https://www.mathsisfun.com/data/least-squares-calculator.html), aby ste videli, ako hodnoty čísel ovplyvňujú čiaru.

## Korelácia

Ešte jeden termín, ktorý je dobré pochopiť, je **Korelačný koeficient** medzi danými premennými X a Y. Pomocou rozptýleného grafu môžete rýchlo vizualizovať tento koeficient. Graf, kde sú body rozptýlené pozdĺž čistej čiary, má vysokú koreláciu, zatiaľ čo graf, kde sú body rozptýlené všade medzi X a Y, má nízku koreláciu.

Dobrý lineárny regresný model bude taký, ktorý má vysoký (bližšie k 1 než k 0) Korelačný koeficient použitím metódy najmenších štvorcov s regresnou čiarou.

✅ Spustite notebook sprevádzajúci túto lekciu a pozrite sa na rozptýlený graf Mesiac k Cene. Zdá sa vám, že dáta spájajúce Mesiac s Cenou predaja tekvíc majú vysokú alebo nízku koreláciu podľa vašej vizuálnej interpretácie rozptýleného grafu? Zmení sa to, ak namiesto `Month` použijete detailnejšie meranie, napríklad *deň v roku* (t.j. počet dní od začiatku roka)?

Nižšie v kóde predpokladáme, že sme údaje vyčistili a získali dátový rámec nazvaný `new_pumpkins`, podobný nasledovnému:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kód na vyčistenie dát je dostupný v [`notebook.ipynb`](notebook.ipynb). Vykonali sme rovnaké čistiace kroky ako v predchádzajúcej lekcii a vypočítali stĺpec `DayOfYear` pomocou nasledujúceho výrazu:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Keďže už rozumiete matematike za lineárnou regresiou, vytvorme regresný model, aby sme zistili, či vieme predpovedať, ktoré balenie tekvíc bude mať najlepšiu cenu. Niekto, kto kupuje tekvice na jesennú výzdobu, by možno chcel tieto informácie, aby mohol optimalizovať nákup balení tekvíc pre svoj patch.

## Hľadanie korelácie

[![ML pre začiatočníkov - Hľadanie korelácie: kľúč k lineárnej regresii](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pre začiatočníkov - Hľadanie korelácie: kľúč k lineárnej regresii")

> 🎥 Kliknite na obrázok vyššie pre krátke video o korelácii.

Z predchádzajúcej lekcie ste pravdepodobne videli, že priemerná cena podľa mesiacov vyzerá takto:

<img alt="Priemerná cena podľa mesiaca" src="../../../../translated_images/sk/barchart.a833ea9194346d76.webp" width="50%"/>

To naznačuje, že nejaká korelácia tam bude, a môžeme skúsiť natrénovať lineárny regresný model na predpovedanie vzťahu medzi `Month` a `Price`, alebo medzi `DayOfYear` a `Price`. Tu je rozptýlený graf, ktorý ukazuje druhý vzťah:

<img alt="Rozptýlený graf Cena vs. Deň v roku" src="../../../../translated_images/sk/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Skúsme zistiť koreláciu pomocou funkcie `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Zdá sa, že korelácia je pomerne malá, -0.15 podľa `Month` a -0.17 podľa `DayOfMonth`, ale môže tu byť iný dôležitý vzťah. Vyzerá to, že existujú rôzne skupiny cien zodpovedajúce rôznym odrodám tekvíc. Aby sme túto hypotézu potvrdili, nakreslime každú kategóriu tekvíc inou farbou. Pre odovzdanie parametra `ax` funkcii `scatter` môžeme vykresliť všetky body do rovnakého grafu:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Rozptýlený graf Cena vs. Deň v roku s farebným rozlíšením" src="../../../../translated_images/sk/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Naše vyšetrovanie naznačuje, že odroda má väčší vplyv na celkovú cenu než samotný dátum predaja. Vidíme to aj na stĺpcovom grafe:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stĺpcový graf ceny podľa odrody" src="../../../../translated_images/sk/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Zamerajme sa teraz na jednu odrodu tekvíc, 'pie type', a pozrime sa, aký vplyv má dátum na cenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Rozptýlený graf Cena vs. Deň v roku pre pie type" src="../../../../translated_images/sk/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Ak teraz vypočítame koreláciu medzi `Price` a `DayOfYear` pomocou funkcie `corr`, získame približne `-0.27` — čo znamená, že natrénovanie prediktívneho modelu má zmysel.

> Pred trénovaním lineárneho regresného modelu je dôležité zabezpečiť, že naše dáta sú čisté. Lineárna regresia nefunguje dobre s chýbajúcimi hodnotami, preto je rozumné zbaviť sa všetkých prázdnych buniek:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Iný prístup by bol vyplniť tieto prázdne hodnoty priemernými hodnotami príslušného stĺpca.

## Jednoduchá lineárna regresia

[![ML pre začiatočníkov - Lineárna a polynomiálna regresia pomocou Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pre začiatočníkov - Lineárna a polynomiálna regresia pomocou Scikit-learn")

> 🎥 Kliknite na obrázok vyššie pre krátke video o lineárnej a polynomiálnej regresii.

Na trénovanie nášho lineárneho regresného modelu použijeme knižnicu **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Začneme tým, že oddelíme vstupné hodnoty (vlastnosti) a očakávaný výstup (štítok) do samostatných numpy polí:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Všimnite si, že sme museli vykonať `reshape` na vstupných dátach, aby ich balíček Linear Regression správne pochopil. Lineárna regresia očakáva vstup v tvare 2D poľa, kde každý riadok poľa zodpovedá vektoru vstupných vlastností. V našom prípade, keďže máme iba jeden vstup, potrebujeme pole tvaru N&times;1, kde N je veľkosť datasetu.

Potom musíme rozdeliť údaje na tréningové a testovacie datasety, aby sme mohli po trénovaní modelu overiť jeho výkon:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Nakoniec samotné trénovanie lineárneho regresného modelu zaberie len dva riadky kódu. Definujeme objekt `LinearRegression` a prispôsobíme ho našim dátam pomocou metódy `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` po natrénovaní obsahuje všetky koeficienty regresie, ku ktorým sa dá pristúpiť pomocou vlastnosti `.coef_`. V našom prípade je len jeden koeficient, ktorý by mal byť okolo `-0.017`. To znamená, že ceny sa zdajú s časom mierne znižovať, ale nie príliš, približne o 2 centy za deň. Môžeme tiež pristúpiť k priesečníku regresie s osou Y pomocou `lin_reg.intercept_` – v našom prípade to bude okolo `21`, čo značí cenu na začiatku roka.

Aby sme videli, aká je presnosť nášho modelu, môžeme predikovať ceny na testovacej množine dát a potom zmerať, ako sú naše predpovede blízke očakávaným hodnotám. To sa dá urobiť pomocou metriky strednej štvorcovej chyby (MSE), čo je priemer všetkých štvorcových rozdielov medzi očakávanou a predikovanou hodnotou.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Naša chyba sa javí okolo 2 bodov, čo je približne 17%. Nie je to príliš dobré. Ďalším ukazovateľom kvality modelu je **koeficient determinácie**, ktorý môžeme získať takto:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Ak je hodnota 0, znamená to, že model neberie do úvahy vstupné dáta a správa sa ako *najhorší lineárny prediktor*, ktorý je jednoducho priemernou hodnotou výsledku. Hodnota 1 znamená, že dokážeme dokonale predpovedať všetky očakávané výstupy. V našom prípade je koeficient okolo 0.06, čo je dosť nízke.

Môžeme tiež vykresliť testovacie dáta spolu s regresnou čiarou, aby sme lepšie videli, ako regresia funguje v našom prípade:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/sk/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomická regresia

Ďalším typom lineárnej regresie je polynomická regresia. Kým niekedy existuje lineárny vzťah medzi premennými – čím väčšia je tekvica objemom, tým vyššia je cena – niekedy sa tieto vzťahy nedajú zobraziť ako rovina alebo priamka.

✅ Tu je [niekoľko ďalších príkladov](https://online.stat.psu.edu/stat501/lesson/9/9.8) dát, pre ktoré by bolo vhodné použiť polynomickú regresiu

Pozrite sa ešte raz na vzťah medzi dátumom a cenou. Zdá sa vám, že by mal byť nevyhnutne analyzovaný priamkou? Nemôžu ceny kolísať? V tomto prípade môžete skúsiť polynomickú regresiu.

✅ Polynomické výrazy sú matematické výrazy, ktoré môžu obsahovať jednu alebo viac premenných a koeficientov

Polynomická regresia vytvára zakrivenú čiaru, aby lepšie vyhovela nelineárnym dátam. V našom prípade, ak do vstupných dát zahrnieme druhú mocninu premennej `DayOfYear`, mali by sme byť schopní prispôsobiť dáta parabolickou krivkou, ktorá bude mať minimum v určitom bode v priebehu roka.

Scikit-learn obsahuje užitočné [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) na kombinovanie rôznych krokov spracovania dát dokopy. **Pipeline** je reťazec **estimatorov**. V našom prípade vytvoríme pipeline, ktorá najprv pridá polynomické prvky do nášho modelu a potom trénuje regresiu:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Použitie `PolynomialFeatures(2)` znamená, že zahrnieme všetky polynómy druhého stupňa z vstupných dát. V našom prípade to bude iba `DayOfYear`<sup>2</sup>, ale pri dvoch vstupných premenných X a Y sa pridajú X<sup>2</sup>, XY a Y<sup>2</sup>. Môžeme tiež použiť polynómy vyšších stupňov, ak chceme.

Pipeline možno používať rovnako ako pôvodný objekt `LinearRegression`, teda môžeme pipeline natrénovať pomocou `fit` a potom použiť `predict` na získanie výsledkov predikcie. Tu je graf zobrazujúci testovacie dáta a aproximačnú krivku:

<img alt="Polynomial regression" src="../../../../translated_images/sk/poly-results.ee587348f0f1f60b.webp" width="50%" />

Použitím polynomickej regresie môžeme dosiahnuť mierne nižšiu MSE a vyšší koeficient determinácie, ale nie výrazne. Musíme zohľadniť ďalšie vlastnosti!

> Vidíte, že minimálne ceny tekvíc sa prejavujú niekde okolo Halloweenu. Ako by ste to vysvetlili?

🎃 Gratulujeme, práve ste vytvorili model, ktorý môže pomôcť predpovedať cenu tekvíc na koláče. Pravdepodobne môžete rovnaký postup zopakovať pre všetky druhy tekvíc, ale to by bolo zdĺhavé. Naučíme sa teraz, ako zohľadniť odrodu tekvice v našom modeli!

## Kategorické vlastnosti

V ideálnom svete chceme byť schopní predpovedať ceny pre rôzne odrody tekvíc pomocou toho istého modelu. Avšak stĺpec `Variety` je trochu iný ako stĺpce ako `Month`, pretože obsahuje nečíselné hodnoty. Takéto stĺpce sa nazývajú **kategorické**.

[![ML pre začiatočníkov – predikcie kategórií pomocou lineárnej regresie](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pre začiatočníkov – predikcie kategórií pomocou lineárnej regresie")

> 🎥 Kliknite na obrázok vyššie pre krátky videopríklad použitia kategorických vlastností.

Tu vidíte, ako priemerná cena závisí na odrode:

<img alt="Average price by variety" src="../../../../translated_images/sk/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Aby sme zohľadnili odrodu, musíme ju najskôr premeniť na číselnú formu, teda **zakódovať** ju. Existuje niekoľko spôsobov, ako to urobiť:

* Jednoduché **číselné kódovanie** vytvorí tabuľku rôznych odrôd a potom nahradí názov odrody indexom z tejto tabuľky. To nie je najlepšia voľba pre lineárnu regresiu, pretože lineárna regresia vezme skutočnú číslenú hodnotu indexu a vynásobí ju koeficientom, čím ju pridá k výsledku. V našom prípade je vzťah medzi číslom indexu a cenou zjavne nelineárny, aj keď zabezpečíme, že indexy budú usporiadané určitým spôsobom.
* **One-hot encoding** nahradí stĺpec `Variety` štyrmi rôznymi stĺpcami, po jednom pre každú odrodu. Každý stĺpec bude obsahovať `1`, ak príslušný riadok je danej odrody, a `0` inak. To znamená, že v lineárnej regresii budú štyri koeficienty, jeden pre každú odrodu tekvíc, zodpovedajúce „počiatočnej cene“ (alebo skôr „dodatočnej cene“) pre túto konkrétnu odrodu.

Nasledujúci kód ukazuje, ako môžeme one-hot kódovať odrodu:

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

Na trénovanie lineárnej regresie so vstupom ako one-hot kódovaná odroda stačí správne inicializovať dáta `X` a `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Zvyšok kódu je rovnaký ako sme používali vyššie na trénovanie lineárnej regresie. Ak to vyskúšate, uvidíte, že stredná štvorcová chyba je približne rovnaká, ale získame oveľa vyšší koeficient determinácie (~77%). Pre ešte presnejšie predikcie môžeme zohľadniť ďalšie kategorické vlastnosti, ako aj číselné vlastnosti, napríklad `Month` alebo `DayOfYear`. Na získanie jedného veľkého poľa vlastností môžeme použiť `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tu tiež zohľadňujeme `City` a typ `Package`, čo nám dáva MSE 2.84 (10%) a determináciu 0.94!

## Spojme to všetko dokopy

Na vytvorenie najlepšieho modelu môžeme použiť kombinované (one-hot kódované kategorické + číselné) dáta z vyššie uvedeného príkladu spolu s polynomickou regresiou. Tu je kompletný kód pre vašu pohodlnosť:

```python
# nastaviť tréningové dáta
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# vykonať rozdelenie na trénovaciu a testovaciu množinu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# nastaviť a trénovať pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predpovedať výsledky pre testovacie dáta
pred = pipeline.predict(X_test)

# vypočítať MSE a koeficient určenia
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

To by nám malo dať najlepší koeficient determinácie takmer 97% a MSE=2.23 (~8% chyba predikcie).

| Model | MSE | Koeficient determinácie |
|-------|-----|-------------------------|
| Lineárna regresia s `DayOfYear` | 2.77 (17,2%) | 0.07 |
| Polynomická regresia s `DayOfYear` | 2.73 (17,0%) | 0.08 |
| Lineárna regresia s `Variety` | 5.24 (19,7%) | 0.77 |
| Lineárna regresia so všetkými vlastnosťami | 2.84 (10,5%) | 0.94 |
| Polynomická regresia so všetkými vlastnosťami | 2.23 (8,25%) | 0.97 |

🏆 Výborne! V tejto lekcii ste vytvorili štyri regresné modely a zlepšili kvalitu modelu na 97%. V záverečnej sekcii o regresii sa naučíte o logistickej regresii na určenie kategórií.

---
## 🚀Výzva

Otestujte niekoľko rôznych premenných v tomto zápisníku a zistite, ako korelácia súvisí s presnosťou modelu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

V tejto lekcii sme sa naučili o lineárnej regresii. Existujú aj iné dôležité typy regresie. Prečítajte si o technikách Stepwise, Ridge, Lasso a Elasticnet. Dobrou študijnou pomôckou je [kurz Stanford Statistical Learning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Zadanie

[Postavte model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Upozornenie**:  
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, uvedomte si, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Originálny dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->