# Regressziós modell építése Scikit-learn használatával: regresszió négyféleképpen

## Kezdő megjegyzés

Lineáris regressziót akkor használunk, amikor egy **numerikus értéket** akarunk megjósolni (például ház árát, hőmérsékletet vagy értékesítést).
Ez úgy működik, hogy megkeresi azt a legjobb egyenest, amely legjobban reprezentálja a bemeneti jellemzők és a kimenet közötti kapcsolatot.

Ebben a leckében a fogalom megértésére koncentrálunk, mielőtt további, fejlettebb regressziós technikákat vizsgálnánk.
![Lineáris vs polinomiális regresszió infografika](../../../../translated_images/hu/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika készítője: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ez a lecke elérhető R nyelven is!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Bevezetés

Eddig megvizsgáltad, mi az a regresszió, és mintákat néztél meg a sütőtök árképzési adatbázisból, amelyet az egész lecke során fogunk használni. Meg is jelenítetted azt a Matplotlib segítségével.

Most készen állsz arra, hogy mélyebben beleásd magad a regresszió témájába a gépi tanuláshoz. Míg a megjelenítés lehetővé teszi az adatok megértését, a gépi tanulás valódi ereje a _modellek betanításában_ rejlik. A modelleket történeti adatokon tanítjuk meg, hogy automatikusan megragadják az adatok közötti összefüggéseket, és lehetővé tegyék, hogy új adatokra előrejelzéseket készítsünk, melyekkel a modell még nem találkozott.

Ebben a leckében két regressziótípust fogsz megismerni: az _alapvető lineáris regressziót_ és a _polinomiális regressziót_, néhány matematikai háttérrel együtt. Ezek a modellek lehetővé teszik, hogy előre jelezzük a sütőtök árakat különböző bemeneti adatok alapján.

[![Gépi tanulás kezdőknek – A lineáris regresszió megértése](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Gépi tanulás kezdőknek – A lineáris regresszió megértése")

> 🎥 Kattints a fenti képre egy rövid videós áttekintésért a lineáris regresszióról.

> A tananyag során minimális matematikai ismerettel számolunk, és arra törekszünk, hogy a más területekről érkező diákok számára is érthető legyen, ezért figyelj a megjegyzésekre, 🧮 kiemelésekre, ábrákra és egyéb tanulási segédeszközökre, amelyek segítik a megértést.

### Előfeltétel

Már meg kell ismerned a sütőtök adatok szerkezetét, amelyeket megvizsgálunk. Ezek előre betöltve és megtisztítva szerepelnek ennél a leckénél az _notebook.ipynb_ fájlban. Ebben a fájlban a sütőtök ár bushelenként jelenik meg egy új adatkeretben. Győződj meg róla, hogy futtatni tudod ezeket a jegyzetfüzeteket a Visual Studio Code kerneljeiben.

### Előkészület

Emlékeztetőül: ezt az adatot azért töltöd be, hogy kérdéseket tehess fel vele kapcsolatban.

- Mikor a legjobb idő sütőtököt vásárolni?
- Milyen árat várhatok mini sütőtökök egy csomagjára?
- Érdemes-e fél bushel kosárban vagy inkább 1 1/9 bushel dobozban venni?

Nézzük tovább ezt az adatot.

Az előző leckében létrehoztál egy Pandas adatkeretet, amelybe betöltötted az eredeti adatállomány egy részét és egységesítetted az árakat bushelenként. Így azonban csak kb. 400 adatpontot nyertél, és csak a őszi hónapokra vonatkozóan.

Nézd meg az ebben a leckében előre betöltött adatot a jegyzetfüzet mellékleteként. Az adat már be van töltve, és egy kezdeti pontdiagram is készült a hónap adatainak megjelenítésére. Talán sikerül a tisztítással árnyaltabb képet kapni az adat természetéről.

## Egy lineáris regressziós egyenes

Ahogy az 1. leckében tanultad, a lineáris regressziós feladat célja egy olyan egyenes ábrázolása, amely:

- **Variable kapcsolatok bemutatása**. Megmutatja a változók közötti kapcsolatot
- **Előrejelzés készítése**. Pontosan megjósolja, hogy az új adatpont hol fog elhelyezkedni az egyeneshez képest.

Tipikusan a **legkisebb négyzetek regresszió** alkalmazása révén rajzolunk ilyen egyenest. A "legkisebb négyzetek" kifejezés arra a folyamatra utal, amely során minimalizáljuk a modell összes hibáját. Minden adatpontra megmérjük a függőleges távolságot (reziduális), mely az adott pont és a regressziós egyenes közötti távolság.

Ezeket a távolságokat négyzetre emeljük két fő okból:

1. **Iránynál fontosabb a nagyság:** Az a hibát, ha -5 vagy +5, egyformán kezeljük. A négyzetre emeléssel az összes érték pozitívvá válik.

2. **Eltérő értékek büntetése:** A négyzetre emelés nagyobb súlyt ad a nagyobb hibáknak, így az egyenes közelebb kerül az eltérő pontokhoz.

Mindezek után összeadjuk az összes négyzetre emelt értéket. A cél az, hogy megtaláljuk azt az egyenest, amelynél ez az összeg a legkisebb (legkisebb lehetséges érték) – innen ered a "Legkisebb négyzetek" név.

> **🧮 Mutasd a képletet**
> 
> Ezt az egyenest, az úgynevezett _legjobb illeszkedésű egyenest_, a [következő képlettel](https://en.wikipedia.org/wiki/Simple_linear_regression) lehet leírni:
> 
> ```
> Y = a + bX
> ```
>
> `X` a ’magyarázó változó’. `Y` a ’függő változó’. Az egyenes meredeksége `b`, az `a` pedig az y-tengely metszete, vagyis az `Y` értéke, amikor `X = 0`.
>
>![a meredekség kiszámítása](../../../../translated_images/hu/slope.f3c9d5910ddbfcf9.webp)
>
> Először számítsuk ki a `b` meredekséget. Infografika készítője: [Jen Looper](https://twitter.com/jenlooper)
>
> Más szóval, és utalva a sütőtök adataink eredeti kérdésére: "jósoljuk meg a sütőtök bushelenkénti árát hónapra lebontva", az `X` az árra, az `Y` pedig az értékesítés hónapjára utalna.
>
>![egyenlet kitöltése](../../../../translated_images/hu/calculation.a209813050a1ddb1.webp)
>
> Számítsuk ki az `Y` értékét. Ha körülbelül 4 dollárt fizetsz, bizonyára április van! Infografika készítője: [Jen Looper](https://twitter.com/jenlooper)
>
> A képletnek ki kell számolnia az egyenes meredekségét, amely az y-metszettől, azaz attól függ, hol helyezkedik el az `Y`, amikor `X = 0`.
>
> Megfigyelheted a számítási módot a [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) weboldalon. Látogass el erre [a legkisebb négyzetek számológépre](https://www.mathsisfun.com/data/least-squares-calculator.html), hogy lásd, miként befolyásolják az értékek az egyenest.

## Korreláció

Van még egy fogalom, amit meg kell érteni, ez pedig a **korrelációs együttható** az adott X és Y változók között. Egy pontdiagram segítségével könnyen meg lehet jeleníteni ezt az együtthatót. Ha az adatpontok szépen egy egyenes mentén helyezkednek el, akkor magas a korreláció, ha pedig mindenfelé szóródnak az X és Y között, akkor alacsony.

Egy jó lineáris regressziós modell lesz olyan, amely szoros (inkább 1-hez, semmint 0-hoz közeli) korrelációs együtthatóval rendelkezik a legkisebb négyzetek regressziós módszerével egy regressziós vonal mellett.

✅ Futtasd az ehhez a leckéhez mellékelt jegyzetfüzetet, és nézd meg a Hónap-Arány pontdiagramot. A hónap és ára közötti kapcsolat a sütőtök értékesítésében szerinted magas vagy alacsony korrelációjú a pontdiagram vizuális értelmezése alapján? Változik-e, ha finomabb mértéket használunk a `Month` helyett, például az *év napja* (az év első napja óta eltelt napok száma) alapján?

A lentebbi kódban azt feltételezzük, hogy megtisztítottuk az adatot, és létrejött egy `new_pumpkins` nevű adatkeret az alábbi módon:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Az adatok tisztítására szolgáló kód elérhető a [`notebook.ipynb`](notebook.ipynb) fájlban. Ugyanazokat a tisztítási lépéseket hajtottuk végre, mint az előző leckében, és kiszámoltuk a `DayOfYear` oszlopot a következő kifejezés használatával:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Most, hogy érted a lineáris regresszió mögötti matematikát, hozzunk létre egy regressziós modellt, hogy megnézzük, meg tudjuk-e jósolni, melyik sütőtök csomag árazása lesz a legjobb. Valaki, aki ünnepi sütőtök díszhez vásárol, ezt az információt használhatja vásárlási döntések optimalizálására.

## Korreláció keresése

[![Gépi tanulás kezdőknek - Korreláció keresése: Az összefüggés a lineáris regresszió kulcsa](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Gépi tanulás kezdőknek - Korreláció keresése: Az összefüggés a lineáris regresszió kulcsa")

> 🎥 Kattints a fenti képre egy rövid videós áttekintésért a korrelációról.

Az előző leckéből valószínűleg láttad, hogy különböző hónapok átlagára így néz ki:

<img alt="Átlagár hónapokra bontva" src="../../../../translated_images/hu/barchart.a833ea9194346d76.webp" width="50%"/>

Ez arra utal, hogy van összefüggés, és megpróbálhatjuk a lineáris regresszió modellt betanítani a `Month` és az ár, vagy a `DayOfYear` és az ár kapcsolatára. Az alábbi szórásterület ábrán ennek az utóbbinak a viszonya látható:

<img alt="Szórásdiagram az ár és az év napja szerint" src="../../../../translated_images/hu/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Nézzük meg, milyen korreláció van az `corr` függvénnyel:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Úgy tűnik, a korreláció meglehetősen kicsi, -0,15 a hónap szerint és -0,17 az év napja szerint, de lehet, hogy egy másik fontos összefüggés is létezik. Úgy tűnik, különböző árkategóriák léteznek a sütőtök fajták szerint. Ennek megerősítésére próbáljuk meg az egyes sütőtök kategóriákat különböző színnel megjeleníteni. Az `ax` paraméter átadásával a `scatter` függvénynek az összes pont ugyanazon a grafikonon ábrázolható:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Szórásdiagram az ár és az év napja színkódolt" src="../../../../translated_images/hu/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Vizsgálatunk arra utal, hogy a fajta hatása nagyobb az ár összértékére, mint az eladási időponté. Ezt egy oszlopdiagramon is megláthatjuk:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Oszlopdiagram az ár és a fajta szerint" src="../../../../translated_images/hu/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Most egyelőre csak az egyik sütőtök fajtára, a ’pie type’-ra koncentráljunk, és nézzük meg, milyen hatása van az értékesítési dátumnak az árra:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Szórásdiagram az ár és az év napja a pie típusú sütőtöknél" src="../../../../translated_images/hu/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Ha most kiszámoljuk a korrelációt a `Price` és a `DayOfYear` között az `corr` függvénnyel, az kb. `-0.27` lesz – ami azt jelenti, hogy érdemes prediktív modellt tanítani.

> A lineáris regressziós modell betanítása előtt fontos, hogy az adatunk tiszta legyen. A lineáris regresszió nem működik jól hiányzó értékekkel, ezért érdemes megszabadulni minden hiányzó adattól:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Másik megközelítés lehet a hiányzó értékek kitöltése az adott oszlop átlagával.

## Egyszerű lineáris regresszió

[![Gépi tanulás kezdőknek – Lineáris és polinomiális regresszió Scikit-learn segítségével](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Gépi tanulás kezdőknek – Lineáris és polinomiális regresszió Scikit-learn segítségével")

> 🎥 Kattints a fenti képre egy rövid videós áttekintésért a lineáris és polinomiális regresszióról.

Lineáris regressziós modellünk betanításához a **Scikit-learn** könyvtárat fogjuk használni.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Kezdjük azzal, hogy a bemeneti értékeket (jellemzők) és a várt kimenetet (címkét) külön numpy tömbökbe rendezzük:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Figyeld meg, hogy a bemeneti adatokat át kellett formáznunk (`reshape`), hogy a Lineáris Regresszió csomag helyesen értelmezze őket. A Lineáris Regresszió 2D tömböt vár bemenetként, ahol a tömb minden sora egy vektor a bemeneti jellemzőkről. Mivel nekünk csak egy bemenetünk van, egy N×1 alakú tömböt kell adnunk, ahol N az adatkészlet mérete.

Ezután az adatot szét kell osztanunk tanító és teszt halmazokra, hogy a modell betanítása után tesztelni tudjuk azt:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Végül a lineáris regressziós modell tényleges betanítása csak két kód sorból áll. Létrehozzuk a `LinearRegression` objektumot, és betanítjuk az adatainkra a `fit` metódussal:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

A `LinearRegression` objektum a `fit`-elés után tartalmazza a regresszió összes együtthatóját, amelyekhez a `.coef_` tulajdonságon keresztül férhetünk hozzá. Esetünkben csak egy együttható van, amely körülbelül `-0.017` körül várható. Ez azt jelenti, hogy az árak idővel kissé csökkennek, de nem sokat, körülbelül 2 centet naponta. A regresszió Y tengellyel való metszéspontját is elérhetjük a `lin_reg.intercept_` segítségével – ami nálunk körülbelül `21` lesz, jelezve az év eleji árat.

Hogy lássuk, mennyire pontos a modellünk, először árakat jósolhatunk egy tesztadatállományra, majd mérhetjük, mennyire közel vannak a jóslataink a várt értékekhez. Ezt elvégezhetjük a négyzetes hibák átlagával, azaz a mean square error (MSE) metrikával, amely a várt és a jósolt értékek közötti négyzetes különbségek átlaga.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

A hibánk úgy tűnik, körülbelül 2 pont körül van, ami kb. 17%. Nem túl jó. A modell minőségének másik mutatója az **elszámolási együttható (coefficient of determination)**, amely így határozható meg:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Ha az érték 0, az azt jelenti, hogy a modell nem veszi figyelembe a bemeneti adatokat, és a *legrosszabb lineáris prediktorként* működik, ami egyszerűen az eredmény átlagértéke. Ha az érték 1, az azt jelenti, hogy tökéletesen tudjuk előre jelezni az összes várt kimenetet. Nálunk az együttható körülbelül 0.06, ami elég alacsony.

A tesztadatokat a regressziós vonallal együtt is ábrázolhatjuk, hogy jobban lássuk, hogyan működik a regresszió a mi esetünkben:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/hu/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polinomiális regresszió

A lineáris regressziónak egy másik típusa a polinomiális regresszió. Bár néha a változók közt lineáris kapcsolat van – például minél nagyobb a tök térfogata, annál magasabb az ára –, előfordul, hogy ezek a kapcsolatok nem ábrázolhatók egy síkkal vagy egyenes vonallal.

✅ Itt vannak [további példák](https://online.stat.psu.edu/stat501/lesson/9/9.8) olyan adatokra, amelyek esetében polinomiális regresszió alkalmazható.

Nézzük újra az összefüggést a Dátum és az Ár között. Ez a pontfelhő úgy tűnik, hogy feltétlenül egyenes vonallal kellene elemezni? Nem ingadozhatnak az árak? Ilyen esetben polinomiális regressziót próbálhatunk ki.

✅ A polinomok matematikai kifejezések, amelyek egy vagy több változóból és együtthatóból állhatnak.

A polinomiális regresszió egy görbe vonalat hoz létre, hogy jobban illeszkedjen a nemlineáris adatokra. Nálunk, ha a bemeneti adathoz hozzáadjuk a `DayOfYear` négyzetét, akkor képesek leszünk egy parabolikus görbével illeszteni az adatokat, amelynek a minimuma az év egy bizonyos pontján lesz.

A Scikit-learn tartalmaz egy hasznos [pipeline API-t](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), amellyel több adatfeldolgozási lépést is összekapcsolhatunk. Egy **pipeline** egy **becslők** láncolata. Nálunk egy olyan pipeline-t hozunk létre, amely először polinomiális jellemzőket ad a modellhez, majd megtanítja a regressziót:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

A `PolynomialFeatures(2)` azt jelenti, hogy az összes másodfokú polinomot belefoglaljuk a bemeneti adatból. Nálunk ez csak a `DayOfYear`<sup>2</sup>-t jelenti, de ha két bemeneti változónk, X és Y van, akkor hozzáadódik az X<sup>2</sup>, XY és Y<sup>2</sup> is. Természetesen magasabb fokú polinomokat is használhatunk.

A pipeline-okat ugyanúgy használhatjuk, mint az eredeti `LinearRegression` objektumot, azaz illeszthetjük (`fit`), majd jósolhatunk (`predict`). Itt látható egy grafikon a tesztadatokról és az illesztett görbéről:

<img alt="Polynomial regression" src="../../../../translated_images/hu/poly-results.ee587348f0f1f60b.webp" width="50%" />

A polinomiális regresszióval kissé alacsonyabb MSE-t és magasabb determinációt érhetünk el, de nem jelentősen. Más jellemzőket is figyelembe kell vennünk!

> Látható, hogy a legkisebb tökárak nagyjából Halloween környékén vannak. Hogyan magyaráznád ezt?

🎃 Gratulálok, most egy olyan modellt hoztál létre, amely segít előre jelezni a sütőtök árát. Valószínűleg ugyanígy eljárhatsz az összes tökfajtával, de ez fárasztó lenne. Tanuljuk meg inkább, hogyan vegyük figyelembe a tökfajtát a modellünkben!

## Kategorizált jellemzők

Az ideális világban ugyanazzal a modellel szeretnénk különböző tökfajták árát előrejelezni. Azonban a `Variety` oszlop eltér a `Month`-hoz hasonló oszlopoktól, mert nem numerikus értékeket tartalmaz. Az ilyen oszlopokat **kategóriális** oszlopoknak nevezzük.

[![ML kezdőknek - Kategóriás jellemzők előrejelzése lineáris regresszióval](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML kezdőknek - Kategóriás jellemzők előrejelzése lineáris regresszióval")

> 🎥 Kattints a fenti képre a kategóriás jellemzők használatát bemutató rövid videóért.

Itt látható, hogyan függ az átlagár a fajtától:

<img alt="Average price by variety" src="../../../../translated_images/hu/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Ahhoz, hogy figyelembe vegyük a fajtát, először numerikus formába kell konvertálnunk, vagyis **kódolnunk** kell. Többféleképpen tehetjük ezt meg:

* Az egyszerű **numerikus kódolás** egy táblázatot épít a különböző fajtákról, majd a fajta nevét a táblázatbeli indexére cseréli. Ez nem a legjobb módszer a lineáris regresszióhoz, mert a regresszió a kód numerikus értékét veszi figyelembe és szorozza együtthatóval. Nálunk az indexszám és az ár közötti kapcsolat egyértelműen nem lineáris, még akkor sem, ha az indexeket valamilyen sorrendbe rendezzük.
* A **one-hot kódolás** a `Variety` oszlop helyett 4 külön oszlopot készít, egyet-egyet minden fajtára. Minden oszlop 1-et tartalmaz, ha az adott sor az adott fajta, különben 0-t. Ez azt jelenti, hogy a lineáris regresszióban négy együttható lesz, egy-egy minden tökfajtára, amelyek az adott fajta "kiinduló ára" (vagy inkább "további ára") felelős.

Az alábbi kód megmutatja, hogyan kódolhatjuk one-hot módszerrel a fajtát:

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

A lineáris regresszió tanításához one-hot kódolt fajtával csak helyesen kell inicializálni az `X` és `y` adatokat:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

A további kód ugyanaz, mint amit fent használtunk a lineáris regresszió tanításához. Ha kipróbálod, azt látod, hogy a négyzetes hiba nagyjából ugyanaz marad, viszont jóval magasabb lesz az elszámolási együttható (~77%). Még pontosabb jóslatokhoz több kategóriás jellemzőt is bevonhatunk, illetve numerikus jellemzőket, például `Month` vagy `DayOfYear`. Az adatok egyetlen nagy tömbbé egyesítéséhez a `join` használható:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Itt a `City`-t és a `Package` típust is figyelembe vesszük, ami MSE=2.84 (10%) és determináció=0.94 eredményt ad!

## Összeillesztés

A legjobb modell elkészítéséhez egyesíthetjük a fent említett (one-hot kódolt kategóriás + numerikus) adatokat a polinomiális regresszióval. Alább a teljes kód a könnyebb használathoz:

```python
# tréning adatok előkészítése
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# tanuló-teszt adatfelosztás végrehajtása
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# pipeline beállítása és betanítása
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# tesztadatokra történő eredményjóslás
pred = pipeline.predict(X_test)

# MSE és determináció kiszámítása
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Ez megközelítőleg 97% determinációs együtthatót és MSE=2.23 (~8% előrejelzési hibát) eredményez.

| Modell | MSE | Determináció |
|-------|-----|---------------|
| `DayOfYear` lineáris | 2.77 (17.2%) | 0.07 |
| `DayOfYear` polinomiális | 2.73 (17.0%) | 0.08 |
| `Variety` lineáris | 5.24 (19.7%) | 0.77 |
| Minden jellemző lineáris | 2.84 (10.5%) | 0.94 |
| Minden jellemző polinomiális | 2.23 (8.25%) | 0.97 |

🏆 Szép munka! Ebben a leckében négy regressziós modellt hoztál létre, és a modellminőséget 97%-ra javítottad. Az utolsó fejezetben a logisztikus regresszióról tanulsz majd a kategóriák meghatározásához.

---
## 🚀Kihívás

Tesztelj különböző változókat ebben a jegyzetfüzetben, és figyeld meg, hogyan tükröződik a korreláció a modell pontosságában.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Összefoglalás és önálló tanulás

Ebben a leckében a lineáris regresszióval ismerkedtünk meg. Más fontos regressziótípusok is vannak. Olvass a Stepwise, Ridge, Lasso és Elasticnet technikákról. Egy ajánlott tanfolyam további ismeretekért a [Stanford Statisztikai Tanulás kurzusa](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Feladat

[Modell készítése](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Nyilatkozat**:
Jelen dokumentumot az AI fordító szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével fordítottuk. Bár igyekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hivatalos forrásnak. Kritikus információk esetén professzionális, emberi fordítást javaslunk. Nem vállalunk felelősséget a fordítás használatából eredő félreértésekért vagy félreértelmezésekért.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->