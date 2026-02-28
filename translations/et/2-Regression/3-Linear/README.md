# Ehita regressioonimudel, kasutades Scikit-learn'i: regressioon neli erinevat viisi

## Algaja märkus

Lineaarset regressiooni kasutatakse, kui tahame prognoosida **numbrilist väärtust** (näiteks maja hind, temperatuur või müük).
See töötab leidmisega sirgjoont, mis kõige paremini esindab seost sisendfunktsioonide ja väljundi vahel.

Selles õppetükis keskendume mõiste mõistmisele enne keerukamate regressioonitehnikate uurimist.
![Lineaarne ja polünoomne regressioon infograafik](../../../../translated_images/et/linear-polynomial.5523c7cb6576ccab.webp)
> Infograafik autorilt [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Eelnev loengu visuaaltest](https://ff-quizzes.netlify.app/en/ml/)

> ### [See õppetükk on saadaval ka R keeles!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Sissejuhatus

Senini oled uurinud, mis on regressioon, kasutades näidisandmeid kõrvitsahinnast, mida kasutame kogu õppetüki vältel. Oled seda ka visualiseerinud Matplotlib'i abil.

Nüüd oled valmis süvenema regressioonisse masinõppes. Kuigi visualiseerimine aitab andmeid mõista, tuleb masinõppe tõeline jõud _mudelite koolitamisest_. Mudelid koolitatakse ajalooliste andmete põhjal, et automaatselt jäädvustada andmete sõltuvused, võimaldades ennustada tulemusi uutele andmetele, mida mudel varem näinud ei ole.

Selles õppetükis õpid tundma kahte regressioonitüüpi: _lihtsat lineaarset regressiooni_ ja _polünoomset regressiooni_, koos mõningate matemaatiliste aluste selgitustega. Need mudelid võimaldavad meil ennustada kõrvitsahindu erinevate sisendandmete põhjal.

[![Masinõpe algajatele - Lineaarse regressiooni mõistmine](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Masinõpe algajatele - Lineaarse regressiooni mõistmine")

> 🎥 Klõpsa ülalolevale pildile, et vaadata lühikest ülevaadet lineaarse regressiooni kohta.

> Selle õppekava jooksul eeldame vähest matemaatikateadmiste hulka ja püüame muuta selle teiste valdkondade üliõpilastele arusaadavaks, nii et jälgi märkusi, 🧮 selgitusi, diagramme ja teisi õppematerjale, mis hõlbustavad mõistmist.

### Eelteadmised

Sul peaks nüüd olema tuttav kõrvitsaandmete struktuur, mida me uurime. Need on eelnevalt laetud ja puhastatud antud õppetüki _notebook.ipynb_ failis. Failis kuvatakse kõrvitsahind busheli kohta uues andmeraamis. Veendu, et suudad need märkmikud Visual Studio Code'is tuumades jooksutada.

### Ettevalmistus

Meenutuseks: sa laadid neid andmeid, et esitada neile küsimusi.

- Millal on parim aeg kõrvitsaid osta?
- Millist hinda võib oodata minikõrvitsate paki eest?
- Kas peaksin ostma neid poolbusheli korvides või 1 1/9 busheli karbis?
Jätkame andmete uurimist.

Eelmisel õppetunnil lõid Pandase andmeraami ja täitsid selle osa algsest andmestikust, standardiseerides hinnad busheli alusel. Selle tegemisega kogusid siiski ainult umbes 400 andmepunkti ja ainult sügisekuude kohta.

Vaata andmeid, mida me eelnevalt laadsime selle õppetüki kaasasolevasse märkmikku. Andmed on eelnevalt laetud ja tehtud algne hajuvusdiagramm, mis näitab kuupõhiseid andmeid. Võib-olla saame andmete olemuse kohta veidi detailsemalt teada, tehes veel puhastust.

## Lineaarse regressioonijoon

Nagu õppetükis 1 õppisid, on lineaarse regressiooni eesmärk joonistada joon, mis:

- **Näitab muutujate seoseid**. Näitab seost muutujate vahel
- **Teeb ennustusi**. Teeb täpseid prognoose, kuhu uus andmepunkt joone suhtes langeb.
 
Tavaliselt joonistatakse sellist joont meetodi **Vähimate ruutude regressioon** abil. Mõiste "Vähimate ruutude" viitab meie mudeli koguviga minimeerimise protsessile. Iga andmepunkti puhul mõõdame vertikaalkauguse (nimetatakse jääkveaks) tegeliku punkti ja regressioonijoone vahel.

Me ruudutame need kaugused kahe peamise põhjuse tõttu:

1. **Suurema tähtsus kui suuna puhul:** Tahame, et -5 vea suurus oleks sama kui +5 viga. Ruudutamine teeb kõik väärtused positiivseks.

2. **Ebatavaliste väärtuste karistamine:** Ruudutamine annab suurematele vigadele suurema kaalu, sundides joont olema lähemal kaugel olevatele punktidele.

Seejärel liidame need ruudutatud väärtused kokku. Meie eesmärk on leida see konkreetne joon, millel see summa on minimaalne (võimalikult väike väärtus) — seega nimi "Vähimate ruutude".

> **🧮 Näita mulle matemaatikat**  
>  
> Seda joont, mida nimetatakse _parima sobivusega joon_, saab väljendada [valemi abil](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>
> `X` on 'selgitav muutuja'. `Y` on 'sõltuv muutuja'. Joonte kalle on `b` ja `a` on y-lõikepunkt, mis tähistab `Y` väärtust, kui `X = 0`.
>
>![kallet arvutada](../../../../translated_images/et/slope.f3c9d5910ddbfcf9.webp)
>
> Esiteks arvuta kalle `b`. Infograafik autorilt [Jen Looper](https://twitter.com/jenlooper)
>
> Teisisõnu, viidates meie kõrvitsaandmete algsele küsimusele: "prognoosi kõrvitsa hind busheli kohta kuu järgi", viitab `X` hinnale ja `Y` müügikuule.
>
>![valemi lõpetamine](../../../../translated_images/et/calculation.a209813050a1ddb1.webp)
>
> Arvuta `Y` väärtus. Kui maksad umbes 4 dollarit, peab see olema aprill! Infograafik autorilt [Jen Looper](https://twitter.com/jenlooper)
>
> Matemaatika, mis arvutab joone, peab demonstreerima joone kalde, mis sõltub ka lõikepunktist, ehk kus `Y` asub, kui `X = 0`.
>
> Saad vaadata nende väärtuste arvutusmeetodit veebisaidil [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Samuti külasta [kasutades vähimate ruutude kalkulaatorit](https://www.mathsisfun.com/data/least-squares-calculator.html), et näha, kuidas numbrite väärtused mõjutavad joont.

## Korrelatsioon

Veel üks mõiste, mida mõista, on **Korrelatsioonikordaja** antud X ja Y muutujate vahel. Hajuvusdiagrammiga saab seda kordajat kiiresti visualiseerida. Kui andmepunktid paiknevad korrapärases reas, on kõrge korrelatsioon; kui punktid on hajutatud kõikjale X ja Y vahel, on korrelatsioon madal.

Hea lineaarne regressioonimudel omab kõrget (lähemal 1-le kui 0-le) korrelatsioonikordajat, kasutades Vähimate ruutude regressiooni meetodit koos regressioonijoonega.

✅ Käivita selle õppetüki märkmik ja vaata kuupõhist hinna hajuvusdiagrammi. Kas andmed, mis seovad kuupäeva ja kõrvitsate hinna, näivad omavat suurt või väikest korrelatsiooni vastavalt sinu visuaalsele tõlgendusele hajuvusdiagrammil? Kas see muutub, kui kasutad peenemat mõõdikut kui `Kuu`, nt *aasta päev* (st päevade arv aasta algusest)?

Järgmises koodis eeldame, et andmed on puhastatud ja meil on andmeraamistik nimega `new_pumpkins`, mis on sarnane järgmisele:

ID | Kuu | AastaPäev | Tüüp | Linn | Pakk | Madal hind | Kõrge hind | Hind
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | KÕRVITSA TIPP | BALTIMORE | 1 1/9 busheli kastid | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | KÕRVITSA TIPP | BALTIMORE | 1 1/9 busheli kastid | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | KÕRVITSA TIPP | BALTIMORE | 1 1/9 busheli kastid | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | KÕRVITSA TIPP | BALTIMORE | 1 1/9 busheli kastid | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | KÕRVITSA TIPP | BALTIMORE | 1 1/9 busheli kastid | 15.0 | 15.0 | 13.636364

> Andmete puhastamise kood on saadaval failis [`notebook.ipynb`](notebook.ipynb). Oleme teinud samad puhastamise sammud nagu eelnevas õppetükis ning arvutanud `AastaPäev` veeru järgmiselt: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nüüd, kui sul on arusaam lineaarse regressiooni matemaatikast, loome regressioonimudeli, et näha, kas võime ennustada, milline kõrvitsapakend annab parimad hinnad. Keegi, kes ostab kõrvitsaid püha-kõrvitsapeenrale, võib vajada seda infot, et osta parima hinnaga kõrvitsapakette.

## Korrelatsiooni otsimine

[![Masinõpe algajatele - korrelatsiooni otsimine: lineaarse regressiooni võti](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Masinõpe algajatele - korrelatsiooni otsimine: lineaarse regressiooni võti")

> 🎥 Klõpsa ülalolevale pildile, et vaadata lühikest ülevaadet korrelatsioonist.

Eelmisest õppetükist oled ilmselt näinud, et kuu keskmine hind näeb välja selline:

<img alt="Keskmine hind kuu kaupa" src="../../../../translated_images/et/barchart.a833ea9194346d76.webp" width="50%"/>

See viitab võimalusele, et korrelatsioon võiks eksisteerida, ja võime proovida treenida lineaarset regressioonimudelit, mis ennustab seost `Kuu` ja `Hind` vahel või `AastaPäev` ja `Hind` vahel. Siin on hajuvusdiagramm, mis näitab viimast seost:

<img alt="Hajuvusdiagramm hinna ja aasta päeva vahel" src="../../../../translated_images/et/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vaatame, kas korrelatsiooni on, kasutades funktsiooni `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Tundub, et korrelatsioon on suhteliselt väike, -0.15 `Kuu` järgi ja -0.17 `AastaPäeva` järgi, kuid võib olla mõni teine tähtis seos. Näib, et hinnad jagunevad erinevatesse gruppidesse vastavalt kõrvitsatüübile. Selle kinnitamiseks joonistame iga kõrvitsaliigi erinevas värvitoonis. Andmepunktide samaaegseks joonistamiseks peame `scatter` joonistamismetoodil kasutama parameetrit `ax`:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Hajuvusdiagramm hinna ja aasta päeva vahel värviliselt" src="../../../../translated_images/et/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Meie uurimine viitab, et liigiga on suurem mõju üldisele hinnale kui müügikuupäeval. Seda näeme ka tulpdiagrammilt:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Tulpdiagramm hindade kohta liigiti" src="../../../../translated_images/et/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Keskendume hetkel ainult ühele kõrvitsaliigile, 'pirukaliigile', ja vaatame, kuidas hind sõltub kuupäevast:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Hajuvusdiagramm hinna ja aasta päeva vahel 'pirukaliigile'" src="../../../../translated_images/et/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Kui nüüd arvutada korrelatsioon `Price` ja `DayOfYear` vahel funktsiooniga `corr`, saame ligikaudu `-0.27` - mis tähendab, et prognoosimudeli koolitamine on mõistlik.

> Enne lineaarse regressioonimudeli koolitamist on oluline veenduda, et andmed on puhtad. Lineaarne regressioon ei tööta hästi puuduvate väärtustega, seega on mõistlik tühjad lahtrid eemaldada:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Teine võimalus oleks need tühjad väärtused asendada vastava veeru keskmisega.

## Lihtne lineaarne regressioon

[![Masinõpe algajatele - lineaarne ja polünoomne regressioon Scikit-learniga](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Masinõpe algajatele - lineaarne ja polünoomne regressioon Scikit-learniga")

> 🎥 Klõpsa ülalolevale pildile, et vaadata lühikest ülevaadet lineaarse ja polünoomse regressiooni kohta.

Me koolitame oma lineaarse regressioonimudeli, kasutades **Scikit-learn'i** teeki.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Alustame sisendväärtuste (tunnuste) ja oodatud väljundi (sildi) eraldamisest eraldi numpy massiivideks:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Märka, et pidime sisendandmeid töötlema `reshape` abil, et LinearRepression pakett saaks neid õigesti mõista. Lineaarne regressioon eeldab 2D massiivi sisendina, kus iga rea vastab sisendfunktsioonide vektorile. Meie puhul, kui meil on ainult üks sisend, vajame massiivi kuju N &times; 1, kus N on andmestiku suurus.

Seejärel peame andmed jagama koolitus- ja testandmeteks, et saaksime mudelit pärast koolitust valideerida:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Lõpuks võtab lineaarse regressioonimudeli koolitus vaid kaks koodirida. Defineerime `LinearRegression` objekti ja sobitame selle meie andmetega meetodiga `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` objekt sisaldab pärast `fit`-imist regressiooni kõiki koefitsiente, millele pääseb ligi `.coef_` omaduse kaudu. Meie puhul on vaid üks koefitsient, mis peaks olema umbes `-0.017`. See tähendab, et hinnad paistavad aja jooksul veidi langemas, aga mitte liiga palju, umbes 2 senti päevas. Samuti saame regressiooni lõikepunkti Y-teljel kätte `lin_reg.intercept_` abil - see on meie puhul umbes `21`, mis näitab hinna väärtust aasta alguses.

Selleks, et näha, kui täpne meie mudel on, saame prognoosida hindu testandmestikul ja seejärel mõõta, kui lähedal on meie prognoosid oodatud väärtustele. Seda saab teha ruutkeskmise vea (MSE) mõõdikuga, mis on kõigi ruutude keskmine erinevus oodatud ja prognoositud väärtuste vahel.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Meie viga paistab olevat umbes 2 punkti, mis on umbes 17%. Mitte kuigi hea. Teine mudeli kvaliteedi näitaja on **determinisatsioonikordaja**, mida saab saada järgmise koodiga:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Kui väärtus on 0, tähendab see, et mudel ei võta sisendandmeid arvesse ja toimib nagu *kõige halvem lineaarne prognoosija*, mis on lihtsalt tulemuse keskmine väärtus. Väärtus 1 tähendab, et suudame kõiki oodatud väljundeid täiuslikult prognoosida. Meie puhul on kordaja umbes 0.06, mis on üsna madal.

Testandmeid võime koos regressioonijoonisega joonistada, et paremini näha, kuidas regressioon meie puhul töötab:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineaarne regressioon" src="../../../../translated_images/et/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polünoomne regressioon

Teine lineaarse regressiooni tüüp on polünoomne regressioon. Kuigi mõnikord on muutujate vahel lineaarne seos - näiteks suurem kõrvits mahult tähendab kõrgemat hinda -, siis mõnikord neid seoseid ei saa joonistada tasandina ega sirgjoonena.

✅ Siin on [veel mõned näited](https://online.stat.psu.edu/stat501/lesson/9/9.8) andmetest, mille puhul võiks kasutada polünoomset regressiooni

Võta veelkord pilk peale seosele kuupäeva ja hinna vahel. Kas see hajuvusdiagramm tundub kindlasti nii, et seda peaks tingimata analüüsima sirgjoonega? Kas hinnad ei kõigu? Sellisel juhul võid proovida polünoomset regressiooni.

✅ Polünoomid on matemaatilised avaldised, mis võivad koosneda ühest või mitmest muutujast ja koefitsiendist

Polünoomne regressioon loob kõverjoone, mis paremini sobib mittelineaarsete andmetega. Meie puhul, kui lisame sisendandmetesse ruutfunktsiooni `DayOfYear` muutujast, peaksime suutma sobitada andmeid paraboolkuvaga, millel on aasta jooksul mingi miinimumtipp.

Scikit-learn sisaldab kasulikku [pipeline API-d](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), et kombineerida erinevad andmetöötluse sammud. **Pipelines** on **estimatsioonide** kett. Meie puhul loome väärtusahelas esimese sammuna polünoomsed tunnused ja seejärel treenime regressiooni:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)` kasutamine tähendab, et kaasame kõik teise astme polünoomid sisendandmetest. Meie puhul tähendab see ainult `DayOfYear`<sup>2</sup>, aga kui on kaks sisendmuutujat X ja Y, lisab see X<sup>2</sup>, XY ja Y<sup>2</sup>. Saame kasutada ka kõrgema astme polünoome, kui soovime.

Pipeline'idega saab tööd teha samamoodi nagu algse `LinearRegression` objektiga, s.t. saame `fit` väärtusahela ja seejärel `predict` meetodit kasutades saada prognoosid. Järgneval graafikul on testandmed ja ligikaudne kõver:

<img alt="Polünoomne regressioon" src="../../../../translated_images/et/poly-results.ee587348f0f1f60b.webp" width="50%" />

Polünoomse regressiooni kasutamisel saame kergelt madalama MSE ja kõrgema determinatsiooni, kuid mitte märkimisväärselt. Peame arvesse võtma ka teisi tunnuseid!

> Näed, et kõrvitsate hinnad on minimaalsed kusagil õuduspeo (Halloween) ajal. Kuidas seda seletada?

🎃 Palju õnne, sa just lõid mudeli, mis aitab prognoosida kookkõrvitsate hinda. Saame tõenäoliselt sama protseduuri korrata kõigi kõrvitsaliikide jaoks, aga see oleks tüütu. Õpime nüüd, kuidas mudelis arvestada kõrvitsa sorti!

## Kategoorilised tunnused

Ideaalis tahame suuta prognoosida erinevate kõrvitsaliikide hindu sama mudeli abil. Kuid `Variety` (sort) veerg erineb sellistest veergudest nagu `Month` (kuu), sest see sisaldab mitte-arvulisi väärtusi. Selliseid veerge nimetatakse **kategoorilisteks**.

[![ML algajatele - kategooriliste tunnuste prognoosimine lineaarse regressiooniga](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML algajatele - kategooriliste tunnuste prognoosimine lineaarse regressiooniga")

> 🎥 Klõpsa ülaloleval pildil, et näha lühikest videot kategooriliste tunnuste kasutamisest.

Siin näed, kuidas keskmine hind sõltub sordist:

<img alt="Keskmine hind sortide kaupa" src="../../../../translated_images/et/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Sordi arvesse võtmiseks peame esmalt selle numbriliseks muutma ehk **kodeerima**. Selleks on mitmeid võimalusi:

* Lihtne **numbriline kodeerimine** loob nimekirja erinevatest sortidest ja asendab seejärel sordinime selle nimekirja indeksi vastu. See pole lineaarse regressiooni jaoks parim idee, sest lineaarne regressioon kasutab indeksi tegelikku numbrilist väärtust ja lisab seda tulemusele teatud koefitsiendiga. Meie puhul on seos indeksi numbri ja hinna vahel selgelt mittelineaarne, isegi kui indekseid järjestada kindlal viisil.
* **One-hot kodeerimine** asendab `Variety` veeru nelja eraldi veeruga, ühe iga sordi jaoks. Igas veerus on väärtuseks `1`, kui vastav rida on antud sorti, ja `0` muul juhul. See tähendab, et lineaarse regressiooni jaoks on neli koefitsienti, üks iga kõrvitsaliigi jaoks, mis vastutab antud sordi "algusehinna" (või pigem "täiendava hinna") eest.

Alljärgnev kood näitab, kuidas sorti one-hot kodeerida:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | SEGASED PÄRANDLIIGID | KOOKKÕRVITS
----|-----------|-----------|---------------------|------------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Et kasutada lineaarset regressiooni one-hot kodeeritud sortide põhjal, peame lihtsalt õigesti algatama andmed `X` ja `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Ülejäänud kood on sama, mida kasutati eelnevalt lineaarse regressiooni treenimiseks. Kui proovida, näeme, et ruutkeskmine viga on umbes sama, kuid determinatsioonikordaja on palju kõrgem (~77%). Veelgi täpsemate prognooside saamiseks võime arvesse võtta rohkem kategoorilisi tunnuseid ning ka numbrilisi tunnuseid nagu `Month` või `DayOfYear`. Kõigi tunnuste ühtseks massiiviks saamiseks võime kasutada `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Siin võtame arvesse ka `City` ja `Package` tüüpi, mis annab meile MSE väärtuse 2.84 (10%) ja determinatsiooni 0.94!

## Kõik kokku

Parima mudeli koostamiseks võime kasutada ülaltoodud näite kombineeritud (one-hot kodeeritud kategoorilised + numbrilised) andmed koos polünoomse regressiooniga. Siin on mugavaks kasutamiseks täielik kood:

```python
# seadista treeningandmed
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# tee koolitus- ja testandmete jagamine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# seadista ja treeni andmetöötluskanal
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# ennusta tulemused testandmete jaoks
pred = pipeline.predict(X_test)

# arvuta keskmine ruutviga ja määramisaste
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

See peaks andma parima determinatsioonikordaja, mis on peaaegu 97%, ning MSE=2.23 (~8% ennustuse viga).

| Mudel | MSE | Determinatsioon |
|-------|-----|-----------------|
| `DayOfYear` lineaarne | 2.77 (17.2%) | 0.07 |
| `DayOfYear` polünoomne | 2.73 (17.0%) | 0.08 |
| `Variety` lineaarne | 5.24 (19.7%) | 0.77 |
| Kõik tunnused lineaarne | 2.84 (10.5%) | 0.94 |
| Kõik tunnused polünoomne | 2.23 (8.25%) | 0.97 |

🏆 Väga hästi! Sa lõid nelja mudelit ühes õppetükis ja parandasid mudeli kvaliteedi 97% peale. Regressiooni viimases osas õpid logistilisest regressioonist, mida kasutatakse kategooriate määramiseks.

---
## 🚀Väljakutse

Proovi selles märkmes mitu erinevat muutujat, et näha, kuidas korrelatsioon vastab mudeli täpsusele.

## [Loengu järgse test](https://ff-quizzes.netlify.app/en/ml/)

## Kordamine & iseseisev õppimine

Selles õppetükis õppisime lineaarset regressiooni. On ka teisi olulisi regressioonitüüpe. Loe Stepwise-, Ridge-, Lasso- ja Elasticnet-meetodite kohta. Heaks õppeallikaks on [Stanfordi statistilise õppe kursus](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Kodutöö

[Ehita mudel](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastutusest loobumine**:
See dokument on tõlgitud kasutades tehisintellekti tõlkimisteenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi püüame tagada täpsust, palun arvestage, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algset dokumenti selle emakeeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitatakse kasutada professionaalset inimtõlget. Me ei vastuta selle tõlkega seotud arusaamatuste või valesti mõistmiste eest.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->