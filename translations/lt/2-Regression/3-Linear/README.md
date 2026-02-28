# Sukurkite regresijos modelį naudodami Scikit-learn: regresija keturiais būdais

## Pradedančiųjų pastaba

Linijinė regresija naudojama, kai norime nuspėti **skaitinę reikšmę** (pavyzdžiui, namo kainą, temperatūrą ar pardavimus).
Ji veikia ieškodama tiesės, kuri geriausiai atspindi ryšį tarp įvesties požymių ir išvesties.

Šiame pamokoje daugiausia dėmesio skiriame koncepcijos supratimui, prieš nagrinėjant pažangesnes regresijos technikas.
![Linijinė ir polininė regresija infografika](../../../../translated_images/lt/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika autoriaus [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Priešpaskaitinis testas](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka taip pat prieinama R kalba!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Įvadas

Iki šiol jūs susipažinote, kas yra regresija, naudodami pavyzdinius duomenis iš moliūgų kainų rinkinio, kurį naudosime per visą pamoką. Taip pat juos vizualizavote naudodami Matplotlib.

Dabar esate pasiruošę giliau gilintis į regresiją ML kontekste. Nors vizualizacija leidžia geriau suprasti duomenis, tikroji Mašininio Mokymosi galia kyla iš _modelių mokymo_. Modeliai mokomi pagal istorinį duomenų rinkinį, kad automatiškai užfiksuotų duomenų priklausomybes ir leistų prognozuoti rezultatus naujiems, anksčiau nematytiems duomenims.

Šioje pamokoje sužinosite daugiau apie du regresijos tipus: _pagrindinę linijinę regresiją_ ir _polininę regresiją_, taip pat apie dalį matematikos, esančios šių metodų pagrindu. Šie modeliai leis mums prognozuoti moliūgų kainas, remiantis skirtingais įvesties duomenimis.

[![ML pradedantiesiems - Linijinės regresijos supratimas](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pradedantiesiems - Linijinės regresijos supratimas")

> 🎥 Paspauskite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą linijinės regresijos apžvalgą.

> Per visą šią kurso programą mes laikome, kad matematikos žinios minimalios ir stengiamės jas padaryti prieinamas studentams iš kitų sričių, todėl atkreipkite dėmesį į pastabas, 🧮 iškylamuosius langus, diagramas ir kitus mokymosi įrankius, padedančius suvokimui.

### Reikalavimai

Jau turėtumėte būti susipažinę su moliūgų duomenų struktūra, kurią nagrinėjame. Ją galite rasti iš anksto įkeltą ir išvalytą šios pamokos _notebook.ipynb_ faile. Faile moliūgų kaina rodoma už bushelį naujame duomenų rinkinyje. Užtikrinkite, kad galite paleisti šiuos užrašų knygelių failus Visual Studio Code branduoliuose.

### Paruošimas

Primename, kad duomenis įkeliame tam, kad galėtume užduoti klausimus.

- Kada geriausia pirkti moliūgus?
- Kokios kainos galiu tikėtis už iešminius moliūgus dėžėje?
- Ar pirkti juos per pusės bushelio krepšius, ar per 1 1/9 bushelio dėžutes?
Toliau gilinamės į šiuos duomenis.

Praeitoje pamokoje sukūrėte Pandas duomenų rinkinį ir užpildėte jį dalimi originalių duomenų, standartizuodami kainas pagal bushelį. Tačiau tai leido surinkti tik apie 400 duomenų taškų ir tik rudens mėnesiams.

Pažiūrėkite į duomenis, kuriuos iš anksto įkėlėme šios pamokos užrašų knygelėje. Duomenys įkelti ir parodytas pradinis sklaidos grafikas rodo mėnesio duomenis. Galbūt galime dar šiek tiek plačiau pažvelgti į duomenų pobūdį, juos dar labiau išvalydami.

## Linijinės regresijos tiesė

Kaip sužinojote 1-pamokoje, linijinės regresijos užduotis yra nubraižyti tiesę, kuri:

- **Atvaizduoja kintamųjų ryšius**. Parodo sąryšį tarp kintamųjų
- **Leidžia prognozuoti**. Tiksliai numatyti, kur naujas duomenų taškas kris santykyje su ta tiesė.

 Įprasta **mažiausių kvadratų regresijoje** piešti tokį tiesiųjį grafiką. Terminas „Mažiausių kvadratų“ reiškia procesą, kurio metu minimalizuojama bendroji klaida modelyje. Kiekvienam duomenų taškui matuojame vertikalų atstumą (vadinamą likučiu) tarp tikrosios reikšmės ir mūsų regresijos linijos.

Šiuos atstumus kvadratuojame dėl dviejų pagrindinių priežasčių:

1. **Dydžio svarba, o ne kryptis:** Norime, kad klaida -5 būtų vertinama taip pat kaip +5. Kvadratuojant visos reikšmės tampa teigiamos.

2. **Išskirtinių atvejų bausmė:** Kvadratuojant didelės klaidos įgauna didesnį svorį, dėl to tiesė stengiasi būti arčiau toli esančių taškų.

Tada sudedame visus kvadratuotus atstumus. Mūsų tikslas – rasti tą tiesę, kuri minimizuoja šį sumą (mažiausia įmanoma reikšmė) – todėl vadinama „mažiausių kvadratų“ metodu.

> **🧮 Parodykite man matematiką**  
>  
> Ši tiesė, vadinama _geriausiai pritaikyta tiesė_, gali būti išreikšta [lygtimi](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>  
> ```
> Y = a + bX
> ```
>
> `X` yra 'paaiškinamasis kintamasis'. `Y` yra 'priklausomas kintamasis'. Tiesės nuolydis yra `b`, o `a` yra y-interceptas, kuris reiškia `Y` reikšmę, kai `X = 0`.  
>
>![nuolydžio skaičiavimas](../../../../translated_images/lt/slope.f3c9d5910ddbfcf9.webp)
>
> Pirmiausia apskaičiuojame nuolydį `b`. Infografika autoriaus [Jen Looper](https://twitter.com/jenlooper)
>
> Kitaip tariant, kalbant apie mūsų moliūgų duomenų pradinį klausimą: „prognozuoti moliūgų kainą už bushelį pagal mėnesį“, `X` reikštų kainą, o `Y` būtų pardavimo mėnuo.
>
>![lygtys užbaigimas](../../../../translated_images/lt/calculation.a209813050a1ddb1.webp)
>
> Apskaičiuokite `Y` reikšmę. Jei mokate apie 4 USD, tai turi būti balandis! Infografika autoriaus [Jen Looper](https://twitter.com/jenlooper)
>
> Matematikos formule turi parodyti linijos nuolydį, kuris taip pat priklauso nuo sankirtos, arba kur `Y` yra, kai `X=0`.
>
> Galite stebėti skaičiavimo metodą šioje svetainėje [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Taip pat apsilankykite [Šiame mažiausių kvadratų skaičiuoklyje](https://www.mathsisfun.com/data/least-squares-calculator.html), kad pamatytumėte, kaip skaičių vertės veikia tiesę.

## Koreliacija

Dar vienas svarbus terminas yra **koreliacijos koeficientas** tarp tam tikrų X ir Y kintamųjų. Naudodami sklaidos grafiką greitai galite vizualizuoti šį koeficientą. Grafikas su duomenų taškais, susitelkusiais į tvarkingą liniją, turi aukštą koreliaciją, o su duomenų taškais išsibarsčiusiais tarp X ir Y - žemą koreliaciją.

Geras linijinės regresijos modelis turi būti toks, kurio koreliacijos koeficientas pagal mažiausių kvadratų regresijos metodą yra aukštas (arčiau 1 nei 0).

✅ Paleiskite šios pamokos užrašų knygelę ir pažiūrėkite į Month–Price sklaidos grafiką. Ar duomenys, susiejantys mėnesį su moliūgų kainomis, atrodo turintys aukštą ar žemą koreliaciją pagal jūsų vizualinę sklaidos grafiką? Ar tai pasikeičia, jei vietoje `Month` panaudojate smulkesnį matavimą, pvz., *metų dieną* (t.y. dienų skaičių nuo metų pradžios)?

Toliau pateiktame kode skelbiame, kad duomenys buvo sutvarkyti ir mes turime duomenų rėmelį `new_pumpkins`, panašų į šį:

ID | Mėnuo | MetųDiena | Veislė | Miestas | Pakuotė | Žemiausia kaina | Aukščiausia kaina | Kaina
---|-------|-----------|---------|---------|---------|----------------|------------------|-------
70 | 9     | 267       | PYRAGO TIPO  | BALTIMORA   | 1 1/9 bushelio dėžutės | 15.0           | 15.0             | 13.636364
71 | 9     | 267       | PYRAGO TIPO  | BALTIMORA   | 1 1/9 bushelio dėžutės | 18.0           | 18.0             | 16.363636
72 | 10    | 274       | PYRAGO TIPO  | BALTIMORA   | 1 1/9 bushelio dėžutės | 18.0           | 18.0             | 16.363636
73 | 10    | 274       | PYRAGO TIPO  | BALTIMORA   | 1 1/9 bushelio dėžutės | 17.0           | 17.0             | 15.454545
74 | 10    | 281       | PYRAGO TIPO  | BALTIMORA   | 1 1/9 bushelio dėžutės | 15.0           | 15.0             | 13.636364

> Duomenų valymo kodas yra prieinamas faile [`notebook.ipynb`](notebook.ipynb). Atlikome tuos pačius valymo žingsnius kaip ir ankstesnėje pamokoje ir apskaičiavome `DayOfYear` stulpelį naudodami šią išraišką: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Dabar, kai suprantate linijinės regresijos matematinius pagrindus, sukurkime Regresijos modelį, kad sužinotume, ar galime prognozuoti, kuri moliūgų pakuotė turės geriausias kainas. Kas nors, perkantis moliūgus šventiniam moliūgų laukui, norėtų turėti šią informaciją, kad galėtų optimizuoti moliūgų pirkimus.

## Koreliacijos paieška

[![ML pradedantiesiems - Koreliacijos paieška: raktas į linijinę regresiją](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pradedantiesiems - Koreliacijos paieška: raktas į linijinę regresiją")

> 🎥 Paspauskite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą koreliacijos apžvalgą.

Iš ankstesnės pamokos greičiausiai jau matėte, kad vidutinė kainų tendencija pagal mėnesius atrodo maždaug taip:

<img alt="Vidutinė kaina pagal mėnesį" src="../../../../translated_images/lt/barchart.a833ea9194346d76.webp" width="50%"/>

Tai rodo, kad turėtų būti kažkokia koreliacija, ir galime bandyti apmokyti linijinės regresijos modelį prognozuoti ryšį tarp `Month` ir `Price` arba tarp `DayOfYear` ir `Price`. Štai sklaidos grafikas, rodantis pastarąjį ryšį:

<img alt="Sklaidos grafikas: Kaina prieš Metų dieną" src="../../../../translated_images/lt/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />

Pažiūrėkime, ar yra koreliacija naudodami funkciją `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Atrodo, kad koreliacija gana maža, -0,15 pagal `Month` ir -0,17 pagal `DayOfMonth`, bet gali būti kitas svarbus ryšys. Atrodo, kad kainų grupės atitinka skirtingas moliūgų veisles. Norėdami patvirtinti šią hipotezę, nubraižykime kiekvieną moliūgų kategoriją skirtingomis spalvomis. Paduodami `ax` parametrą funkcijai `scatter`, galime nubraižyti visus taškus vienoje diagramoje:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Sklaidos grafikas: Kaina prieš Metų dieną su spalvomis" src="../../../../translated_images/lt/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />

Mūsų tyrimas rodo, kad veislė labiau veikia bendrą kainą nei tikroji pardavimo data. Tai galime pamatyti ir juostinėje diagramoje:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Juostinė diagrama: kaina pagal veislę" src="../../../../translated_images/lt/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Tuo tarpu sutelkime dėmesį tik į vieną moliūgų veislę, ‘pie type’, ir pažiūrėkime, kokį poveikį data turi kainai:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Sklaidos grafikas: Kaina prieš Metų dieną, tik pie type" src="../../../../translated_images/lt/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />

Jei dabar apskaičiuosime koreliaciją tarp `Price` ir `DayOfYear` panaudodami funkciją `corr`, gausime kažką panašaus į `-0.27` – o tai reiškia, kad verta apmokyti prognozuojamą modelį.

> Prieš pradėdami apmokyti linijinės regresijos modelį, svarbu įsitikinti, kad duomenys yra švarūs. Linijinė regresija prastai veikia su trūkstamomis reikšmėmis, todėl verta pašalinti visas tuščias langelius:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Kitas būdas - užpildyti tuščias reikšmes atitinkamų stulpelių vidurkiais.

## Paprasta linijinė regresija

[![ML pradedantiesiems - Linijinė ir polininė regresija naudojant Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pradedantiesiems - Linijinė ir polininė regresija naudojant Scikit-learn")

> 🎥 Paspauskite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą linijinės ir polininės regresijos apžvalgą.

Mūsų Linijinės regresijos modelio mokymui naudosime **Scikit-learn** biblioteką.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Pradėsime atskirdami įvesties reikšmes ( požymius) ir laukiamą išvestį (etiketę) į atskirus numpy masyvus:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Atkreipkite dėmesį, kad reikėjo atlikti `reshape` su įvesties duomenimis, kad Linijinės regresijos paketas suprastų juos teisingai. Linijinė regresija laukia 2D masyvo kaip įvesties, kur kiekviena masyvo eilutė atitinka požymių vektorių. Mūsų atveju, kai turime tik vieną įvestį, mums reikia masyvo formos N×1, kur N – duomenų rinkinio dydis.

Tada turime padalinti duomenis į mokymo ir testavimo rinkinius, kad galėtume modelį patikrinti po mokymo:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Galiausiai, faktinis Linijinės regresijos modelio mokymas užima tik dvi kodo eilutes. Apibrėžiame `LinearRegression` objektą ir pritaikome jį mūsų duomenims naudodami `fit` metodą:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` objektas po `fit` pritaikymo turi visus regresijos koeficientus, prie kurių galima prieiti naudodami `.coef_` savybę. Mūsų atveju yra tik vienas koeficientas, kuris turėtų būti maždaug `-0.017`. Tai reiškia, kad kainos, atrodo, šiek tiek krinta laikui bėgant, bet ne per daug, apie 2 centus per dieną. Taip pat galime prieiti prie regresijos susikirtimo su Y ašimi naudodami `lin_reg.intercept_` – mūsų atveju jis bus maždaug `21`, rodantis kainą metų pradžioje.

Norėdami pamatyti, kokia tiksliai yra mūsų modelio kokybė, galime prognozuoti kainas testiniame duomenų rinkinyje ir tada pamatuoti, kiek mūsų prognozės yra arti tikėtinų reikšmių. Tai galima padaryti naudojant vidurinės kvadratinės paklaidos (MSE) metriką, kuri yra visų kvadratinių skirtumų tarp tikėtinos ir prognozuotos reikšmės vidurkis.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Mūsų klaida atrodo apie 2 taškus, tai apie ~17%. Ne per gera. Kitas modelio kokybės indikatorius yra **nustatymo koeficientas**, kurį galima gauti taip:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Jei reikšmė yra 0, tai reiškia, kad modelis neatsižvelgia į įvesties duomenis ir veikia kaip *blogiausias tiesinis prognozuotojas*, kuris tiesiog paima vidutinę reikšmę. Reikšmė 1 reiškia, kad galime tobulai prognozuoti visas tikėtinas reikšmes. Mūsų atveju koeficientas yra apie 0.06, kas yra gan žema.

Taip pat galime nupiešti testinius duomenis kartu su regresijos linija, kad geriau matytume, kaip regresija veikia mūsų atveju:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/lt/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polinominė regresija

Kitas linijinės regresijos tipas yra polinominė regresija. Nors kartais tarp kintamųjų yra tiesinė priklausomybė – kuo didesnis moliūnas tūriu, tuo didesnė kaina – kartais šių priklausomybių negalima nubraižyti plokštuma ar tiesia linija.

✅ Čia yra [daugiau pavyzdžių](https://online.stat.psu.edu/stat501/lesson/9/9.8) duomenų, kuriems gali tikti polinominė regresija

Dar kartą pažiūrėkite į priklausomybę tarp Datos ir Kainos. Ar šis taškų debesis būtinai turėtų būti analizuojamas tiesia linija? Ar kainos negali svyruoti? Tokiu atveju galite išbandyti polinominę regresiją.

✅ Polinomai yra matematiniai išraiškos, kurios gali susidaryti iš vieno ar daugiau kintamųjų ir koeficientų

Polinominė regresija sukuria išlenktą liniją, kad geriau pritaikytų netiesinius duomenis. Mūsų atveju, jei į įvesties duomenis įtrauksime pakeltą kvadratu `DayOfYear` kintamąjį, galėsime pritaikyti duomenis parabolės formai, turinčiai minimumą metų viduje.

Scikit-learn turi naudingą [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), leidžiančią sujungti skirtingus duomenų apdorojimo žingsnius. **Pipeline** yra **lankų** seka. Mūsų atveju sukursime pipeline, kuris pirmiausia prideda polinominius požymius prie mūsų modelio ir tada treniruoja regresiją:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Naudojant `PolynomialFeatures(2)` reiškia, kad įtraukiame visus antro laipsnio polinomus iš įvesties duomenų. Mūsų atveju tai reikštų tik `DayOfYear`<sup>2</sup>, bet turint du kintamuosius X ir Y, bus pridedama X<sup>2</sup>, XY ir Y<sup>2</sup>. Taip pat galime naudoti aukštesnio laipsnio polinomus, jei norime.

Pipelines galima naudoti taip pat kaip originalų `LinearRegression` objektą, t.y. galime pritaikyti `fit` pipeline, o paskui naudoti `predict`, kad gautume prognozių rezultatus. Čia pateiktas grafikas, rodantis testinius duomenis ir aproksimacinę kreivę:

<img alt="Polynomial regression" src="../../../../translated_images/lt/poly-results.ee587348f0f1f60b.webp" width="50%" />

Naudodami polinominę regresiją galime gauti šiek tiek mažesnį MSE ir didesnį nustatymo koeficientą, bet ne žymiai. Turime atsižvelgti į daugiau požymių!

> Galite pastebėti, kad mažiausios moliūnų kainos fiksuojamos kažkur apie Heloviną. Kaip tai galėtumėte paaiškinti?

🎃 Sveikiname, ką tik sukūrėte modelį, kuris gali padėti prognozuoti pyraginių moliūnų kainą. Tikriausiai galite tą patį padaryti visoms moliūnų rūšims, bet tai būtų varginanti užduotis. Dabar išmoksime, kaip atsižvelgti į moliūnų veislę mūsų modelyje!

## Kategoriniai požymiai

Idealioje pasaulyje norėtume galėti prognozuoti kainas skirtingoms moliūnų veislėms naudojant tą patį modelį. Tačiau stulpelis `Variety` yra šiek tiek kitoks nei, pavyzdžiui, `Month`, nes jame yra ne skaitinės reikšmės. Tokie stulpeliai vadinami **kategoriniais**.

[![ML pradedantiesiems – kategorinių požymių prognozės su tiesine regresija](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pradedantiesiems – kategorinių požymių prognozės su tiesine regresija")

> 🎥 Paspauskite aukščiau esančią nuotrauką, jei norite trumpą vaizdo įrašą apie kategorinių požymių naudojimą.

Čia matote, kaip vidutinė kaina priklauso nuo veislės:

<img alt="Average price by variety" src="../../../../translated_images/lt/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Norėdami atsižvelgti į veislę, pirmiausia turime ją paversti skaitine reikšme, arba **užkoduoti** ją. Yra keli būdai, kaip tai padaryti:

* Paprastas **skaitmeninis kodavimas** sukurs skirtingų veislių lentelę, o tada veislės pavadinimą pakeis indeksu šioje lentelėje. Tai nėra geriausias sprendimas tiesinei regresijai, nes tiesinė regresija naudoja tikrąją skaitinę indekso reikšmę ir prideda ją prie rezultato pasvėrusi tam tikru koeficientu. Mūsų atveju priklausomybė tarp indekso numerio ir kainos yra aiškiai netiesinė, net jei užtikrintume, kad indeksai būtų išdėstyti tam tikra tvarka.
* **One-hot kodavimas** pakeis `Variety` stulpelį 4 skirtingais stulpeliais, po vieną kiekvienai veislei. Kiekviename stulpelyje bus `1`, jei atitinkamas įrašas yra tos veislės, ir `0` kitu atveju. Tai reiškia, kad tiesinėje regresijoje bus keturi koeficientai, po vieną kiekvienai moliūnų veislei, atsakingi už "pradinę kainą" (ar tiksliau - "papildomą kainą") konkrečiai veislei.

Žemiau pateiktas kodas, rodantis, kaip galima one-hot koduoti veislę:

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

Norėdami apmokyti tiesinę regresiją, naudodami one-hot koduotą veislę kaip įvestį, tiesiog turime teisingai inicializuoti `X` ir `y` duomenis:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Likusi kodo dalis tokia pati kaip aukščiau naudota tiesinei regresijai treniruoti. Jei išbandysite, pamatysite, kad vidutinė kvadratinė klaida bus maždaug tokia pati, bet gausime daug didesnį nustatymo koeficientą (~77%). Norėdami gauti dar tikslesnes prognozes, galime atsižvelgti į daugiau kategorinių požymių, taip pat į skaitinius požymius, tokius kaip `Month` ar `DayOfYear`. Norėdami gauti vieną didelį požymių masyvą, galime naudoti `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Čia taip pat atsižvelgiame į `City` ir `Package` tipą, kas duoda MSE 2.84 (10%) ir nustatymo koeficientą 0.94!

## Apibendrinimas

Kad sukurtume geriausią modelį, galime naudoti sujungtus (one-hot koduotus kategorinius + skaitinius) duomenis kartu su polinomine regresija. Štai viso kodo pavyzdys jūsų patogumui:

```python
# nustatyti mokymo duomenis
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# padaryti treniruočių ir testavimo skaidymą
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# sukonfigūruoti ir apmokyti procesų seką
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prognozuoti rezultatus testavimo duomenims
pred = pipeline.predict(X_test)

# apskaičiuoti vidutinę kvadratinę paklaidą ir determinacijos koeficientą
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Tai turėtų duoti geriausią nustatymo koeficientą beveik 97% ir MSE=2.23 (~8% prognozės klaida).

| Modelis | MSE | Nustatymas |
|---------|-----|------------|
| `DayOfYear` Tiesinė | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polinominė | 2.73 (17.0%) | 0.08 |
| `Variety` Tiesinė | 5.24 (19.7%) | 0.77 |
| Visi požymiai Tiesinė | 2.84 (10.5%) | 0.94 |
| Visi požymiai Polinominė | 2.23 (8.25%) | 0.97 |

🏆 Puikiai! Pamokoje sukūrėte keturis regresijos modelius ir pagerinote modelio kokybę iki 97%. Galutinėje regresijos dalyje susipažinsite su logistinę regresiją, skirtą kategorijoms nustatyti.

---
## 🚀Iššūkis

Išbandykite kelis skirtingus kintamuosius šiame sąsiuvinyje, kad pamatytumėte, kaip koreliacija atitinka modelio tikslumą.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis

Šioje pamokoje išmokome apie tiesinę regresiją. Yra ir kitų svarbių regresijos tipų. Perskaitykite apie žingsninę, „ridge“, lasso ir elasticnet technikas. Geras kursas, norint sužinoti daugiau, yra [Stanford statistinio mokymosi kursas](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Užduotis

[Sukurti modelį](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipkite dėmesį, kad automatizuoti vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojame pasitelkti profesionalų žmogišką vertimą. Mes neatsakome už bet kokius nesusipratimus ar klaidingą interpretaciją, kilusią naudojant šį vertimą.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->