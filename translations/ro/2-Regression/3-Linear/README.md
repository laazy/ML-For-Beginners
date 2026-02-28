# Construiește un model de regresie folosind Scikit-learn: patru moduri de regresie

## Notă pentru începători

Regresia liniară este folosită atunci când dorim să prezicem o **valoare numerică** (de exemplu, prețul unei case, temperatura sau vânzările).
Funcționează prin găsirea unei linii drepte care reprezintă cel mai bine relația dintre caracteristicile de intrare și ieșire.

În această lecție, ne concentrăm pe înțelegerea conceptului înainte de a explora tehnici mai avansate de regresie.
![Infografic regresie liniară vs polinomială](../../../../translated_images/ro/linear-polynomial.5523c7cb6576ccab.webp)
> Infografic realizat de [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Test pre-lectură](https://ff-quizzes.netlify.app/en/ml/)

> ### [Această lecție este disponibilă și în R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introducere

Până acum ai explorat ce este regresia cu date de probă culese din setul de date despre prețurile dovlecilor pe care îl vom folosi pe parcursul acestei lecții. De asemenea, ai vizualizat datele folosind Matplotlib.

Acum ești pregătit să aprofundezi regresia pentru Învățare Automată. În timp ce vizualizarea îți permite să înțelegi datele, adevărata putere a Învățării Automate provine din _antrenarea modelelor_. Modelele sunt antrenate pe date istorice pentru a captura automat dependențele din date și îți permit să prevezi rezultate pentru date noi, pe care modelul nu le-a văzut înainte.

În această lecție, vei învăța mai multe despre două tipuri de regresie: _regresia liniară de bază_ și _regresia polinomială_, alături de unele aspecte matematice care stau la baza acestor tehnici. Aceste modele ne vor permite să prezicem prețurile dovlecilor în funcție de diferite date de intrare.

[![ML pentru începători - Înțelegerea regresiei liniare](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pentru începători - Înțelegerea regresiei liniare")

> 🎥 Dă click pe imaginea de mai sus pentru un scurt videoclip despre regresia liniară.

> Pe tot parcursul acestui curriculum, presupunem un minim de cunoștințe matematice și încercăm să îl facem accesibil pentru studenți din alte domenii, așa că fii atent la note, 🧮 explicații, diagrame și alte instrumente de învățare care te vor ajuta să înțelegi.

### Pregătire prealabilă

Ar trebui să fii familiarizat până acum cu structura datelor despre dovleci pe care le examinăm. Le poți găsi preîncărcate și prelucrate în fișierul _notebook.ipynb_ al acestei lecții. În fișier, prețul dovlecilor este afișat pe bushel într-un nou tabel de date. Asigură-te că poți rula aceste jurnale în kernel-uri din Visual Studio Code.

### Pregătire

Ca o reamintire, încarci aceste date pentru a putea pune întrebări despre ele.

- Care este cel mai bun moment pentru a cumpăra dovleci?
- Ce preț pot să mă aștept pentru o cutie de dovleci miniatură?
- Ar trebui să îi cumpăr în coșuri de jumătate de bushel sau în cutii de 1 1/9 busheli?
Hai să continuăm să explorăm aceste date.

În lecția anterioară, ai creat un tabel Pandas și l-ai populat cu o parte din setul original de date, standardizând prețul pe bushel. Prin această metodă, însă, ai reușit să aduni doar aproximativ 400 de puncte de date și doar pentru lunile de toamnă.

Aruncă o privire la datele preîncărcate în jurnalul însoțitor al acestei lecții. Datele sunt preîncărcate și este reprezentat un grafic dispersie (scatterplot) inițial pentru a arăta datele pe lună. Poate putem să aflăm mai multe detalii despre natura datelor curățându-le mai mult.

## O linie de regresie liniară

După cum ai învățat în Lecția 1, scopul exercițiului de regresie liniară este să putem trasa o linie pentru a:

- **Arăta relațiile dintre variabile**. Arată relația dintre variabile
- **Face predicții**. Face predicții precise privind unde ar cădea un nou punct de date în raport cu acea linie.

Este tipic pentru **regresia celor mai mici pătrate** să traseze acest tip de linie. Termenul „Cei mai mici pătrați” se referă la procesul de minimizare a erorii totale din modelul nostru. Pentru fiecare punct de date, măsurăm distanța verticală (numită reziduală) între punctul real și linia de regresie.

Aceste distanțe le ridicăm la pătrat din două motive principale:

1. **Magnitudine, nu Direcție:** Dorim să tratăm o eroare de -5 la fel ca o eroare de +5. Ridicarea la pătrat transformă toate valorile în pozitive.

2. **Penalizarea valorilor extreme:** Ridicarea la pătrat atribuie o greutate mai mare erorilor mari, forțând linia să fie mai aproape de punctele care sunt mai îndepărtate.

Apoi adunăm toate aceste valori ridicate la pătrat. Scopul nostru este să găsim linia specifică pentru care această sumă finală este cea mai mică (valoarea posibilă cea mai mică) – de aici și numele „Ceilalți mai mici pătrați”.

> **🧮 Arată-mi matematica** 
> 
> Această linie, numită _linia de cea mai bună potrivire_, poate fi exprimată prin [o ecuație](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` este „variabila explicativă”. `Y` este „variabila dependentă”. Panta liniei este `b` iar `a` este interceptul pe axa y, care se referă la valoarea lui `Y` când `X = 0`.
>
>![calculează panta](../../../../translated_images/ro/slope.f3c9d5910ddbfcf9.webp)
>
> Mai întâi, calculează panta `b`. Infografic realizat de [Jen Looper](https://twitter.com/jenlooper)
>
> Cu alte cuvinte, referindu-ne la întrebarea originală din datele noastre despre dovleci: „prezice prețul unui dovleac pe bushel în funcție de lună”, `X` se referă la preț, iar `Y` la luna vânzării.
>
>![completează ecuația](../../../../translated_images/ro/calculation.a209813050a1ddb1.webp)
>
> Calculează valoarea lui Y. Dacă plătești în jur de 4 dolari, trebuie să fie aprilie! Infografic realizat de [Jen Looper](https://twitter.com/jenlooper)
>
> Matematica care calculează linia trebuie să demonstreze panta liniei, care depinde și de intercept, adică locul unde se află `Y` când `X = 0`.
>
> Poți observa metoda de calcul pentru aceste valori pe site-ul [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). De asemenea, vizitează [acest calculator de cei mai mici pătrați](https://www.mathsisfun.com/data/least-squares-calculator.html) pentru a vedea cum valorile numerelor influențează linia.

## Corelația

Un termen în plus de înțeles este **Coeficientul de corelație** între variabilele date X și Y. Folosind un grafic dispersie, poți vizualiza rapid acest coeficient. Un grafic cu puncte dispersate într-o linie ordonată are corelație mare, iar un grafic cu puncte dispersate peste tot între X și Y are corelație scăzută.

Un model bun de regresie liniară va fi unul care are un coeficient de corelație ridicat (mai aproape de 1 decât de 0) folosind metoda regresiei celor mai mici pătrate cu o linie de regresie.

✅ Rulează jurnalul însoțitor acestei lecții și uită-te la graficul dispersie Lună vs. Preț. Datele care leagă Luna de Preț pentru vânzările de dovleci par să aibă corelație mare sau mică, conform interpretării tale vizuale a graficului? Se schimbă asta dacă folosești o măsurătoare mai detaliată în loc de `Month`, de ex. *ziua anului* (adică numărul de zile de la începutul anului)?

În codul de mai jos, vom presupune că am curățat datele și am obținut un tabel de date numit `new_pumpkins`, similar cu următorul:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Codul pentru curățarea datelor este disponibil în [`notebook.ipynb`](notebook.ipynb). Am efectuat aceiași pași de curățare ca în lecția precedentă și am calculat coloana `DayOfYear` folosind expresia următoare:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Acum că ai o înțelegere a matematicii din spatele regresiei liniare, să creăm un model de regresie pentru a vedea dacă putem prezice care pachet de dovleci va avea cele mai bune prețuri. Cineva care cumpără dovleci pentru o patch de dovleci de sărbători ar putea dori această informație pentru a-și optimiza achizițiile de pachete de dovleci pentru patch.

## Căutând corelație

[![ML pentru începători - Căutând Corelația: Cheia regresiei liniare](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pentru începători - Căutând Corelația: Cheia regresiei liniare")

> 🎥 Dă click pe imaginea de mai sus pentru un scurt videoclip despre corelație.

Din lecția precedentă probabil ai observat că prețul mediu pentru diferite luni arată astfel:

<img alt="Prețul mediu pe lună" src="../../../../translated_images/ro/barchart.a833ea9194346d76.webp" width="50%"/>

Acest lucru sugerează că ar trebui să existe o corelație, și putem încerca să antrenăm un model de regresie liniară pentru a prezice relația dintre `Month` și `Price`, sau dintre `DayOfYear` și `Price`. Iată graficul dispersie care arată această a doua relație:

<img alt="Grafic dispersie al Prețului vs Ziua anului" src="../../../../translated_images/ro/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />

Să vedem dacă există corelație folosind funcția `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Se pare că corelația este destul de mică, -0.15 pentru `Month` și -0.17 pentru `DayOfMonth`, dar ar putea exista o altă relație importantă. Se pare că există clustere diferite de prețuri corespunzătoare diferitelor soiuri de dovleci. Pentru a confirma această ipoteză, să desenăm fiecare categorie de dovleci cu o culoare diferită. Transmitând parametrul `ax` către funcția de plotare `scatter` putem plota toate punctele pe același grafic:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Grafic dispersie al Prețului vs Ziua anului" src="../../../../translated_images/ro/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />

Investigația noastră sugerează că soiul are un efect mai mare asupra prețului total decât data efectivă a vânzării. Putem vedea asta cu un grafic cu bare:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Grafic cu bare al prețului vs soiului" src="../../../../translated_images/ro/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Să ne concentrăm momentan doar pe un soi de dovleci, cel „pie type”, și să vedem ce efect are data asupra prețului:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Grafic dispersie al Prețului vs Ziua anului" src="../../../../translated_images/ro/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />

Dacă acum calculăm corelația dintre `Price` și `DayOfYear` folosind funcția `corr`, vom obține ceva de genul `-0.27` - ceea ce înseamnă că antrenarea unui model predictiv are sens.

> Înainte de a antrena un model de regresie liniară, este important să ne asigurăm că datele sunt curate. Regresia liniară nu funcționează bine cu valori lipsă, deci este recomandat să eliminăm toate celulele goale:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

O altă abordare ar fi să completăm acele valori lipsă cu valorile medii din coloana corespunzătoare.

## Regresie liniară simplă

[![ML pentru începători - Regresie liniară și polinomială folosind Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pentru începători - Regresie liniară și polinomială folosind Scikit-learn")

> 🎥 Dă click pe imaginea de mai sus pentru un scurt videoclip despre regresia liniară și polinomială.

Pentru a antrena modelul nostru de regresie liniară vom folosi biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Începem prin separarea valorilor de intrare (caracteristici) și a ieșirii așteptate (etichetei) în array-uri numpy separate:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Observă că a fost necesară folosirea metodei `reshape` pe datele de intrare pentru ca pachetul de regresie liniară să le înțeleagă corect. Regresia liniară așteaptă un array 2D ca intrare, unde fiecare rând al array-ului corespunde unui vector de caracteristici de intrare. În cazul nostru, deoarece avem o singură intrare, avem nevoie de un array cu forma N&times;1, unde N este dimensiunea setului de date.

Apoi, trebuie să împărțim datele în seturi de antrenament și test pentru a valida modelul după antrenare:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

În cele din urmă, antrenarea modelelor propriu-zise de regresie liniară se face în doar două linii de cod. Definim obiectul `LinearRegression`, și îl potrivim pe date folosind metoda `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Obiectul `LinearRegression` după `fit`-are conține toți coeficienții regresiei, care pot fi accesați folosind proprietatea `.coef_`. În cazul nostru, există doar un coeficient, care ar trebui să fie în jur de `-0.017`. Aceasta înseamnă că prețurile par să scadă puțin în timp, dar nu prea mult, în jur de 2 cenți pe zi. De asemenea, putem accesa punctul de intersecție al regresiei cu axa Y folosind `lin_reg.intercept_` - acesta va fi în jur de `21` în cazul nostru, indicând prețul la începutul anului.

Pentru a vedea cât de precis este modelul nostru, putem prezice prețurile pe un set de date de test, apoi putem măsura cât de aproape sunt predicțiile noastre de valorile așteptate. Acest lucru se poate face folosind metrica eroarea pătratică medie (MSE), care este media tuturor diferențelor pătrate dintre valori așteptate și valori prezise.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Eroarea noastră pare să fie în jur de 2 puncte, ceea ce este ~17%. Nu prea bine. Un alt indicator al calității modelului este **coeficientul de determinare**, care poate fi obținut astfel:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Dacă valoarea este 0, înseamnă că modelul nu ia în considerare datele de intrare și acționează ca *cel mai slab predictor liniar*, care este pur și simplu o valoare medie a rezultatului. Valoarea 1 înseamnă că putem prezice perfect toate rezultatele așteptate. În cazul nostru, coeficientul este în jur de 0.06, ceea ce este destul de scăzut.

Putem, de asemenea, să reprezentăm grafic datele de test împreună cu linia de regresie pentru a vedea mai bine cum funcționează regresia în cazul nostru:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/ro/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regresie Polinomială

Un alt tip de Regressie Liniară este Regresia Polinomială. Deși uneori există o relație liniară între variabile - cu cât dovleacul este mai mare ca volum, cu atât este mai mare prețul - uneori aceste relații nu pot fi reprezentate ca un plan sau o linie dreaptă.

✅ Iată [câteva exemple suplimentare](https://online.stat.psu.edu/stat501/lesson/9/9.8) de date care ar putea necesita Regresie Polinomială

Privim din nou relația dintre Data și Preț. Pare acest scatterplot să trebuiască neapărat analizat printr-o linie dreaptă? Nu pot prețurile fluctua? În acest caz, poți încerca regresia polinomială.

✅ Polinoamele sunt expresii matematice care pot consta din unul sau mai mulți termeni și coeficienți

Regresia polinomială creează o curbă pentru a se potrivi mai bine datelor neliniare. În cazul nostru, dacă includem o variabilă pătratică `DayOfYear` în datele de intrare, ar trebui să putem ajusta datele noastre cu o curbă parabolică, care va avea un minim într-un anumit punct din an.

Scikit-learn include un util [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) pentru a combina diferite etape de procesare a datelor împreună. Un **pipeline** este un lanț de **estimatori**. În cazul nostru, vom crea un pipeline care mai întâi adaugă caracteristici polinomiale modelului, apoi antrenează regresia:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Folosind `PolynomialFeatures(2)` înseamnă că vom include toți polinomii de gradul doi din datele de intrare. În cazul nostru, aceasta înseamnă doar `DayOfYear`<sup>2</sup>, dar având două variabile de intrare X și Y, aceasta va adăuga X<sup>2</sup>, XY și Y<sup>2</sup>. Putem de asemenea să folosim polinoame de grad mai înalt dacă dorim.

Pipeline-urile pot fi folosite în același mod ca obiectul original `LinearRegression`, adică putem face `fit` pe pipeline, apoi folosi `predict` pentru a obține rezultatele predicției. Iată graficul care arată datele de test și curba de aproximare:

<img alt="Polynomial regression" src="../../../../translated_images/ro/poly-results.ee587348f0f1f60b.webp" width="50%" />

Folosind Regresia Polinomială, putem obține o MSE puțin mai mică și un coeficient de determinare mai mare, dar fără o îmbunătățire semnificativă. Trebuie să luăm în considerare și alte caracteristici!

> Se poate observa că prețurile minime ale dovlecilor sunt observate în jur de Halloween. Cum ai putea explica acest lucru?

🎃 Felicitări, tocmai ai creat un model care poate ajuta la prezicerea prețului dovlecilor pentru plăcintă. Probabil poți repeta aceeași procedură pentru toate tipurile de dovleci, dar asta ar fi obositor. Acum să învățăm cum să luăm în calcul soiul de dovleac în modelul nostru!

## Caracteristici categorice

În lumea ideală, vrem să putem prezice prețurile pentru diferite soiuri de dovleci folosind același model. Totuși, coloana `Variety` este oarecum diferită de coloane precum `Month`, pentru că conține valori nenumerice. Astfel de coloane se numesc **categorice**.

[![ML pentru începători - Predicții cu caracteristici categorice în regresia liniară](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pentru începători - Predicții cu caracteristici categorice în regresia liniară")

> 🎥 Fă click pe imaginea de mai sus pentru un scurt videoclip despre utilizarea caracteristicilor categorice.

Aici poți vedea cum prețul mediu depinde de soi:

<img alt="Average price by variety" src="../../../../translated_images/ro/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Pentru a lua în calcul soiul, mai întâi trebuie să-l convertim în formă numerică, sau să-l **encodăm**. Există mai multe moduri de a face asta:

* O simplă **codificare numerică** va crea un tabel cu soiurile diferite și apoi va înlocui numele soiului cu un index din acel tabel. Aceasta nu este o idee bună pentru regresia liniară, deoarece regresia liniară ia valoarea numerică efectivă a indexului și o adaugă în rezultat, înmulțind cu un coeficient. În cazul nostru, relația dintre numărul indexului și preț este clar neliniară, chiar dacă ne asigurăm că indicii sunt ordonați într-un anumit mod.
* **Codificarea one-hot** va înlocui coloana `Variety` cu 4 coloane diferite, câte una pentru fiecare soi. Fiecare coloană va conține `1` dacă rândul respectiv este dintr-un anumit soi și `0` în caz contrar. Aceasta înseamnă că vor exista patru coeficienți în regresia liniară, câte unul pentru fiecare soi de dovleac, responsabili pentru „prețul de bază” (sau mai degrabă „prețul suplimentar”) pentru acel soi în particular.

Codul de mai jos arată cum putem aplica one-hot encoding pentru soi:

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

Pentru a antrena regresia liniară folosind soiul one-hot encoded drept intrare, trebuie doar să inițializăm corect datele X și y:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Restul codului este la fel ca cel folosit mai sus pentru a antrena regresia liniară. Dacă încerci, vei vedea că eroarea pătratică medie este cam aceeași, dar coeficientul de determinare crește mult (~77%). Pentru predicții și mai precise, putem lua în calcul mai multe caracteristici categorice, precum și caracteristici numerice, cum ar fi `Month` sau `DayOfYear`. Pentru a obține un singur set mare de caracteristici, putem folosi `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aici luăm în considerare și `City` și tipul `Package`, ceea ce ne oferă MSE 2.84 (10%) și coeficient de determinare 0.94!

## Combinând totul

Pentru a face cel mai bun model, putem folosi date combinate (categorice one-hot encoded + numerice) din exemplul de mai sus împreună cu Regresia Polinomială. Iată codul complet pentru conveniența ta:

```python
# configurare date de antrenament
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# realizare divizare antrenament-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurare și antrenare pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prezice rezultate pentru datele de test
pred = pipeline.predict(X_test)

# calculare MSE și coeficient de determinare
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Aceasta ar trebui să ne ofere cel mai bun coeficient de determinare de aproape 97% și MSE=2.23 (~8% eroare de predicție).

| Model | MSE | Determinare |
|-------|-----|-------------|
| Linear `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomial `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Linear `Variety` | 5.24 (19.7%) | 0.77 |
| Linear toate caracteristicile | 2.84 (10.5%) | 0.94 |
| Polinomial toate caracteristicile | 2.23 (8.25%) | 0.97 |

🏆 Foarte bine! Ai creat patru modele de Regresie într-o singură lecție și ai îmbunătățit calitatea modelului la 97%. În secțiunea finală despre Regresie vei învăța despre Regresia Logistică pentru determinarea categoriilor.

---
## 🚀Provocare

Testează mai multe variabile diferite în acest notebook pentru a vedea cum se corelează acestea cu acuratețea modelului.

## [Test post-lector](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Auto-studiu

În această lecție am învățat despre Regresia Liniară. Există și alte tipuri importante de Regresie. Citește despre tehnicile Stepwise, Ridge, Lasso și Elasticnet. Un curs bun pentru aprofundare este [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Temă

[Build a Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinarea responsabilității**:  
Acest document a fost tradus folosind serviciul de traducere automată AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa de origine trebuie considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm nicio responsabilitate pentru eventualele neînțelegeri sau interpretări eronate care pot rezulta din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->