# Izgradite regresijski model koristeći Scikit-learn: regresija na četiri načina

## Bilješka za početnike

Linearna regresija se koristi kada želimo predvidjeti **numeričku vrijednost** (na primjer, cijenu kuće, temperaturu ili prodaju).
Radi tako da pronalazi pravu liniju koja najbolje predstavlja odnos između ulaznih značajki i izlaza.

U ovom dijelu fokusiramo se na razumijevanje koncepta prije nego što istražimo naprednije tehnike regresije.
![Infografika linearne nasuprot polinomnoj regresiji](../../../../translated_images/hr/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ovaj je lekciju dostupan i u R-u!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Uvod 

Do sada ste istraživali što je regresija koristeći uzorak podataka prikupljenih iz skupa podataka o cijenama bundeva kojeg koristimo kroz cijelu ovu lekciju. Također ste ga vizualizirali koristeći Matplotlib.

Sada ste spremni dublje zaroniti u regresiju za ML. Dok vizualizacija omogućuje razumijevanje podataka, prava moć strojnog učenja dolazi iz _treniranja modela_. Modeli se treniraju na povijesnim podacima kako bi automatski uhvatili ovisnosti podataka, i omogućuju vam da predviđate ishode za nove podatke koje model ranije nije vidio.

U ovoj lekciji naučit ćete više o dvije vrste regresije: _osnovna linearna regresija_ i _polinomna regresija_, zajedno s dijelom matematike koja stoji iza ovih tehnika. Ti modeli će nam omogućiti predviđanje cijene bundeva ovisno o različitim ulaznim podacima.

[![ML za početnike - Razumijevanje linearne regresije](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Kliknite na slikovnu poveznicu za kratki video pregled linearne regresije.

> Kroz ovaj kurikulum pretpostavljamo minimalno matematičko znanje i nastojimo ga učiniti pristupačnim studentima iz drugih područja, pa pripazite na napomene, 🧮 objašnjenja, dijagrame i druge alate za učenje koji pomažu razumijevanju.

### Preduvjeti

Do sada biste trebali biti upoznati sa strukturom podataka o bundevama koje proučavamo. Možete ih pronaći predinstalirane i predobrijeđene u _notebook.ipynb_ datoteci ove lekcije. U datoteci se prikazuje cijena bundeve po bushelu u novom okviru podataka. Provjerite da možete pokrenuti ove bilježnice u Visual Studio Code kernelima.

### Priprema

Kao podsjetnik, učitavate ove podatke kako biste ih mogli ispitivati.

- Kada je najbolje vrijeme za kupnju bundeva? 
- Koju cijenu mogu očekivati za paket minijaturnih bundeva?
- Trebam li ih kupiti u košarama od pola bushela ili u kutiji od 1 1/9 bushela?
Nastavimo s ispitivanjem ovih podataka.

U prethodnoj ste lekciji kreirali Pandas okvir podataka i ispunili ga dijelom izvornog skupa podataka, standardizirajući cijene po bushelu. Međutim, učinili ste to samo za oko 400 podataka i samo za jesenske mjesece.

Pogledajte podatke koje smo predinstalirali u bilježnici ove lekcije. Podaci su učitani, a prikazan je početni scatterplot koji prikazuje podatke mjeseci. Možemo li možda dobiti malo više detalja o prirodi podataka čišćenjem?

## Linija linearne regresije

Kao što ste naučili u Lekciji 1, cilj linearne regresije je biti u stanju nacrtati liniju koja:

- **Prikazuje odnose među varijablama**. Prikazuje odnos između varijabli
- **Daje predviđanja**. Točno predviđa gdje bi novi podatak pao u odnosu na tu liniju.

Tipično je za **metodu najmanjih kvadrata** crtanje takve linije. Pojam "najmanjih kvadrata" odnosi se na proces minimiziranja ukupne greške u našem modelu. Za svaku točku mjerimo vertikalnu udaljenost (nazvanu residuum) između stvarne točke i naše regresijske linije.

Te udaljenosti kvadriramo zbog dva glavna razloga:

1. **Magnitude nad smjerom:** Želimo tretirati grešku -5 isto kao i grešku +5. Kvadriranje sve vrijednosti čini pozitivnima.

2. **Kaznjavanje odmetnika:** Kvadriranje daje veću težinu većim greškama, prisiljavajući liniju da ostane bliže udaljenim točkama.

Zatim zbrojimo sve te kvadrirane vrijednosti. Naš cilj je pronaći točnu liniju gdje je taj konačni zbroj najmanji (najmanja moguća vrijednost)—otuda naziv "najmanjih kvadrata."

> **🧮 Pokaži mi matematiku** 
> 
> Ta linija, nazvana _linija najboljeg pristajanja_, može se izraziti [jednadžbom](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'objašnjavajuća varijabla'. `Y` je 'ovisna varijabla'. Nagib linije je `b`, a `a` je odsječak na y-osi, što se odnosi na vrijednost `Y` kada je `X = 0`. 
>
>![izračunaj nagib](../../../../translated_images/hr/slope.f3c9d5910ddbfcf9.webp)
>
> Prvo, izračunajte nagib `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Drugim riječima, i referirajući se na naše izvornog pitanje o bundevama: "predvidjeti cijenu bundeve po bushelu prema mjesecu", `X` bi se odnosilo na cijenu, a `Y` na mjesec prodaje.
>
>![dovrši jednadžbu](../../../../translated_images/hr/calculation.a209813050a1ddb1.webp)
>
> Izračunajte vrijednost Y. Ako plaćate oko 4 dolara, mora biti travanj! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika koja izračunava liniju mora pokazati nagib linije, koji također ovisi o odsječku ili gdje se `Y` nalazi kada je `X = 0`.
>
> Metodu izračuna ovih vrijednosti možete promatrati na web stranici [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Također posjetite [ovaj kalkulator najmanjih kvadrata](https://www.mathsisfun.com/data/least-squares-calculator.html) da vidite kako vrijednosti brojeva utječu na liniju.

## Korelacija

Još jedan pojam koji treba razumjeti jest **koeficijent korelacije** između zadanih X i Y varijabli. Koristeći scatterplot, brzo možete vizualizirati ovaj koeficijent. Grafikon s točkama raspoređenim uredno u liniji ima visoku korelaciju, dok grafikon s točkama raštrkanima posvuda između X i Y ima nisku korelaciju.

Dobar model linearne regresije bit će onaj koji ima visok (bliži 1 nego 0) koeficijent korelacije koristeći metodu najmanjih kvadrata s regresijskom linijom.

✅ Pokrenite bilježnicu uz ovu lekciju i pogledajte scatterplot Mjesec prema Cijeni. Ima li podataka koji povezuju mjesec s cijenom prodaje bundeva visoku ili nisku korelaciju, prema vašoj vizualnoj interpretaciji scatterplota? Mijenja li se to ako koristite finiju mjeru umjesto `Month`, npr. *dan u godini* (broj dana od početka godine)?

U sljedećem kodu pretpostavit ćemo da smo očistili podatke i dobili okvir podataka nazvan `new_pumpkins`, sličan sljedećem:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kod za čišćenje podataka nalazi se u [`notebook.ipynb`](notebook.ipynb). Izveli smo iste korake čišćenja kao u prethodnoj lekciji te izračunali stupac `DayOfYear` koristeći sljedeći izraz:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sada kad imate razumijevanje matematike iza linearne regresije, kreirajmo regresijski model da vidimo možemo li predvidjeti koji paket bundeva će imati najbolje cijene. Netko tko kupuje bundeve za blagdanski prikaz bundeva može htjeti ovu informaciju kako bi optimizirao kupnju paketa bundeva za prikaz.

## Traženje korelacije

[![ML za početnike - Traženje korelacije: Ključ za linearnu regresiju](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Kliknite na slikovnu poveznicu za kratki video pregled korelacije.

Iz prethodne lekcije vjerojatno ste vidjeli da prosječna cijena za različite mjesece izgleda ovako:

<img alt="Prosječna cijena po mjesecu" src="../../../../translated_images/hr/barchart.a833ea9194346d76.webp" width="50%"/>

To sugerira da bi trebala postojati neka korelacija, i možemo pokušati trenirati linearni regresijski model da predvidi odnos između `Month` i `Price`, ili između `DayOfYear` i `Price`. Evo scatterplota koji prikazuje ovaj zadnji odnos:

<img alt="Scatter plot cijena u odnosu na dan u godini" src="../../../../translated_images/hr/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Pogledajmo ima li korelacije koristeći funkciju `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Izgleda da je korelacija prilično mala, -0.15 prema `Month` i -0.17 prema `DayOfMonth`, ali moglo bi postojati još neki važan odnos. Čini se da postoje različiti klasteri cijena koji odgovaraju različitim vrstama bundeva. Da bismo potvrdili ovu hipotezu, nacrtajmo svaku kategoriju bundeva različitom bojom. Prosljeđivanjem parametra `ax` funkciji `scatter` možemo nacrtati sve točke na istom grafikonu:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot cijena u odnosu na dan u godini s bojama" src="../../../../translated_images/hr/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Naša istraga sugerira da sorta ima veći utjecaj na ukupnu cijenu nego stvarni datum prodaje. To možemo vidjeti na stupčastom grafikonu:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stupčasti grafikon cijene po vrstama" src="../../../../translated_images/hr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Usredotočimo se na trenutak samo na jednu vrstu bundeve, 'pie type', i vidimo kakav utjecaj datum ima na cijenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot cijena u odnosu na dan u godini za pie vrste bundeva" src="../../../../translated_images/hr/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Ako sad izračunamo korelaciju između `Price` i `DayOfYear` koristeći funkciju `corr`, dobit ćemo nešto oko `-0.27` - što znači da ima smisla trenirati prediktivni model.

> Prije treniranja modela linearne regresije važno je osigurati da su naši podaci čisti. Linearna regresija ne funkcionira dobro s nedostajućim vrijednostima, stoga ima smisla ukloniti sve prazne ćelije:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Drugi pristup bio bi popuniti prazne vrijednosti prosječnim vrijednostima iz odgovarajućeg stupca.

## Jednostavna linearna regresija

[![ML za početnike - Linearna i polinomna regresija koristeći Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Kliknite na slikovnu poveznicu za kratki video pregled linearne i polinomne regresije.

Za treniranje našeg modela linearne regresije koristit ćemo **Scikit-learn** biblioteku.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Započinjemo razdvajanjem ulaznih vrijednosti (značajki) i očekivanog izlaza (oznake) u zasebne numpy nizove:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Primijetite da smo morali napraviti `reshape` ulaznih podataka kako bi Linear Regression paket ispravno razumio podatke. Linearna regresija očekuje 2D niz kao ulaz, gdje svaki redak niza odgovara vektoru ulaznih značajki. U našem slučaju, budući da imamo samo jedan ulaz, potrebna nam je matrica dimenzija N&times;1, gdje je N veličina skupa podataka.

Zatim trebamo podijeliti podatke na trening i test skupove, kako bismo mogli validirati naš model nakon treniranja:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Na kraju, treniranje stvarnog modela linearne regresije traje samo dvije linije koda. Definiramo objekt `LinearRegression` i prilagođavamo ga našim podacima korištenjem metode `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` nakon `fit`-iranja sadrži sve koeficijente regresije, kojima se može pristupiti pomoću `.coef_` svojstva. U našem slučaju postoji samo jedan koeficijent, koji bi trebao biti oko `-0.017`. To znači da cijene izgledaju kao da malo padaju s vremenom, ali ne previše, oko 2 centa dnevno. Također možemo pristupiti sjecištu regresije s Y-osi pomoću `lin_reg.intercept_` - ono će biti oko `21` u našem slučaju, što označava cijenu na početku godine.

Da bismo vidjeli koliko je naš model točan, možemo predvidjeti cijene na testnom skupu podataka, a zatim izmjeriti koliko su naša predviđanja bliska očekivanim vrijednostima. To se može učiniti pomoću metrike srednje kvadratne pogreške (MSE), što je srednja vrijednost svih kvadratnih razlika između očekivane i predviđene vrijednosti.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Naša pogreška izgleda da je oko 2 boda, što je ~17%. Nije baš dobro. Još jedan pokazatelj kvalitete modela je **koeficijent determinacije**, koji se može dobiti ovako:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Ako je vrijednost 0, to znači da model ne uzima u obzir ulazne podatke i djeluje kao *najgori linearni prediktor*, što je jednostavno srednja vrijednost rezultata. Vrijednost 1 znači da možemo savršeno predvidjeti sve očekivane izlaze. U našem slučaju, koeficijent je oko 0.06, što je prilično nisko.

Također možemo prikazati testne podatke zajedno s regresijskom linijom kako bismo bolje vidjeli kako regresija radi u našem slučaju:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/hr/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polinomska regresija

Druga vrsta linearne regresije je polinomska regresija. Dok ponekad postoji linearna veza između varijabli - što je bundeva veća po volumenu, to je viša cijena - ponekad ove veze ne mogu biti prikazane kao ravnina ili ravna linija.

✅ Evo [još nekoliko primjera](https://online.stat.psu.edu/stat501/lesson/9/9.8) podataka koji bi mogli koristiti polinomsku regresiju

Pogledajte ponovno odnos između Datuma i Cijene. Izgleda li ovaj scatterplot kao da ga nužno treba analizirati ravnom linijom? Ne mogu li cijene varirati? U tom slučaju možete pokušati polinomsku regresiju.

✅ Polinomi su matematički izrazi koji mogu sadržavati jednu ili više varijabli i koeficijenata.

Polinomska regresija stvara zakrivljenu liniju koja bolje pristaje nelinearnim podacima. U našem slučaju, ako u ulazne podatke uključimo kvadratnu varijablu `DayOfYear`, trebali bismo moći modelirati podatke parabolom koja će imati minimum u nekoj točki tijekom godine.

Scikit-learn uključuje korisno [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) za kombiniranje različitih koraka obrade podataka zajedno. **Pipeline** je lanac **procjenitelja**. U našem slučaju, stvorit ćemo pipeline koji prvo dodaje polinomske značajke našem modelu, a zatim trenira regresiju:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Korištenje `PolynomialFeatures(2)` znači da ćemo uključiti sve polinome drugog stupnja iz ulaznih podataka. U našem slučaju to će jednostavno značiti `DayOfYear`<sup>2</sup>, ali s dvije ulazne varijable X i Y, to će dodati X<sup>2</sup>, XY i Y<sup>2</sup>. Također možemo koristiti polinome višeg stupnja ako želimo.

Pipelinove se mogu koristiti na isti način kao izvorni objekt `LinearRegression`, tj. možemo `fit`-ati pipeline, a zatim koristiti `predict` za dobivanje predviđanja. Evo grafikona koji prikazuje testne podatke i aproksimacijsku krivulju:

<img alt="Polynomial regression" src="../../../../translated_images/hr/poly-results.ee587348f0f1f60b.webp" width="50%" />

Korištenjem polinomske regresije možemo dobiti nešto nižu MSE i viši koeficijent determinacije, ali ne značajno. Moramo uzeti u obzir i druge značajke!

> Možete vidjeti da su minimalne cijene bundeva opažene negdje oko Noći vještica. Kako to možete objasniti?

🎃 Čestitamo, upravo ste stvorili model koji može pomoći u predviđanju cijene pita od bundeva. Vjerojatno možete ponoviti isti postupak za sve vrste bundeva, ali to bi bilo zamorno. Naučimo sada kako u model uključiti vrstu bundeve!

## Kategorijske značajke

U idealnom svijetu želimo moći predvidjeti cijene za različite sorte bundeva koristeći isti model. Međutim, stupac `Variety` se razlikuje od stupaca poput `Month`, jer sadrži nenumeričke vrijednosti. Takvi se stupci nazivaju **kategorijskim**.

[![ML za početnike - Predviđanja kategorijskih značajki linearnom regresijom](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML za početnike - Predviđanja kategorijskih značajki linearnom regresijom")

> 🎥 Kliknite gornju sliku za kratak video pregled korištenja kategorijskih značajki.

Ovdje možete vidjeti kako prosječna cijena ovisi o sorti:

<img alt="Average price by variety" src="../../../../translated_images/hr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Da bismo uzeli sortu u obzir, prvo je moramo pretvoriti u numerički oblik, ili je **enkodirati**. Postoji nekoliko načina kako to možemo učiniti:

* Jednostavna **numerička enkodiranja** će napraviti tablicu različitih sorti, a zatim zamijeniti ime sorte indeksom u toj tablici. To nije najbolja ideja za linearnu regresiju, jer linearna regresija uzima stvarnu numeričku vrijednost indeksa i dodaje je rezultatu množeći s nekim koeficijentom. U našem slučaju, veza između broja indeksa i cijene je jasno nelinearna, čak i ako osiguramo da su indeksi uređeni na neki specifičan način.
* **One-hot enkodiranje** zamijenit će stupac `Variety` s 4 različita stupca, po jedan za svaku sortu. Svaki stupac će sadržavati `1` ako je odgovarajući redak određene sorte, a `0` inače. To znači da će postojati četiri koeficijenta u linearnoj regresiji, po jedan za svaku sortu bundeve, koji će biti odgovorni za "početnu cijenu" (ili bolje rečeno "dodatnu cijenu") za tu određenu sortu.

Donji kôd pokazuje kako možemo one-hot enkodirati sortu:

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

Da bismo trenirali linearnu regresiju koristeći one-hot enkodiranu sortu kao ulaz, samo trebamo pravilno inicijalizirati podatke `X` i `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Ostatak koda je isti kao što smo gore koristili za treniranje Linearne Regresije. Ako to probate, vidjet ćete da je srednja kvadratna pogreška približno ista, ali dobivamo puno veći koeficijent determinacije (~77%). Da bismo dobili još točnija predviđanja, možemo uzeti u obzir još kategorijskih značajki, kao i numeričkih značajki, poput `Month` ili `DayOfYear`. Da bismo dobili jedan veliki niz značajki, možemo koristiti `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Ovdje također uzimamo u obzir `City` i tip `Package`, što nam daje MSE 2.84 (10%), i determinaciju 0.94!

## Sve spojeno

Da bismo napravili najbolji model, možemo koristiti kombinirane (one-hot enkodirane kategorijske + numeričke) podatke iz prethodnog primjera zajedno s polinomskom regresijom. Evo kompletnog koda radi vaše udobnosti:

```python
# postavi podatke za trening
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# napravi podjelu podataka na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# postavi i treniraj pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predvidi rezultate za testne podatke
pred = pipeline.predict(X_test)

# izračunaj MSE i koeficijent determinacije
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

To bi nam trebalo dati najbolji koeficijent determinacije od gotovo 97%, i MSE=2.23 (~8% pogreška predviđanja).

| Model | MSE | Determinacija |
|-------|-----|--------------|
| Linearni `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomski `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Linearna `Variety` | 5.24 (19.7%) | 0.77 |
| Sve značajke Linearno | 2.84 (10.5%) | 0.94 |
| Sve značajke Polinomski | 2.23 (8.25%) | 0.97 |

🏆 Svaka čast! Stvorili ste četiri modela regresije u jednoj lekciji i poboljšali kvalitetu modela na 97%. U završnom dijelu o regresiji naučit ćete o logističkoj regresiji za određivanje kategorija.

---
## 🚀Izazov

Testirajte nekoliko različitih varijabli u ovom bilježniku da vidite kako korelacija odgovara točnosti modela.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalan rad

U ovoj lekciji smo naučili o linearnoj regresiji. Postoje i druge važne vrste regresije. Pročitajte o tehnikama Stepwise, Ridge, Lasso i Elasticnet. Dobar tečaj za daljnje proučavanje je [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Zadatak

[Izradi model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Odricanje od odgovornosti**:
Ovaj je dokument preveden pomoću AI usluge prevođenja [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za kritične informacije preporuča se profesionalni prijevod ljudskog prevoditelja. Ne snosimo odgovornost za bilo kakva nesporazumevanja ili pogrešne interpretacije nastale korištenjem ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->