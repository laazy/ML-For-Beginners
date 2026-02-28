# Vytvoření regresního modelu pomocí Scikit-learn: regrese čtyřmi způsoby

## Poznámka pro začátečníky

Lineární regrese se používá, když chceme předpovědět **číselnou hodnotu** (například cenu domu, teplotu nebo prodeje). Funguje tak, že najde přímku, která nejlépe reprezentuje vztah mezi vstupními rysy a výstupem.

V této lekci se zaměříme na pochopení konceptu před tím, než prozkoumáme pokročilejší regresní techniky.
![Infografika lineární vs. polynomiální regrese](../../../../translated_images/cs/linear-polynomial.5523c7cb6576ccab.webp)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Přednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tato lekce je dostupná také v R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Úvod

Dosud jste prozkoumali, co je regrese, na ukázkových datech ze sady dat o cenách dýní, kterou budeme používat v celé této lekci. Také jste si ji vizualizovali pomocí Matplotlib.

Nyní jste připraveni ponořit se hlouběji do regrese pro strojové učení. Zatímco vizualizace vám umožní porozumět datům, skutečná síla strojového učení spočívá v _trénování modelů_. Modely jsou trénovány na historických datech, aby automaticky zachytily závislosti v datech, a umožňují vám předpovídat výsledky pro nová data, která model dosud neviděl.

V této lekci se naučíte více o dvou typech regrese: _základní lineární regresi_ a _polynomiální regresi_, společně s některou z matematiky, která stojí za těmito technikami. Tyto modely nám umožní předpovídat ceny dýní v závislosti na různých vstupních datech. 

[![Strojové učení pro začátečníky – Pochopení lineární regrese](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Strojové učení pro začátečníky – Pochopení lineární regrese")

> 🎥 Klikněte na obrázek výše pro krátké video přehled lineární regrese.

> V celém tomto kurzu předpokládáme minimální znalost matematiky a snažíme se ji zpřístupnit studentům z jiných oborů, proto sledujte poznámky, 🧮 upozornění, diagramy a další učební pomůcky pro lepší porozumění.

### Předpoklady

Měli byste už být obeznámeni se strukturou dat o dýních, která zkoumáme. Najdete je přednačtená a předvyčištěná v souboru _notebook.ipynb_ této lekce. V souboru je cena dýní zobrazena za bushel v novém datovém rámci. Ujistěte se, že můžete spouštět tyto notebooky v kernelu ve Visual Studio Code.

### Příprava

Pro připomenutí, tato data načítáte proto, abyste si na ně mohli klást otázky.

- Kdy je nejlepší čas koupit dýně? 
- Jakou cenu mohu očekávat za balení mini dýní?
- Měl bych je kupovat v polovině bushelového koše, nebo v 1 1/9 bushelové krabici?
Pojďme v tomto zkoumání dat pokračovat.

V předchozí lekci jste vytvořili Pandas datový rámec a naplnili jej částí původních dat, přičemž jste ceny standardizovali za bushel. Tím jste však získali pouze asi 400 datových bodů a jen pro podzimní měsíce.

Podívejte se na data, která jsme přednačetli v přidruženém notebooku této lekce. Data jsou přednačtená a je vykreslen prvotní scatterplot zobrazující data podle měsíců. Možná můžeme získat podrobnější informace o povaze dat jejich dalším čištěním.

## Lineární regresní přímka

Jak jste se naučili v Lekci 1, cílem lineární regrese je být schopen vykreslit přímku, která:

- **Ukáže vztahy proměnných**. Ukáže vztah mezi proměnnými
- **Umožní předpovědi**. Umožní přesně předpovědět, kde by se nový datový bod mohl nacházet vzhledem k této přímce.
 
Je typické pro **metodu nejmenších čtverců**, že se takováto přímka kreslí. Termín "nejmenší čtverce" odkazuje na proces minimalizace celkové chyby v našem modelu. Pro každý datový bod měříme vertikální vzdálenost (nazývanou reziduál) mezi skutečným bodem a naší regresní přímkou.  

Tyto vzdálenosti umocňujeme na druhou ze dvou hlavních důvodů:  

1. **Velikost nad směrem:** Chceme, aby chyba -5 byla stejně vážná jako chyba +5. Umocnění na druhou zaručí, že všechny hodnoty jsou kladné.  

2. **Trestání odlehlých hodnot:** Umocnění na druhou dává větší váhu větším chybám, což nutí přímku být blíže k bodům, které jsou daleko.  

Poté všechny tyto umocněné hodnoty sečteme. Naším cílem je najít specifickou přímku, kde je tenhle součet nejmenší (nejmenší možná hodnota) - odtud název "nejmenší čtverce".  

> **🧮 Ukázat matematiku** 
> 
> Tato přímka, nazývaná _přímka nejlepšího přizpůsobení_, může být vyjádřena pomocí [rovnice](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'vysvětlující proměnná'. `Y` je 'závislá proměnná'. Směrnice přímky je `b` a `a` je průsečík s osou y, což je hodnota `Y`, když `X = 0`. 
>
>![výpočet směrnice](../../../../translated_images/cs/slope.f3c9d5910ddbfcf9.webp)
>
> Nejprve vypočítejte směrnici `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Jinými slovy, a s odkazem na naši původní otázku ohledně dat o dýních: "předpověď ceny dýně za bushel podle měsíce", `X` by odkazovalo na měsíc a `Y` by odkazovalo na cenu.
>
>![dokončení rovnice](../../../../translated_images/cs/calculation.a209813050a1ddb1.webp)
>
> Vypočítejte hodnotu Y. Pokud platíte kolem 4 dolarů, musí to být duben! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika počítající přímku musí demonstrovat směrnici přímky, která také závisí na průsečíku, tedy kde je `Y` situováno, když `X = 0`.
>
> Metodu výpočtu těchto hodnot můžete vidět na webu [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Navštivte také [tuto kalkulačku metody nejmenších čtverců](https://www.mathsisfun.com/data/least-squares-calculator.html) a sledujte, jak hodnoty čísel ovlivňují přímku.

## Korelace

Ještě jeden termín k pochopení je **koeficient korelace** mezi danými proměnnými X a Y. Pomocí scatterplotu můžete tento koeficient rychle vizualizovat. Graf s body rozprostřenými po úhledné přímce má vysokou korelaci, ale graf s body rozptýlenými všude po ose X a Y má korelaci nízkou.

Dobrý model lineární regrese bude ten, který má vysoký (blíže k 1 než k 0) koeficient korelace používající metodu nejmenších čtverců s regresní přímkou.

✅ Spusťte si notebook přidružený k této lekci a podívejte se na scatterplot spojení Měsíc vs. Cena. Zdá se podle vašeho vizuálního hodnocení scatterplotu, že data spojující měsíc s cenou pro prodej dýní mají vysokou nebo nízkou korelaci? Změní se to, když použijete jemnější měřítko namísto `Month`, např. *den v roce* (tedy počet dní od začátku roku)?

V níže uvedeném kódu předpokládáme, že jsme data vyčistili a získali datový rámec nazvaný `new_pumpkins`, podobný následujícímu:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kód pro vyčištění dat je dostupný v [`notebook.ipynb`](notebook.ipynb). Provedli jsme stejné kroky čištění jako v předchozí lekci a dopočítali sloupec `DayOfYear` pomocí následujícího výrazu: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nyní, když máte pochopení matematiky stojící za lineární regresí, vytvoříme regresní model, abychom zjistili, zda můžeme předpovědět, které balení dýní bude mít nejlepší ceny. Někdo, kdo kupuje dýně pro sváteční dýňovou zahradu, by mohl chtít tyto informace, aby mohl optimalizovat nákupy balení dýní pro svou zahradu.

## Hledání korelace

[![Strojové učení pro začátečníky – Hledání korelace: Klíč k lineární regresi](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Strojové učení pro začátečníky – Hledání korelace: Klíč k lineární regresi")

> 🎥 Klikněte na obrázek výše pro krátké video přehled korelace.

Z předchozí lekce jste pravděpodobně viděli, že průměrná cena za různé měsíce vypadá takto:

<img alt="Průměrná cena podle měsíce" src="../../../../translated_images/cs/barchart.a833ea9194346d76.webp" width="50%"/>

To naznačuje, že by měla být nějaká korelace, a můžeme vyzkoušet natrénovat lineární regresní model k předpovědi vztahu mezi `Month` a `Price`, nebo mezi `DayOfYear` a `Price`. Zde je scatter plot, který ukazuje druhý zmíněný vztah:

<img alt="Scatter plot Cena vs. Den v roce" src="../../../../translated_images/cs/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Podívejme se, zda existuje korelace pomocí funkce `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Zdá se, že korelace je poměrně malá, -0.15 podle `Month` a -0.17 podle `DayOfMonth`, ale mohlo by existovat jiné důležité spojení. Zdá se, že existují různé shluky cen odpovídající různým odrůdám dýní. Abychom tuto hypotézu potvrdili, vykreslíme každou kategorii dýní s jinou barvou. Předáním parametru `ax` do funkce `scatter` můžeme vykreslit všechny body na stejném grafu:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot Cena vs. Den v roce s barvami" src="../../../../translated_images/cs/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Naše vyšetřování naznačuje, že odrůda má větší vliv na celkovou cenu než skutečné datum prodeje. Vidíme to i na sloupcovém grafu:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Sloupcový graf cen podle odrůdy" src="../../../../translated_images/cs/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Zaměříme se prozatím pouze na jednu odrůdu dýní, 'pie type', a podíváme se, jaký vliv má datum na cenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot Cena vs. Den v roce pro dýně pie type" src="../../../../translated_images/cs/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Pokud nyní vypočítáme korelaci mezi `Price` a `DayOfYear` pomocí funkce `corr`, dostaneme asi `-0.27` – což znamená, že trénování prediktivního modelu dává smysl.

> Před trénováním modelu lineární regrese je důležité mít jistotu, že jsou data vyčištěná. Lineární regrese nefunguje dobře s chybějícími hodnotami, proto je vhodné se zbavit všech prázdných buněk:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Dalším přístupem by bylo nahradit tyto chybějící hodnoty průměrnými hodnotami příslušného sloupce.

## Jednoduchá lineární regrese

[![Strojové učení pro začátečníky – Lineární a polynomiální regrese pomocí Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Strojové učení pro začátečníky – Lineární a polynomiální regrese pomocí Scikit-learn")

> 🎥 Klikněte na obrázek výše pro krátké video přehled lineární a polynomiální regrese.

Pro natrénování našeho modelu lineární regrese použijeme knihovnu **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Začneme oddělením vstupních hodnot (rysy) a očekávaných výstupů (štítky) do samostatných numpy polí:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Všimněte si, že jsme museli provést změnu tvaru (`reshape`) vstupních dat, aby je balíček lineární regrese správně pochopil. Lineární regrese očekává 2D pole jako vstup, kde každý řádek pole odpovídá vektoru vstupních rysů. V našem případě, protože máme pouze jeden vstup - potřebujeme pole o tvaru N&times;1, kde N je velikost datové sady.

Následně je třeba data rozdělit na trénovací a testovací sady, abychom mohli model po tréninku ověřit:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Nakonec samotné trénování lineárního regresního modelu trvá jen dva řádky kódu. Definujeme objekt `LinearRegression` a přizpůsobíme ho našim datům pomocí metody `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` po provedení `fit` obsahuje všechny koeficienty regrese, ke kterým lze přistupovat pomocí vlastnosti `.coef_`. V našem případě je jen jeden koeficient, který by měl být přibližně `-0.017`. To znamená, že ceny se zdají s časem mírně snižovat, ale ne příliš, kolem 2 centů za den. K průsečíku regrese s osou Y se lze také dostat pomocí `lin_reg.intercept_` - v našem případě bude přibližně `21`, což ukazuje cenu na začátku roku.

Abychom mohli zjistit, jak přesný náš model je, můžeme předpovídat ceny na testovací datové sadě a pak změřit, jak blízko jsou naše předpovědi ke skutečným hodnotám. To lze provést pomocí metriky střední kvadratické chyby (MSE), což je průměr všech druhých mocnin rozdílů mezi očekávanou a předpovězenou hodnotou.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Naše chyba se zdá být kolem 2 bodů, což je asi 17 %. Není to moc dobré. Dalším ukazatelem kvality modelu je **koeficient determinace**, který lze získat takto:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Pokud je hodnota 0, znamená to, že model nevnímá vstupní data a chová se jako *nejhorší lineární prediktor*, což je jednoduše průměr výsledku. Hodnota 1 znamená, že můžeme dokonale předpovědět všechny očekávané výstupy. V našem případě je koeficient kolem 0,06, což je poměrně nízké.

Můžeme také vykreslit testovací data spolu s regresní přímkou, abychom lépe viděli, jak regrese v našem případě funguje:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/cs/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomická regrese

Dalším typem lineární regrese je polynomická regrese. Zatímco někdy existuje mezi veličinami lineární vztah – čím větší dýně objemem, tím vyšší cena – jindy tyto vztahy nelze zobrazit jako rovinu nebo přímku. 

✅ Zde je [několik dalších příkladů](https://online.stat.psu.edu/stat501/lesson/9/9.8) dat, která by mohla využít polynomickou regresi

Podívejte se znovu na vztah mezi Datem a Cenou. Zdá se, že by tento bodový graf nutně měl být analyzován pomocí přímky? Nemohou ceny kolísat? V takovém případě můžete vyzkoušet polynomickou regresi.

✅ Polynom jsou matematické výrazy, které mohou obsahovat jednu nebo více proměnných a koeficientů

Polynomická regrese vytváří zakřivenou křivku, aby lépe seděla na nelineární data. V našem případě, pokud do vstupních dat zahrneme druhou mocninu proměnné `DayOfYear`, měli bychom být schopni přizpůsobit naše data parabole, která bude mít minimum v určitém bodě během roku.

Scikit-learn obsahuje užitečné [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) pro spojování různých kroků zpracování dat dohromady. **Pipeline** je řetězec **estimatorů**. V našem případě vytvoříme pipeline, která nejprve přidá polynomické rysy k našemu modelu a pak natrénuje regresi:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Použití `PolynomialFeatures(2)` znamená, že zahrneme všechny polynomy druhého stupně ze vstupních dat. V našem případě to bude jen `DayOfYear`<sup>2</sup>, ale pokud máme dvě vstupní proměnné X a Y, přidá to X<sup>2</sup>, XY a Y<sup>2</sup>. Můžeme také použít vyšší mocniny, pokud chceme.

Pipeline lze používat stejným způsobem jako původní objekt `LinearRegression`, tj. můžeme pipeline `fit`-nout a pak použít `predict` pro získání predikcí. Zde je graf ukazující testovací data a aproximační křivku:

<img alt="Polynomial regression" src="../../../../translated_images/cs/poly-results.ee587348f0f1f60b.webp" width="50%" />

S využitím polynomické regrese můžeme získat mírně nižší MSE a vyšší koeficient determinace, ale ne dramaticky. Musíme vzít v úvahu i další rysy!

> Vidíte, že minimální ceny dýní jsou pozorovány někde kolem Halloweenu. Jak byste tento jev vysvětlili? 

🎃 Gratulujeme, právě jste vytvořili model, který může pomoci předpovědět cenu dýní na koláče. Pravděpodobně byste stejný postup mohli opakovat pro všechny druhy dýní, ale to by bylo zdlouhavé. Naučme se nyní, jak brát v úvahu odrůdu dýně v našem modelu!

## Kategorické rysy

V ideálním světě chceme být schopni předpovídat ceny pro různé odrůdy dýní pomocí stejného modelu. Sloupec `Variety` je však trochu odlišný od sloupců jako `Month`, protože obsahuje nečíselné hodnoty. Takové sloupce se nazývají **kategorické**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Klikněte na obrázek výše pro krátké video o použití kategorických rysů.

Zde vidíte, jak průměrná cena závisí na odrůdě:

<img alt="Average price by variety" src="../../../../translated_images/cs/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Abychom mohli zohlednit odrůdu, musíme ji nejdříve převést do číselné podoby, tedy **zakódovat** ji. Existuje několik způsobů, jak to udělat:

* Jednoduché **číselné kódování** vytvoří tabulku různých odrůd a pak nahradí název odrůdy indexem v této tabulce. To však není nejlepší nápad pro lineární regresi, protože lineární regrese vezme skutečnou číselnou hodnotu indexu a přidá ji k výsledku, vynásobí nějakým koeficientem. V našem případě je vztah mezi číslem indexu a cenou jasně nelineární, i když zajistíme, že indexy jsou uspořádány nějak specificky.
* **One-hot encoding** nahradí sloupec `Variety` čtyřmi různými sloupci, jedním pro každou odrůdu. Každý sloupec bude obsahovat `1`, pokud příslušný řádek odpovídá dané odrůdě, a `0` jinak. To znamená, že v lineární regresi budou čtyři koeficienty, po jednom pro každou odrůdu dýní, zodpovědné za "startovací cenu" (nebo spíše "přídavnou cenu") pro danou odrůdu.

Níže uvedený kód ukazuje, jak lze odrůdu zakódovat one-hot metodou:

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

Pro trénování lineární regrese s one-hot zakódovanou odrůdou jako vstupem stačí správně inicializovat `X` a `y` data:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Zbytek kódu je stejný jako ten, který jsme použili výše pro trénování lineární regrese. Pokud to vyzkoušíte, uvidíte, že střední kvadratická chyba je přibližně stejná, ale koeficient determinace je podstatně vyšší (~77 %). Pro ještě přesnější předpovědi můžeme vzít v úvahu více kategorických rysů i číselné rysy, jako jsou `Month` nebo `DayOfYear`. Pro získání jedné velké množiny rysů můžeme použít `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Zde také bereme v úvahu `City` a typ `Package`, což nám dává MSE 2,84 (10 %) a koeficient determinace 0,94!

## Spojení všeho dohromady

Pro vytvoření nejlepšího modelu můžeme použít kombinovaná (one-hot zakódovaná kategorická + číselná) data z výše uvedeného příkladu společně s polynomickou regresí. Pro vaše pohodlí je zde úplný kód:

```python
# připravit tréninková data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# provést rozdělení na tréninkovou a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# nastavit a natrénovat pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# předpovědět výsledky pro testovací data
pred = pipeline.predict(X_test)

# vypočítat MSE a koeficient determinace
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

To by mělo dát nejlepší koeficient determinace téměř 97 % a MSE=2,23 (~8 % chybovost předpovědi).

| Model | MSE | Determinace |
|-------|-----|---------------|
| Lineární s `DayOfYear` | 2,77 (17,2 %) | 0,07 |
| Polynomická s `DayOfYear` | 2,73 (17,0 %) | 0,08 |
| Lineární s `Variety` | 5,24 (19,7 %) | 0,77 |
| Lineární se všemi rysy | 2,84 (10,5 %) | 0,94 |
| Polynomická se všemi rysy | 2,23 (8,25 %) | 0,97 |

🏆 Výborně! Vytvořili jste čtyři regresní modely v jedné lekci a vylepšili kvalitu modelu na 97 %. V poslední části o regresi se naučíte o logistické regresi pro určení kategorií. 

---
## 🚀Výzva

Otestujte v tomto notebooku několik různých proměnných a sledujte, jak korelace odpovídá přesnosti modelu.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Recenze a samostudium

V této lekci jsme se naučili o lineární regresi. Existují i jiné důležité typy regrese. Přečtěte si o postupech Stepwise, Ridge, Lasso a Elasticnet. Dobrým kurzem, který se můžete dále učit, je [Stanfordský kurz statistického učení](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Zadání

[Vybuduj model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zřeknutí se odpovědnosti**:  
Tento dokument byl přeložen pomocí automatické překladové služby [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože usilujeme o přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho mateřském jazyce by měl být považován za autoritativní zdroj. Pro kritické informace doporučujeme profesionální lidský překlad. Nejsme odpovědní za jakékoliv nedorozumění nebo nesprávné interpretace vzniklé použitím tohoto překladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->