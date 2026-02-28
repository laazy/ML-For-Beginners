# Gumawa ng modelo ng regresyon gamit ang Scikit-learn: regression sa apat na paraan

## Tala para sa Baguhan

Ginagamit ang linear regression kapag gusto nating hulaan ang isang **numerikal na halaga** (halimbawa, presyo ng bahay, temperatura, o benta).
Ito ay gumagana sa paghahanap ng tuwid na linya na pinakamahusay na kumakatawan sa relasyon sa pagitan ng mga input feature at output.

Sa araling ito, nakatuon tayo sa pag-unawa ng konsepto bago tuklasin ang mas advanced na mga teknik sa regresyon.
![Linear vs polynomial regression infographic](../../../../translated_images/tl/linear-polynomial.5523c7cb6576ccab.webp)
> Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Available ang araling ito sa R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Panimula

Sa ngayon ay napag-aralan mo na kung ano ang regresyon gamit ang sample na datos mula sa pumpkin pricing dataset na gagamitin natin sa buong araling ito. Na-visualize mo rin ito gamit ang Matplotlib.

Ngayon ay handa ka nang sumisid nang mas malalim sa regresyon para sa ML. Habang ang visualization ay nagpapadali upang maintindihan ang data, ang tunay na lakas ng Machine Learning ay nagmumula sa _pagsasanay ng mga modelo_. Ang mga modelo ay sinasanay gamit ang makasaysayang datos upang awtomatikong makuha ang mga ugnayan ng datos, at nagbibigay-daan ito upang mahulaan ang mga kinalabasan para sa bagong data na hindi pa nakita ng modelo.

Sa araling ito, matututuhan mo pa ang tungkol sa dalawang uri ng regresyon: _basic linear regression_ at _polynomial regression_, kasama ang ilan sa mga matematikal na batayan ng mga teknik na ito. Makakatulong ang mga modelong ito upang mahulaan ang presyo ng kalabasa batay sa iba't ibang input datos.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 I-click ang larawan sa itaas para sa maikling video na pagpapakilala sa linear regression.

> Sa buong kurikulum na ito, ipinagpapalagay namin ang pinaka-kaunting kaalaman sa matematika, at layuning gawing madaling maunawaan para sa mga estudyanteng nagmula sa iba't ibang larangan, kaya bantayan ang mga tala, 🧮 mga paalala, mga diagram, at iba pang mga gamit sa pagkatuto upang makatulong sa pag-unawa.

### Kinakailangang Kaalaman

Dapat ay pamilyar ka na sa istraktura ng pumpkin data na sinusuri natin. Makikita mo ito na naka-preload at nalinis na sa _notebook.ipynb_ file ng araling ito. Sa file, ipinapakita ang presyo ng kalabasa kada bushel sa isang bagong data frame. Siguraduhing kaya mong patakbuhin ang mga notebook na ito gamit ang kernels sa Visual Studio Code.

### Paghahanda

Bilang paalala, niloload mo ang datos upang makapagtanong tungkol dito. 

- Kailan ang pinakamainam na oras upang bumili ng mga kalabasa? 
- Ano ang presyo na maaasahan ko sa isang kahon ng mga miniature pumpkins?
- Dapat ba akong bumili ng mga ito sa pamamagitan ng half-bushel baskets o sa 1 1/9 bushel box?

Tuloy natin ang pagsisiyasat sa datos na ito.

Sa naunang aralin, gumawa ka ng Pandas data frame at pinunan ito ng bahagi ng orihinal na dataset, pinagpantay-pantay ang presyo ayon sa bushel. Sa paggawa nito, nakalap mo lamang ang humigit-kumulang 400 na datapoints at para lamang sa mga buwan ng taglagas.

Tingnan ang datos na naka-preload sa notebook na kasama ng araling ito. Naipakita na ang scatterplot upang ipakita ang data ng buwan. Maaaring makakuha tayo ng mas detalyadong kaalaman tungkol sa likas ng datos sa pamamagitan ng mas malalim na paglilinis nito.

## Isang linya ng linear regression

Tulad ng natutunan mo sa Aralin 1, ang layunin ng linear regression exercise ay makapag-plot ng linya upang:

- **Ipakita ang relasyon ng mga variable**. Ipakita ang relasyon sa pagitan ng mga variable
- **Gumawa ng mga hulang prediksyon**. Gumawa ng tumpak na prediksyon kung saan mahuhulog ang bagong datapoint kaugnay ng linyang iyon.

Karaniwan sa **Least-Squares Regression** ang pagguhit ng ganitong uri ng linya. Ang terminong "Least-Squares" ay tumutukoy sa proseso ng pagbawas sa kabuuang error sa modelo natin. Para sa bawat data point, sinusukat natin ang patayong distansya (tinatawag na residual) sa pagitan ng aktwal na punto at ng ating regression line.

Ikinakuwadrado natin ang mga distansyang ito sa dalawang pangunahing dahilan:

1. **Laki kaysa Direksyon:** Gusto nating tratuhin ang error na -5 gaya ng error na +5. Ginagawa nitong positibo ang lahat ng halaga sa pagkuwadrado.

2. **Pagbigay ng Parusa sa Malalaking Error:** Nagbibigay ang pagkuwadrado ng mas malaking bigat sa mas malalaking error, na pinipilit ang linya na maging malapit sa mga puntos na malayo.

Pagkatapos, pinagsasama-sama natin ang lahat ng mga na-kuwadradong halaga. Layunin natin mahanap ang tiyak na linya kung saan ang kabuuang halaga nito ay pinakamababa (ang pinakamaliit na posibleng halaga)—kaya tinawag itong "Least-Squares."

> **🧮 Ipakita sa akin ang math**  
> 
> Ang linyang ito, na tinatawag na _line of best fit_ ay maaaring ipahayag sa pamamagitan ng [isang equation](https://en.wikipedia.org/wiki/Simple_linear_regression):  
>
> ```
> Y = a + bX
> ```
>
> Ang `X` ay ang 'explanatory variable'. Ang `Y` ay ang 'dependent variable'. Ang slope ng linya ay `b` at ang `a` ay ang y-intercept, na tumutukoy sa halaga ng `Y` kapag `X = 0`.
>
>![calculate the slope](../../../../translated_images/tl/slope.f3c9d5910ddbfcf9.webp)
>
> Una, kalkulahin ang slope `b`. Infographic ni [Jen Looper](https://twitter.com/jenlooper)
>
> Sa madaling salita, at tumutukoy sa orihinal na tanong ng ating pumpkin data: "hulaan ang presyo ng kalabasa kada bushel bawat buwan", ang `X` ay tumutukoy sa presyo at ang `Y` ay tumutukoy sa buwan ng bentahan.
>
>![complete the equation](../../../../translated_images/tl/calculation.a209813050a1ddb1.webp)
>
> Kalkulahin ang halaga ng Y. Kung nagbabayad ka ng humigit-kumulang $4, siguradong Abril ito! Infographic ni [Jen Looper](https://twitter.com/jenlooper)
>
> Ang matematika na nagkukuwenta ng linyang ito ay dapat ipakita ang slope ng linya, na nakaasa rin sa intercept, o kung saan nakalugar ang `Y` kapag `X = 0`.
>
> Maaari mong obserbahan ang pamamaraan ng pagkalkula ng mga halagang ito sa web site na [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Bisitahin din ang [Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) upang makita kung paano naapektuhan ng mga halagang numero ang linya.

## Korelasyon

Isang terminong kailangang maunawaan ay ang **Correlation Coefficient** sa pagitan ng mga ibinigay na X at Y na variable. Gamit ang scatterplot, mabilis mong maipapakita ang coefficient na ito. Ang plot na may mga datapoints na magkakatabi sa isang maayos na linya ay may mataas na korelasyon, ngunit ang plot na may mga datapoints na kalat-kalat sa pagitan ng X at Y ay may mababang korelasyon.

Ang magandang modelo ng linear regression ay yaong may mataas na (malapit sa 1 kaysa sa 0) Correlation Coefficient gamit ang Least-Squares Regression na may linya ng regresyon.

✅ Patakbuhin ang notebook na kalakip ng araling ito at tingnan ang Month to Price scatterplot. Mukhang mataas o mababa ang korelasyon ng datos sa pagitan ng Buwan at Presyo ng bentahan ng kalabasa, ayon sa iyong visual na interpretasyon ng scatterplot? Nagbabago ba ito kung gagamit ka ng mas maselang sukatan sa halip na `Month`, halimbawa, *araw ng taon* (ibig sabihin, bilang ng mga araw mula sa simula ng taon)?

Sa code sa ibaba, ipagpapalagay natin na nalinis na natin ang data, at nakakuha ng data frame na tinatawag na `new_pumpkins`, tulad ng sumusunod:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Ang code para linisin ang data ay matatagpuan sa [`notebook.ipynb`](notebook.ipynb). Ginawa natin ang parehong hakbang sa paglilinis tulad ng sa naunang aralin, at nakwenta ang column na `DayOfYear` gamit ang sumusunod na expression:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Ngayon na nauunawaan mo na ang matematika sa likod ng linear regression, gawin natin ang isang Regression model upang makita kung kaya nating mahulaan kung aling pakete ng kalabasa ang magkakaroon ng pinakamagandang presyo. Ang isang bumibili ng kalabasa para sa holiday pumpkin patch ay maaaring magustuhan ang impormasyong ito upang mapahusay ang kanilang pagbili ng mga pakete ng kalabasa para sa patch.

## Paghahanap ng Korelasyon

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 I-click ang larawan sa itaas para sa maikling video na pagpapakilala sa korelasyon.

Mula sa naunang aralin, marahil ay nakita mo na ang average na presyo para sa iba't ibang buwan ay ganito:

<img alt="Average price by month" src="../../../../translated_images/tl/barchart.a833ea9194346d76.webp" width="50%"/>

Ipinapahiwatig nito na dapat mayroong ilang korelasyon, at maaari nating subukang sanayin ang linear regression model upang hulaan ang ugnayan ng `Month` at `Price`, o ng `DayOfYear` at `Price`. Narito ang scatter plot na nagpapakita ng huling ugnayan:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/tl/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Tingnan natin kung may korelasyon gamit ang `corr` na function:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Mukhang maliit ang korelasyon, -0.15 para sa `Month` at -0.17 para sa `DayOfMonth`, ngunit maaaring may isa pang mahalagang ugnayan. Mukhang may magkakaibang pangkat ng presyo na tumutugma sa iba't ibang uri ng kalabasa. Upang kumpirmahin ito, ilarawan natin bawat kategorya ng kalabasa gamit ang ibang kulay. Sa pamamagitan ng pagpasa ng `ax` parameter sa `scatter` plotting function, maaari nating ipakita lahat ng puntos sa iisang grapiko:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/tl/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Ang imbestigasyon natin ay nagpapahiwatig na ang uri ay mas may epekto sa pangkalahatang presyo kaysa sa aktwal na petsa ng bentahan. Makikita ito sa isang bar graph:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/tl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Magtuon muna tayo sa isang uri ng kalabasa, ang 'pie type', at tingnan ang epekto ng petsa sa presyo:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/tl/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Kung ngayon ay kakalkulahin natin ang korelasyon sa pagitan ng `Price` at `DayOfYear` gamit ang `corr` function, makakakuha tayo ng humigit-kumulang `-0.27` - na nangangahulugang makatuwiran ang pagsasanay ng prediktibong modelo.

> Bago mag-train ng linear regression model, mahalaga na malinis ang datos. Hindi maganda ang linear regression sa mga kulang na halaga, kaya makatuwiran na alisin ang lahat ng walang laman na cells:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Isa pang pamamaraan ay punan ang mga walang laman na value gamit ang average value mula sa katumbas na column.

## Simple Linear Regression

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 I-click ang larawan sa itaas para sa maikling video na pagpapakilala sa linear at polynomial regression.

Para sanayin ang ating Linear Regression model, gagamitin natin ang **Scikit-learn** library.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Nagsisimula tayo sa paghihiwalay ng mga input na values (features) at ang inaasahang output (label) sa magkahiwalay na numpy arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Tandaan na kinailangang gawin ang `reshape` sa input data upang maunawaan ito nang tama ng Linear Regression package. Inaasahan ng Linear Regression ang 2D-array bilang input, kung saan ang bawat hilera ng array ay tumutugma sa isang vector ng input features. Sa ating kaso, dahil isa lang ang input — kailangan natin ng array na hugis N&times;1, kung saan ang N ay ang laki ng dataset.

Pagkatapos, kailangang hatiin ang data sa train at test datasets, upang ma-validate natin ang ating modelo pagkatapos i-train:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Sa huli, ang pagsasanay ng tunay na Linear Regression model ay nangangailangan lamang ng dalawang linya ng code. Idefine natin ang `LinearRegression` na object, at i-fit ito sa ating data gamit ang `fit` method:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Ang `LinearRegression` na object pagkatapos ng `fit`-ting ay naglalaman ng lahat ng coefficients ng regression, na maaaring ma-access gamit ang `.coef_` property. Sa aming kaso, may isa lamang coefficient, na dapat ay nasa paligid ng `-0.017`. Ibig sabihin nito na tila bumababa ang mga presyo nang bahagya sa paglipas ng panahon, ngunit hindi masyadong malaki, mga 2 sentimo kada araw. Maaari rin nating ma-access ang punto ng intersection ng regression sa Y-axis gamit ang `lin_reg.intercept_` - ito ay nasa paligid ng `21` sa aming kaso, na nagsasaad ng presyo sa simula ng taon.

Para makita kung gaano katumpak ang aming modelo, maaari nating hulaan ang mga presyo sa test dataset, at sukatin kung gaano kalapit ang aming mga hula sa inaasahang mga halaga. Magagawa ito gamit ang mean square error (MSE) na metrics, na siyang mean ng lahat ng squared differences sa pagitan ng inaasahan at hinulaan na halaga.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Ang aming error ay tila nasa paligid ng 2 puntos, na ~17%. Hindi masyadong maganda. Isa pang pananda ng kalidad ng modelo ay ang **coefficient of determination**, na maaaring makuha sa ganitong paraan:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Kung ang halaga ay 0, ibig sabihin ay hindi tinatanggap ng modelo ang input data, at kumikilos bilang *pinakamasamang linear predictor*, na isang mean value ng resulta. Ang halaga na 1 ay nangangahulugan na perpektong mahuhulaan natin lahat ng inaasahang output. Sa aming kaso, ang coefficient ay nasa paligid ng 0.06, na medyo mababa.

Maaari rin nating i-plot ang test data kasama ang regression line para mas makita kung paano gumagana ang regression sa aming kaso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/tl/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomial Regression

Isa pang uri ng Linear Regression ay ang Polynomial Regression. Habang minsan ay may linear na relasyon sa pagitan ng mga variable - mas malaki ang volume ng kalabasa, mas mataas ang presyo - may mga pagkakataon din na hindi maaaring i-plot ang mga relasyong ito bilang isang eroplano o tuwid na linya.

✅ Narito ang [ilang karagdagang halimbawa](https://online.stat.psu.edu/stat501/lesson/9/9.8) ng data na maaaring gumamit ng Polynomial Regression

Balikan ang relasyon sa pagitan ng Date at Price. Tila ba ang scatterplot na ito ay kailangan talagang suriin gamit ang tuwid na linya? Hindi ba maaaring magbago-bago ang presyo? Sa ganitong kaso, maaari mong subukan ang polynomial regression.

✅ Ang mga polynomials ay mga matematikal na ekspresyon na maaaring binubuo ng isa o higit pang mga variable at coefficient

Lumilikha ang polynomial regression ng isang kurbadong linya para mas maayos na maangkop ang nonlinear na data. Sa aming kaso, kung isasama natin ang squared na variable na `DayOfYear` sa input data, dapat nating maangkop ang data sa parabolic curve, na magkakaroon ng pinakamaliit na punto sa loob ng taon.

Kasama sa Scikit-learn ang kapaki-pakinabang na [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para pagsamahin ang iba't ibang hakbang ng pagpoproseso ng data. Ang **pipeline** ay isang kadena ng **estimators**. Sa aming kaso, gagawa tayo ng pipeline na unang magdaragdag ng polynomial features sa modelo, at saka magsasanay ng regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Ang paggamit ng `PolynomialFeatures(2)` ay nangangahulugan na isasama natin lahat ng second-degree polynomials mula sa input data. Sa aming kaso, ito ay nangangahulugan lamang ng `DayOfYear`<sup>2</sup>, ngunit kung may dalawang input variables na X at Y, ito ay magdadagdag ng X<sup>2</sup>, XY at Y<sup>2</sup>. Maaari rin tayong gumamit ng mas mataas na degree na mga polynomials kung nais.

Maaaring gamitin ang Pipelines sa parehong paraan tulad ng orihinal na `LinearRegression` na object, ibig sabihin maaari nating `fit` ang pipeline, at pagkatapos ay gamitin ang `predict` para makuha ang mga resulta ng hula. Narito ang graph na nagpapakita ng test data, at ng approximation curve:

<img alt="Polynomial regression" src="../../../../translated_images/tl/poly-results.ee587348f0f1f60b.webp" width="50%" />

Sa paggamit ng Polynomial Regression, makakakuha tayo ng bahagyang mas mababang MSE at mas mataas na coefficient of determination, ngunit hindi nang malaki. Kailangan nating isaalang-alang ang iba pang mga feature!

> Makikita mo na ang pinakamababang presyo ng kalabasa ay nakikita halos sa paligid ng Halloween. Paano mo ito maipapaliwanag?

🎃 Congratulations, nakagawa ka lang ng modelong makakatulong hulaan ang presyo ng pie pumpkins. Marahil ay maaari mong ulitin ang parehas na proseso para sa lahat ng uri ng kalabasa, ngunit ito ay magiging mahirap. Alamin natin ngayon kung paano isaalang-alang ang pagkakaiba ng uri ng kalabasa sa ating modelo!

## Categorical Features

Sa perpektong mundo, nais nating mahulaan ang mga presyo para sa iba't ibang uri ng kalabasa gamit ang parehong modelo. Gayunpaman, ang column na `Variety` ay iba mula sa mga column tulad ng `Month`, dahil naglalaman ito ng mga non-numeric na halaga. Ang ganitong mga column ay tinatawag na **categorical**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 I-click ang larawan sa itaas para sa maikling video overview tungkol sa paggamit ng categorical features.

Dito makikita kung paano nakadepende ang average na presyo sa variety:

<img alt="Average price by variety" src="../../../../translated_images/tl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para isaalang-alang ang variety, kailangan muna natin itong i-convert sa numeric form, o **i-encode**. Mayroong ilang paraan kung paano ito gagawin:

* Ang simple **numeric encoding** ay gagawa ng table ng iba't ibang varieties, at papalitan ang pangalan ng variety ng isang index sa table na iyon. Hindi ito magandang ideya para sa linear regression, dahil tinatanggap ng linear regression ang aktwal na numeric value ng index, at dinadagdagan ito sa resulta, na pinararami ng isang coefficient. Sa aming kaso, malinaw na hindi linear ang relasyon sa pagitan ng index number at presyo, kahit pa siguraduhin nating nakaayos ang mga indices sa isang partikular na paraan.
* Ang **one-hot encoding** ay papalitan ang column na `Variety` ng 4 na hiwalay na mga column, isa para sa bawat variety. Bawat column ay magkakaroon ng `1` kung ang katumbas na row ay isang partikular na variety, at `0` naman kung hindi. Nangangahulugan ito na magkakaroon ng apat na coefficients sa linear regression, isa para sa bawat uri ng kalabasa, na siyang responsable para sa "simula na presyo" (o mas tamang sabihin ay "karagdagang presyo") para sa partikular na variety na iyon.

Ipinapakita ng code sa ibaba kung paano tayo mag-one-hot encode ng variety:

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

Para magsanay ng linear regression gamit ang one-hot encoded variety bilang input, kailangan lang nating i-initialize nang tama ang `X` at `y` data:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Ang iba pang bahagi ng code ay pareho ng ginamit natin sa itaas para magsanay ng Linear Regression. Kapag sinubukan mo ito, makikita mo na ang mean squared error ay halos pareho, ngunit nakakakuha tayo ng mas mataas na coefficient of determination (~77%). Para makakuha ng mas tumpak na mga hula, maaari nating isaalang-alang ang mas maraming categorical features, pati na rin ang numeric features, tulad ng `Month` o `DayOfYear`. Para makuha ang isang malaking array ng features, maaari nating gamitin ang `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Dito isinasaalang-alang din natin ang `City` at uri ng `Package`, na nagbibigay sa atin ng MSE na 2.84 (10%), at determination na 0.94!

## Pinagsasama-sama

Para makagawa ng pinakamahusay na modelo, maaari nating pagsamahin (one-hot encoded categorical + numeric) data mula sa halimbawa sa itaas kasabay ng Polynomial Regression. Narito ang kumpletong code para sa iyong kaginhawaan:

```python
# ihanda ang datos para sa pagsasanay
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# gumawa ng paghahati sa pagsasanay at pagsubok
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# i-setup at sanayin ang pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# hulaan ang mga resulta para sa test data
pred = pipeline.predict(X_test)

# kalkulahin ang MSE at determinasyon
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Ito ay dapat magbigay sa atin ng pinakamataas na coefficient of determination na halos 97%, at MSE=2.23 (~8% na error sa prediksyon).

| Modelo | MSE | Determination |
|-------|-----|---------------|
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |
| `Variety` Linear | 5.24 (19.7%) | 0.77 |
| Lahat ng feature Linear | 2.84 (10.5%) | 0.94 |
| Lahat ng feature Polynomial | 2.23 (8.25%) | 0.97 |

🏆 Magaling! Nakagawa ka ng apat na Regression na modelo sa isang leksiyon, at napabuti ang kalidad ng modelo sa 97%. Sa huling bahagi tungkol sa Regression, matututuhan mo ang tungkol sa Logistic Regression para magtakda ng mga kategorya.

---
## 🚀Hamong Gawain

Subukan ang ilang iba't ibang variables sa notebook na ito para makita kung paano nauugnay ang correlation sa katumpakan ng modelo.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Repasuhin at Pag-aaral sa Sarili

Sa leksyong ito ay natutuhan natin ang tungkol sa Linear Regression. May iba pang mahahalagang uri ng Regression. Basahin tungkol sa Stepwise, Ridge, Lasso at Elasticnet na mga pamamaraan. Isang magandang kurso para pag-aralan ay ang [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Takdang Aralin

[Build a Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Paalala**:  
Ang dokumentong ito ay isinalin gamit ang serbisyong AI na pagsasalin na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagamat nagsusumikap kaming maging tumpak, pakatandaan na maaaring may mga pagkakamali o di-tiyak na bahagi ang awtomatikong pagsasalin. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na opisyal na sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagutan sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->