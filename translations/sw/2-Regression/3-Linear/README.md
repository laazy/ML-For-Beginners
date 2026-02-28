# Jenga mfano wa urekebishaji ukitumia Scikit-learn: urekebishaji kwa njia nne

## Kumbuka kwa Mwanzo

Urekebishaji wa mstari hutumiwa tunapotaka kutabiri **thamani ya nambari** (kwa mfano, bei ya nyumba, joto, au mauzo).
Hufanya kazi kwa kupata mstari wa moja kwa moja unaowakilisha vizuri uhusiano kati ya vipengele vya ingizo na matokeo.

Katika somo hili, tunazingatia kuelewa dhana kabla ya kuchunguza mbinu za urekebishaji za hali ya juu.
![Linear vs polynomial regression infographic](../../../../translated_images/sw/linear-polynomial.5523c7cb6576ccab.webp)
> Picha ya taarifa na [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Mtihani kabla ya mihadhara](https://ff-quizzes.netlify.app/en/ml/)

> ### [Somo hili linapatikana pia kwa R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Utangulizi

Hadi sasa umechambua ni nini urekebishaji kwa kutumia data ya sampuli kutoka kwenye dataset ya bei ya malenge ambayo tutatumia katika somo hili lote. Pia umeionyesha picha kwa kutumia Matplotlib.

Sasa uko tayari kuingia zaidi katika urekebishaji kwa ML. Wakati uoneshaji picha unakuwezesha kuelewa data, nguvu halisi ya Kujifunza kwa Mashine hutoka kwa _mafunzo ya mifano_. Mifano huandaliwa kwa data ya kihistoria ili kiotomatiki kushika utegemezi wa data, na hukuruhusu kutabiri matokeo kwa data mpya, ambayo mfano haujawahi kuona hapo awali.

Katika somo hili, utajifunza zaidi kuhusu aina mbili za urekebishaji: _urekebishaji wa mstari wa msingi_ na _urekebishaji wa polinomial_, pamoja na baadhi ya hisabati zinazotegemea mbinu hizi. Mifano hiyo itatuwezesha kutabiri bei za malenge kulingana na data ya ingizo tofauti.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya muhtasari wa urekebishaji wa mstari.

> Katika mtaala huu mzima, tunadhani kuwa ujuzi wa hisabati ni mdogo, na tunalenga kuufanya ufikike kwa wanafunzi wanaotoka katika nyanja nyingine, hivyo angalia kumbukumbu, 🧮 maelezo, michoro, na zana nyingine za kujifunzia kusaidia kuelewa.

### Masharti ya Awali

Unapaswa kuwa unafahamu sasa muundo wa data ya malenge tunayochambua. Unaweza kuipata tayari imepakwa na kusafishwa katika faili la _notebook.ipynb_ la somo hili. Katika faili hilo, bei ya malenge inaonyeshwa kwa kila bushel katika fremu mpya ya data. Hakikisha unaweza kuendesha daftari hizi katika kernels ndani ya Visual Studio Code.

### Maandalizi

Kama kumbusho, unachukua data hii ili kuuliza maswali kuhusu yake.

- Ni wakati gani bora kununua malenge?
- Bei gani naweza kutarajia kwa kesi ya malenge madogo?
- Je, napaswa kununua katika vikapu vya nusu bushel au kwa sanduku la 1 1/9 bushel?
Tuwekeze zaidi katika uchambuzi wa data hii.

Katika somo lililopita, uliunda fremu ya data ya Pandas na kuijaza kwa sehemu ya dataset asili, ukifanya viwango vya bei iwe kwa bushel. Hata hivyo, kwa kufanya hivyo, uliweza kukusanya takriban taarifa 400 tu na kwa miezi ya vuli pekee.

Tazama data tuliyoipakia tayari katika daftari la somo hili. Data imepakiwa tayari na muundo wa awali wa pointi umeonyeshwa kuonyesha data ya miezi. Labda tunaweza kupata maelezo zaidi juu ya tabia ya data kwa kusafisha zaidi.

## Mstari wa urekebishaji wa mstari

Kama ulivyojifunza katika Somo la 1, lengo la zoezi la urekebishaji wa mstari ni kuwa na uwezo wa kuchora mstari wa:

- **Kuonyesha uhusiano wa vigezo**. Onyesha uhusiano kati ya vigezo
- **Kutabiri matokeo**. Tengeneza utabiri sahihi wa mahali ambapo pointi mpya itapatikana kwenye mstari huo.

Ni kawaida kwa **Urekebishaji wa Mafuta Madogo** kuchora mstari wa aina hii. Neno "Least-Squares" linahusu mchakato wa kupunguza jumla ya makosa katika mfano wetu. Kwa kila pointi ya data, tunapima umbali wima (uitwao residual) kati ya pointi halisi na mstari wetu wa urekebishaji.

Tunafanyia mraba umbali huu kwa sababu mbili kuu:

1. **Ukubwa zaidi ya Mwelekeo:** Tunataka kushughulikia kosa la -5 kama lilivyo kosa la +5. Kufanya mraba kunafanya thamani zote kuwa chanya.

2. **Kurekebisha Pointi za Mbali:** Kufanya mraba kunatoa uzito zaidi kwa makosa makubwa, na kulazimisha mstari kubaki karibu na pointi ambazo ziko mbali.

Kisha tunaongeza thamani hizi zote za mraba pamoja. Lengo letu ni kupata mstari maalumu ambapo jumla hii ya mwisho ni ndogo zaidi (thamani ndogo kabisa)—ndiyo maana inaitwa "Least-Squares".

> **🧮 Nionyeshe hisabati**
> 
> Mstari huu, uitwao _mstari wa kuendana bora_ unaweza kuoneshwa na [hesabu](https://en.wikipedia.org/wiki/Simple_linear_regression):
> 
> ```
> Y = a + bX
> ```
>
> `X` ni 'kigezo kinachoeleza'. `Y` ni 'kigezo kinachotegemea'. Mteremko wa mstari ni `b` na `a` ni kukatwa kwa y, ambayo ina maana ya thamani ya `Y` wakati `X = 0`.
>
>![hesabu mteremko](../../../../translated_images/sw/slope.f3c9d5910ddbfcf9.webp)
>
> Kwanza, hesabu mteremko `b`. Picha ya taarifa na [Jen Looper](https://twitter.com/jenlooper)
>
> Kwa maneno mengine, na ukitazama swali la awali kuhusu data ya malenge: "tabiri bei ya malenge kwa kila bushel kwa mwezi", `X` itahusu bei na `Y` itasemekana ni mwezi wa mauzo.
>
>![kamilisha hesabu](../../../../translated_images/sw/calculation.a209813050a1ddb1.webp)
>
> Hesabu thamani ya Y. Ikiwa unalipa karibu $4, lazima iwe Aprili! Picha ya taarifa na [Jen Looper](https://twitter.com/jenlooper)
>
> Hesabu ambayo hufanya mstari lazima ionyeshe mteremko wa mstari, ambao pia unategemea sehemu ya kukatwa, au sehemu ambapo `Y` iko wakati `X = 0`.
>
> Unaweza kuona njia ya hesabu ya hizi thamani kwenye tovuti ya [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Pia tembelea [kalkuleta hii ya Least-squares](https://www.mathsisfun.com/data/least-squares-calculator.html) kuangalia jinsi thamani za nambari zinavyoathiri mstari.

## Uhusiano

Neno zaidi la kuelewa ni **Kiwango cha Uhusiano** kati ya vigezo `X` na `Y` vilivyotolewa. Kutumia mchoro wa pointi, unaweza kuona haraka kiwango hiki. Mchoro wenye pointi zilizopangwa katika mstari mzuri una kiwango kikubwa cha uhusiano, lakini mchoro wenye pointi zilizoenea kila mahali kati ya X na Y una kiwango kidogo cha uhusiano.

Mfano mzuri wa urekebishaji wa mstari utakuwa ule wenye Kiwango cha Uhusiano cha juu (karibu na 1 kuliko 0) ukitumia mbinu ya Urekebishaji wa Mafuta Madogo na mstari wa urekebishaji.

✅ Endesha daftari la mazoezi linaloambatana na somo hili na tazama mchoro wa Mia kwa Bei. Je, data inayohusisha Mia na Bei kwa mauzo ya malenge inaonekana kuwa na uhusiano mkubwa au mdogo, kulingana na tafsiri yako ya picha? Je, hiyo inabadilika ikiwa utatumia kipimo chenye undani zaidi badala ya `Mwezi`, mfano *siku ya mwaka* (yaani, idadi ya siku tangu mwanzo wa mwaka)?

Katika msimbo hapa chini, tutaendelea kudhani kuwa tumesafisha data, na kupata fremu ya data iitwayo `new_pumpkins`, inayofanana na ifuatayo:

ID | Mwezi | SikuYaMwaka | Aina | Jiji | Pakiti | Bei ya Chini | Bei ya Juu | Bei
---|--------|-------------|-------|-----|--------|--------------|-----------|-----
70 | 9 | 267 | AINA YA PAI | BALTIMORE | 1 1/9 katoni za bushel | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | AINA YA PAI | BALTIMORE | 1 1/9 katoni za bushel | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | AINA YA PAI | BALTIMORE | 1 1/9 katoni za bushel | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | AINA YA PAI | BALTIMORE | 1 1/9 katoni za bushel | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | AINA YA PAI | BALTIMORE | 1 1/9 katoni za bushel | 15.0 | 15.0 | 13.636364

> Msimbo wa kusafisha data upo katika [`notebook.ipynb`](notebook.ipynb). Tumefanya hatua sawa za usafi kama katika somo lililopita, na tumekuwa tumehesabu safu ya `DayOfYear` kwa kutumia usemi ufuatao:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sasa ukiwa na uelewa wa hisabati nyuma ya urekebishaji wa mstari, hebu tujenge Mfano wa Urekebishaji kuona kama tunaweza kutabiri pakiti gani ya malenge itakuwa na bei bora zaidi. Mtu anayeinunua malenge kwa ajili ya shamba la malenge la sikukuu anaweza kutaka taarifa hii ili kuboresha ununuzi wake wa pakiti za malenge kwa shamba hilo.

## Kutafuta Uhusiano

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya muhtasari wa uhusiano.

Kutoka somo lililopita labda umeona kuwa bei ya wastani kwa miezi tofauti inaonekana kama hii:

<img alt="Average price by month" src="../../../../translated_images/sw/barchart.a833ea9194346d76.webp" width="50%"/>

Hii inaashiria kuwa kunapaswa kuwepo na uhusiano, na tunaweza kujaribu kufundisha mfano wa urekebishaji wa mstari kutabiri uhusiano kati ya `Mwezi` na `Bei`, au kati ya `SikuYaMwaka` na `Bei`. Hapa ni mchoro wa pointi unaoonyesha uhusiano wa mwisho:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sw/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />

Tuwe tuchunguze kama kuna uhusiano kwa kutumia kazi ya `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Inaonekana kuwa uhusiano ni mdogo, -0.15 kwa `Mwezi` na -0.17 kwa `SikuYaMwezi`, lakini kunaweza kuwa na uhusiano mwingine muhimu. Inaonekana kuna makundi tofauti ya bei yanayolingana na aina tofauti za malenge. Ili kuthibitisha dhana hii, tuchore kila kundi la malenge kwa rangi tofauti. Kwa kupitisha parameter `ax` kwa kazi ya kuchora `scatter` tunaweza kuchora pointi zote kwenye chati moja:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sw/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />

Uchunguzi wetu unaonyesha kuwa aina ina athari kubwa zaidi kwa bei kwa ujumla kuliko tarehe halisi ya mauzo. Tunaweza kuona hii kwenye grafu ya bar:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/sw/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Tuzingatie kwa sasa aina moja tu ya malenge, ‘aina ya pai’, na tuangalie athari ya tarehe kwa bei:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/sw/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />

Ikiwa sasa tutahesabu uhusiano kati ya `Bei` na `SikuYaMwaka` kwa kutumia kazi ya `corr`, tutapata kama `-0.27` - ambayo ina maana mafunzo ya mfano wa utabiri ni yenye maana.

> Kabla ya kufundisha mfano wa urekebishaji wa mstari, ni muhimu kuhakikisha data yetu ni safi. Urekebishaji wa mstari hauendani vizuri na vyanzo vya data vilivyo tupu, hivyo ni vyema kuondoa seli zote tupu:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Njia nyingine ni kujaza thamani hizo tupu kwa thamani za wastani kutoka safu husika.

## Urekebishaji Rahisi wa Mstari

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya muhtasari wa urekebishaji wa mstari na polinomial.

Ili kufundisha Mfano wetu wa Urekebishaji wa Mstari, tutatumia maktaba ya **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Tunaanza kwa kutenganisha thamani za ingizo (vipengele) na matokeo yanayotarajiwa (lebo) katika arrays za numpy tofauti:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Kumbuka tulilazimika kufanya `reshape` kwa data ya ingizo ili kifurushi cha Urekebishaji wa Mstari kiweze kuelewa vizuri. Urekebishaji wa mstari unatarajia 2D-array kama ingizo, ambapo kila safu ya array inalingana na vector ya vipengele vya ingizo. Katika kesi yetu, kwa kuwa tuna kipengele kimoja tu, tunahitaji array yenye umbo la N&times;1, ambapo N ni ukubwa wa dataset.

Kisha, tunahitaji kugawanya data kuwa seti za mafunzo na mtihani, ili tuweze kuhakiki mfano wetu baada ya mafunzo:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Mwisho, kufundisha mfano halisi wa Urekebishaji wa Mstari kunachukua mstari miwili tu ya msimbo. Tunaeleza kitu cha `LinearRegression`, na kukifit kwa data yetu kwa kutumia njia ya `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Kitu cha `LinearRegression` baada ya `fit`-ting kina coefficient zote za regression, ambazo zinaweza kupatikana kwa kutumia mali ya `.coef_`. Katika kesi yetu, kuna coefficient moja tu, ambayo inapaswa kuwa karibu na `-0.017`. Hii ina maana kuwa bei inaonekana kushuka kidogo na muda, lakini sio sana, takriban senti 2 kwa siku. Pia tunaweza kupata pointi ya kuingilia ya regression na mhimili wa Y kwa kutumia `lin_reg.intercept_` - itakuwa karibu na `21` katika kesi yetu, ikionyesha bei mwanzoni mwa mwaka.

Ili kuona jinsi modeli yetu ilivyo sahihi, tunaweza kutabiri bei kwenye dataset ya majaribio, kisha kupima jinsi utabiri wetu unavyofanana na thamani zinazotarajiwa. Hii inaweza kufanywa kwa kutumia kipimo cha makosa ya wastani ya mraba (MSE), ambacho ni wastani wa tofauti zilizokwazwa zote kati ya thamani zinazotarajiwa na zitabiriwa.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Hitilafu yetu inaonekana kuwa karibu na pointi 2, ambayo ni ~17%. Sio nzuri sana. Kiashiria kingine cha ubora wa modeli ni **coefficient ya utambuzi**, ambayo inaweza kupatikana hivi:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Kama thamani ni 0, inamaanisha kuwa modeli huchukua data ya pembejeo kwa upande na hutenda kama *mtabiri mbaya zaidi wa mstari*, ambayo ni thamani ya wastani ya matokeo tu. Thamani ya 1 ina maana tunaweza kutabiri kwa ukamilifu matokeo yote yanayotarajiwa. Katika kesi yetu, coefficient ni karibu 0.06, ambayo ni ya chini sana.

Pia tunaweza kuchora data ya majaribio pamoja na mstari wa regression kuona vizuri jinsi regression inavyofanya kazi katika kesi yetu:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/sw/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regression ya Polynomiale

Aina nyingine ya Linear Regression ni Polynomial Regression. Wakati mwingine kuna uhusiano wa mstari kati ya vigezo - vile pumpkin mkubwa kwa ujazo, ndivyo bei huwa juu - wakati mwingine uhusiano huu hauwezi kuchorwa kama uso wa ndege au mstari wa moja kwa moja.

✅ Hapa kuna [mifano mingine zaidi](https://online.stat.psu.edu/stat501/lesson/9/9.8) ya data ambazo zinaweza kutumia Polynomial Regression

Tazama tena uhusiano kati ya Tarehe na Bei. Je, scatterplot hii inaonekana inapaswa kuchambuliwa kwa mstari wa moja kwa moja? Je, bei haziwezi kubadilika? Katika kesi hii, unaweza jaribu polynomial regression.

✅ Polynomiale ni misemo ya kihisabati ambayo inaweza kuwa na kigezo kimoja au zaidi na coefficients

Polynomial regression huunda mstari uliokunja kuelekea kufaa data zisizo za mstari. Katika kesi yetu, kama tutajumuisha variable ya `DayOfYear` iliyofunguliwa kwa mraba katika data ya pembejeo, tunapaswa kuweza kufit data zetu kwa curve ya parabolic, ambayo itakuwa na chini katika sehemu fulani ndani ya mwaka.

Scikit-learn inajumuisha API ya [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) yenye msaada ya kuunganisha hatua tofauti za usindikaji wa data pamoja. **Pipeline** ni mnyororo wa **makadirio**. Katika kesi yetu, tutaunda pipeline ambayo kwanza inaongeza sifa za polynomial kwenye modeli yetu, kisha inafunza regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Kutumia `PolynomialFeatures(2)` inamaanisha kuwa tutajumuisha polynomiale zote za daraja la pili kutoka kwa data ya pembejeo. Katika kesi yetu itamaanisha tu `DayOfYear`<sup>2</sup>, lakini ikizingatiwa variable mbili za pembejeo X na Y, hii itaongeza X<sup>2</sup>, XY na Y<sup>2</sup>. Pia tunaweza kutumia polynomiale za daraja la juu zaidi ikiwa tunataka.

Pipelines zinaweza kutumika kwa namna ile ile kama kitu cha awali cha `LinearRegression`, yaani tunaweza `fit` pipeline, kisha kutumia `predict` kupata matokeo ya utabiri. Hii hapa grafu inayoonyesha data ya mtihani, na curve ya takriban:

<img alt="Polynomial regression" src="../../../../translated_images/sw/poly-results.ee587348f0f1f60b.webp" width="50%" />

Kutumia Polynomial Regression, tunaweza kupata MSE kidogo chini na utambuzi wa juu zaidi, lakini sio kwa kiasi kikubwa. Tunahitaji kuzingatia sifa nyingine!

> Unaweza kuona kuwa bei za chini zaidi za pumpkin hutokea karibu na Halloween. Unawezaje kuelezea hili? 

🎃 Hongera, umeunda modeli inayoweza kusaidia kutabiri bei ya pie pumpkins. Huenda ukarudia taratibu iliyofanana kwa aina zote za pumpkin, lakini hiyo itakuwa ya kuchosha. Hebu tujifunze sasa jinsi ya kuzingatia aina ya pumpkin katika modeli yetu!

## Sifa za Kategoria

Katika dunia bora, tunataka kuwa na uwezo wa kutabiri bei za aina tofauti za pumpkin kwa kutumia modeli moja. Hata hivyo, safu ya `Variety` ni tofauti kidogo na safu kama `Month`, kwa sababu ina thamani zisizo za nambari. Safu kama hizo huitwa **kachategoriali**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Bonyeza picha hapo juu kwa video fupi ya muhtasari wa kutumia sifa za kategoria.

Hapa unaona jinsi bei ya wastani inavyotegemea aina:

<img alt="Average price by variety" src="../../../../translated_images/sw/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Ili kuzingatia aina, kwanza tunahitaji kuibadilisha kuwa fomu ya nambari, au **kuandika kwa kanuni ya nambari**. Kuna njia kadhaa za kufanya hivi:

* **Uandikaji nambari rahisi** utaunda jedwali la aina tofauti, kisha kubadilisha jina la aina kwa index katika jedwali hilo. Hii siyo wazo bora kwa regression ya mstari, kwa sababu regression ya mstari huchukua thamani ya nambari halisi ya index, na kuiongeza kwa matokeo, ikizidishwa na coefficient fulani. Katika kesi yetu, uhusiano kati ya nambari ya index na bei ni dhahiri kuwa sio mstari, hata kama tunahakikisha kuwa index zimepangwa kwa njia maalum.
* **One-hot encoding** itabadilisha safu ya `Variety` kuwa safu 4 tofauti, moja kwa kila aina. Kila safu itakuwa na `1` ikiwa safu husika ni ya aina ile, na `0` vinginevyo. Hii ina maana kuwa kutakuwa na coefficient nne katika regression ya mstari, moja kwa kila aina ya pumpkin, inayohusika na "bei ya mwanzo" (au badala yake "bei ya ziada") kwa aina hiyo.

Msimbo hapa chini unaonyesha jinsi tunavyoweza kufanya one-hot encode kwa aina:

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

Ili kufunza regression ya mstari kwa kutumia aina iliyomo one-hot encoded kama pembejeo, tunahitaji tu kuanzisha data `X` na `y` kwa usahihi:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Msimbo mwingine ni ule ule tuliotumia hapo juu kufunza Linear Regression. Ukijaribu, utaona kuwa makosa ya wastani ya mraba ni takriban sawa, lakini tunapata coefficient ya utambuzi ya juu zaidi (~77%). Ili kupata utabiri sahihi zaidi, tunaweza kuzingatia sifa za kategoria zaidi, pamoja na sifa za nambari, kama vile `Month` au `DayOfYear`. Ili kupata array moja kubwa ya sifa, tunaweza kutumia `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Hapa tunazingatia pia `City` na aina ya `Package`, ambayo inatupa MSE 2.84 (10%), na coefficient 0.94!

## Kuunganisha Yote Pamoja

Ili kutengeneza modeli bora zaidi, tunaweza kutumia data iliyochanganywa (one-hot encoded kategoria + nambari) kutoka kwa mfano hapo juu pamoja na Polynomial Regression. Hapa ni msimbo kamili kwa urahisi wako:

```python
# weka data za mafunzo
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# tengeneza mgawanyiko wa mafunzo-na-majaribio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# weka na funza pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# tabiri matokeo kwa data ya majaribio
pred = pipeline.predict(X_test)

# hesabu MSE na uamuzi
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Hii inapaswa kutupatia coefficient bora kabisa ya utambuzi ya karibu 97%, na MSE=2.23 (~8% hitilafu ya utabiri).

| Modeli | MSE | Utambuzi |
|-------|-----|---------------|
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |
| `Variety` Linear | 5.24 (19.7%) | 0.77 |
| Zote sifa Linear | 2.84 (10.5%) | 0.94 |
| Zote sifa Polynomial | 2.23 (8.25%) | 0.97 |

🏆 Hongera! Umeunda modeli nne za Regression katika somo moja, na kuboresha ubora wa modeli hadi 97%. Katika sehemu ya mwisho kuhusu Regression, utajifunza kuhusu Logistic Regression kutambua makundi.

---
## 🚀Changamoto

Jaribu vigezo tofauti tofauti katika daftari hili kuona jinsi uwiano unavyolingana na usahihi wa modeli.

## [Mtihani baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Hakiki & Kujisomea

Katika somo hili tulijifunza kuhusu Linear Regression. Kuna aina nyingine muhimu za Regression. Soma kuhusu mbinu za Stepwise, Ridge, Lasso na Elasticnet. Kozi nzuri ya kujifunza zaidi ni [Kozi ya Stanford ya Kujifunza kwa Takwimu](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Kazi ya Nyumbani 

[Jenga Modeli](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tafadhali Kumbuka**:
Nyaraka hii imetatuliwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au upungufu wa usahihi. Nyaraka ya asili katika lugha yake lazima itambuliwe kama chanzo rasmi. Kwa taarifa muhimu, tafsiri ya mtaalamu wa binadamu inapendekezwa. Hatubebei wajibu wowote kuhusu kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->