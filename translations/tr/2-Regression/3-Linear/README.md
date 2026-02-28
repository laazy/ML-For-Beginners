# Scikit-learn kullanarak regresyon modeli oluşturma: dört farklı regresyon yöntemi

## Başlangıç Notu

Lineer regresyon, **sayısal bir değeri** tahmin etmek istediğimizde kullanılır (örneğin, ev fiyatı, sıcaklık veya satışlar).  
Girdi özellikleri ile çıktı arasındaki ilişkiyi en iyi temsil eden doğruyu bulmakla çalışır.

Bu derste, daha ileri regresyon tekniklerini keşfetmeden önce kavramı anlamaya odaklanıyoruz.  
![Lineer ve polinom regresyon bilgigramı](../../../../translated_images/tr/linear-polynomial.5523c7cb6576ccab.webp)  
> Bilgigram: [Dasani Madipalli](https://twitter.com/dasani_decoded)  
## [Ön ders sınavı](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bu ders R dilinde de mevcut!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### Giriş

Şimdiye kadar, balkabağı fiyatlandırma veri setinden toplanan örnek verilerle regresyonun ne olduğunu keşfettiniz. Ayrıca Matplotlib kullanarak bu veriyi görselleştirdiniz.

Artık makine öğrenimi için regresyona daha derinlemesine dalmaya hazırsınız. Görselleştirme veriyi anlamanızı sağlarken, makine öğreniminin gerçek gücü _modellerin eğitilmesinden_ gelir. Modeller, geçmiş veriler üzerinde eğitilerek veri bağımlılıklarını otomatik olarak yakalar ve modelin daha önce görmediği yeni veriler için sonuçlar tahmin etmenizi sağlar.

Bu derste, _temel lineer regresyon_ ve _polinom regresyon_ olmak üzere iki regresyon türü ve bu tekniklerin altında yatan matematik hakkında daha fazla bilgi edineceksiniz. Bu modeller farklı girdi verilerine bağlı olarak balkabağı fiyatlarını tahmin etmemizi sağlayacak.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Lineer regresyona kısa video genel bakışı için yukarıdaki görsele tıklayın.

> Bu eğitim programı boyunca matematik bilgisi asgari düzeyde varsayılmakta ve farklı alanlardan gelen öğrenciler için erişilebilir olmayı hedeflemektedir. Bu yüzden notlar, 🧮 alıntılar, diyagramlar ve diğer öğrenme araçlarına dikkat edin.

### Ön Koşul

Şu ana kadar incelediğimiz balkabağı verisinin yapısına aşina olmalısınız. Bu dersin _notebook.ipynb_ dosyasında önceden yüklenmiş ve temizlenmiş olarak bulabilirsiniz. Bu dosyada balkabağı fiyatı bushel başına yeni bir veri çerçevesinde gösterilmiştir.  
Visual Studio Code'da kernel kullanarak bu not defterlerini çalıştırabildiğinizden emin olun.

### Hazırlık

Hatırlatma olarak, bu veriyi sorular sormak için yüklüyorsunuz.

- Balkabaklarını satın almak için en iyi zaman ne zaman?  
- Minyatür balkabaklarının bir kasası için hangi fiyatı bekleyebilirim?  
- Onları yarım bushel sepetlerde mi yoksa 1 1/9 bushel kutuda mı almalıyım?  
Veri üzerinde kazıma işlemine devam edelim.

Önceki derste, bir Pandas veri çerçevesi oluşturdunuz ve orijinal veri setinin bir kısmını bushel bazında fiyatlandırmayı standartlaştırarak doldurdunuz. Ancak bu şekilde sadece sonbahar ayları için yaklaşık 400 veri noktası toplamış oldunuz.

Bu derse eşlik eden not defterinde önceden yüklenmiş verilere göz atın. Veri ön yüklendi ve ay verisini göstermek için ilk dağılım grafiği çizildi. Veriyi daha fazla temizleyerek doğası hakkında biraz daha detay alabiliriz.

## Bir lineer regresyon doğrusu

Ders 1'de öğrendiğiniz gibi, lineer regresyon çalışmasının amacı şu şekildedir:

- **Değişken ilişkilerini göstermek**. Değişkenler arasındaki ilişkiyi göstermek  
- **Tahmin yapmak**. Yeni bir veri noktasının bu doğruya göre nerede yer alacağını doğru tahmin etmek

Bu tür doğruların çizimi için tipik olan **En Küçük Kareler Regresyonu** yöntemidir. "En Küçük Kareler" terimi, modelimizdeki toplam hatayı minimize etme sürecine atıfta bulunur. Her veri noktası için, gerçek nokta ile regresyon doğrumuz arasındaki dikey mesafeyi (rezidü olarak da adlandırılır) ölçeriz.

Bu mesafeleri iki temel nedenle karesini alırız:

1. **Büyüklük yönünden üstünlük:** -5 hata ile +5 hata aynı şekilde ele alınmalı. Karesini almak tüm değerleri pozitif yapar.

2. **Aykırı Değerlere Ceza:** Karesini almak daha büyük hatalara daha fazla ağırlık verir ve doğrunun uzak noktaların yakınında kalmasını zorunlu kılar.

Sonra bu karelenmiş değerlerin tümünü toplarız. Amacımız bu toplamın en küçük olduğu doğruyu bulmaktır; bu yüzden adı "En Küçük Kareler"dir.

> **🧮 Matematiği Göster**  
>  
> Bu doğru, _en uygun uyum doğrusudur_ ve [bir denklemle](https://en.wikipedia.org/wiki/Simple_linear_regression) ifade edilebilir:  
>  
> ```
> Y = a + bX
> ```
>  
> `X` 'açıklayıcı değişken'dir. `Y` ise 'bağımlı değişken'dir. Doğrunun eğimi `b` ve `a` y-kesiti olup, `X = 0` olduğunda `Y` değerini ifade eder.  
>  
>![eğimi hesapla](../../../../translated_images/tr/slope.f3c9d5910ddbfcf9.webp)  
> Önce eğimi `b` hesaplayın. Bilgigram: [Jen Looper](https://twitter.com/jenlooper)  
>  
> Başka bir deyişle ve balkabağı verimizin asıl sorusuna atıfta bulunursak: "ay bazında bushel başına balkabağı fiyatını tahmin et", burada `X` fiyatı, `Y` ise satış ayını temsil eder.  
>  
>![denklemi tamamla](../../../../translated_images/tr/calculation.a209813050a1ddb1.webp)  
> Y'nin değerini hesaplayın. Yaklaşık $4 ödüyorsanız, bu kesinlikle Nisan ayıdır! Bilgigram: [Jen Looper](https://twitter.com/jenlooper)  
>  
> Doğruyu hesaplayan matematik, doğrunun eğimini göstermeli, bu da kesit değerine, yani `Y`'nin `X = 0` olduğundaki konumuna bağlıdır.  
>  
> Bu değerlerin hesaplanma yöntemini [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) sitesinde görebilirsiniz. Ayrıca sayıların çizgi üzerindeki etkisini görmek için [bu En-Küçük Kareler hesaplayıcısını](https://www.mathsisfun.com/data/least-squares-calculator.html) ziyaret edin.

## Korelasyon

Anlamanız gereken bir diğer terim, verilen X ve Y değişkenleri arasındaki **Korelasyon Katsayısı**dır. Bir dağılım grafiği kullanarak bu katsayıyı hızlıca görselleştirebilirsiniz. Noktaların düzgün bir doğru üzerindeyse yüksek korelasyon, her yere saçılmışsa düşük korelasyon vardır.

İyi bir lineer regresyon modeli, En Küçük Kareler Regresyon yöntemi ve bir regresyon doğrusu kullanarak yüksek (0'a değil 1'e daha yakın) bir Korelasyon Katsayısına sahip olacaktır.

✅ Bu derse eşlik eden not defterini çalıştırın ve Ay ile Fiyat arasındaki dağılım grafiğine bakın. Görsel değerlendirmenize göre Balkabağı satışları için Ay ile Fiyat arasındaki veri yüksek mi yoksa düşük korelasyon mu gösteriyor? Daha ayrıntılı bir ölçüm olarak `Ay` yerine *yılın günü* (örneğin, yıl başından itibaren geçen gün sayısı) kullanırsanız bu değişir mi?

Aşağıdaki kodda, verinin temizlendiğini ve `new_pumpkins` adında aşağıdakine benzer bir veri çerçevesi elde edildiğini varsayıyoruz:

ID | Ay | YilinGunu | Çeşit | Şehir | Paket | Düşük Fiyat | Yüksek Fiyat | Fiyat  
---|-------|-----------|---------|------|---------|-----------|------------|-------  
70 | 9 | 267 | TURTA TİPİ | BALTIMORE | 1 1/9 bushel karton | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | TURTA TİPİ | BALTIMORE | 1 1/9 bushel karton | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | TURTA TİPİ | BALTIMORE | 1 1/9 bushel karton | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | TURTA TİPİ | BALTIMORE | 1 1/9 bushel karton | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | TURTA TİPİ | BALTIMORE | 1 1/9 bushel karton | 15.0 | 15.0 | 13.636364  

> Veriyi temizleme kodu [`notebook.ipynb`](notebook.ipynb) dosyasında mevcuttur. Önceki derste yaptığımız aynı temizlik adımlarını uyguladık ve `DayOfYear` sütununu aşağıdaki ifadeyi kullanarak hesapladık:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Lineer regresyonun matematiğini anladığınıza göre, hangi balkabağı paketinin en iyi fiyatlara sahip olacağını tahmin etmek için bir Regresyon modeli oluşturalım. Birisi tatil balkabağı bahçesinde balkabağı satın alıyorsa, bahçe için balkabağı paketlerini optimize etmek amacıyla bu bilgi faydalı olabilir.

## Korelasyon Arayışı

[![ML for beginners - Korelasyon Arayışı: Lineer Regresyonun Anahtarı](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Korelasyon Arayışı: Lineer Regresyonun Anahtarı")

> 🎥 Korelasyon hakkında kısa video genel bakışı için yukarıdaki görsele tıklayın.

Önceki dersten muhtemelen farklı aylar için ortalama fiyatların şu şekilde olduğunu görmüştünüz:

<img alt="Aya göre ortalama fiyat" src="../../../../translated_images/tr/barchart.a833ea9194346d76.webp" width="50%"/>

Bu, bir korelasyon olabileceğini düşündürür ve `Ay` ile `Fiyat` ya da `YilinGunu` ile `Fiyat` arasındaki ilişkiyi tahmin etmek için lineer regresyon modeli eğitebiliriz. İşte sonuncuyu gösteren dağılım grafiği:

<img alt="Fiyat vs. Yılın Günü dağılım grafiği" src="../../../../translated_images/tr/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />  

Korelasyonu `corr` fonksiyonuyla görelim:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Görünüşe göre korelasyon oldukça düşük, `Ay` için -0.15 ve `YilinGunu` için -0.17, ancak başka önemli bir ilişki olabilir. Farklı balkabağı çeşitlerine karşılık gelen farklı fiyat kümeleri var gibi. Bu hipotezi doğrulamak için her balkabağı kategorisini farklı renklerle çizelim. `scatter` çizim fonksiyonuna bir `ax` parametresi geçirilerek tüm noktalar aynı grafikte çizilebilir:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Fiyat vs. Yılın Günü (renkli) dağılım grafiği" src="../../../../translated_images/tr/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />  

Araştırmalarımız, çeşidin satış tarihinden daha fazla fiyat üzerinde etkisi olduğunu gösteriyor. Bunu bir çubuk grafikle görebiliriz:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Çeşide göre fiyat çubuk grafiği" src="../../../../translated_images/tr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />  

Şimdilik yalnızca bir balkabağı çeşidine, 'turta tipi'ne, odaklanalım ve tarihin fiyat üzerindeki etkisine bakalım:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Fiyat vs. Yılın Günü dağılım grafiği" src="../../../../translated_images/tr/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />  

Şimdi `Price` ve `YilinGunu` arasında `corr` fonksiyonunu kullanarak korelasyon hesaplarsak, yaklaşık `-0.27` buluruz - ki bu da tahmin modeli eğitmenin mantıklı olduğunu gösterir.

> Lineer regresyon modeli eğitmeden önce, verinin temiz olması önemlidir. Lineer regresyon eksik değerlerle iyi çalışmaz, bu yüzden boş hücrelerden kurtulmak mantıklıdır:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
Diğer bir yaklaşım, boş değerleri ilgili sütunun ortalama değerleriyle doldurmaktır.

## Basit Lineer Regresyon

[![ML for beginners - Scikit-learn ile Lineer ve Polinom Regresyon](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Scikit-learn ile Lineer ve Polinom Regresyon")

> 🎥 Lineer ve polinom regresyon hakkında kısa video genel bakışı için yukarıdaki görsele tıklayın.

Lineer Regresyon modelimizi eğitmek için **Scikit-learn** kütüphanesini kullanacağız.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
Girdi değerlerini (özellikler) ve beklenen çıktıyı (etiket) ayrı numpy dizilerine ayırarak başlıyoruz:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Lineer Regresyon paketinin doğru anlayabilmesi için girdi verilerini `reshape` yapmamız gerektiğine dikkat edin. Lineer Regresyon, her satırı bir özellik vektörünü temsil eden 2D diziyi bekler. Bizim durumumuzda sadece bir girdi olduğundan, N×1 yapısında bir dizi gerekir, burada N veri seti büyüklüğüdür.

Sonra, modeli eğitip doğrulayabilmek için veriyi eğitim ve test veri setlerine bölmemiz gerekiyor:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
Son olarak, gerçek Lineer Regresyon modelinin eğitimi sadece iki kod satırı alır. `LinearRegression` nesnesini tanımlarız ve `fit` metodu ile verimize uyarlarız:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```
  
`LinearRegression` nesnesi `fit` edildikten sonra regresyonun tüm katsayılarını içerir ve bunlara `.coef_` özelliği aracılığıyla erişilebilir. Bizim durumumuzda, sadece bir katsayı vardır ve bu yaklaşık `-0.017` civarında olmalıdır. Bu, fiyatların zamanla biraz azaldığını ancak çok fazla olmadığını, yaklaşık olarak günde 2 sent düştüğünü gösterir. Regresyonun Y ekseniyle kesiştiği noktaya ise `lin_reg.intercept_` kullanılarak erişilebilir - bizim durumumuzda bu yaklaşık `21` olacak ve yılın başındaki fiyatı gösterecektir.

Modelimizin ne kadar doğru olduğunu görmek için test veri seti üzerinde fiyatları tahmin edebilir ve ardından tahminlerimizin beklenen değerlere ne kadar yakın olduğunu ölçebiliriz. Bu, beklenen ve tahmin edilen değer arasındaki tüm kare farklarının ortalaması olan ortalama kare hata (MSE) metriği ile yapılabilir.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Hata oranımız yaklaşık 2 puan, yani ~%17 civarında görünüyor. Çok iyi değil. Model kalitesinin diğer bir göstergesi ise **belirleme katsayısı**dır ve bu şöyle elde edilir:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Eğer değer 0 ise, modelin giriş verilerini dikkate almadığı ve *en kötü doğrusal tahminci* gibi davrandığı anlamına gelir, bu da sonuçların sadece ortalaması anlamına gelir. 1 değeri ise tüm beklenen çıktıları mükemmel şekilde tahmin edebildiğimizi gösterir. Bizim durumumuzda, katsayı yaklaşık 0.06, bu da oldukça düşüktür.

Regresyonun nasıl çalıştığını daha iyi görmek için test verilerini ve regresyon doğrusunu birlikte çizebiliriz:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/tr/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polinom Regresyonu

Lineer Regresyonun bir başka türü de Polinom Regresyonudur. Bazen değişkenler arasında doğrusal bir ilişki olur - örneğin, balkabağının hacmi ne kadar büyükse, fiyat da o kadar yüksek olur - bazen bu ilişkiler düz bir düzlem veya doğru olarak çizilemez.

✅ İşte Polinom Regresyonu için kullanılabilecek [başka örnekler](https://online.stat.psu.edu/stat501/lesson/9/9.8)

Tarih ve Fiyat arasındaki ilişkiye yeniden bakın. Bu dağılım grafiği gerçekten düz bir doğru ile analiz edilmeli mi? Fiyatlar dalgalanamaz mı? Bu durumda polinom regresyonu deneyebilirsiniz.

✅ Polinomlar, bir veya daha fazla değişken ve katsayı içerebilen matematiksel ifadeleridir.

Polinom regresyon, doğrusal olmayan verilere daha iyi uyması için eğri bir çizgi oluşturur. Bizim durumumuzda, giriş verisine kare `DayOfYear` değişkeni eklendiğinde, verilerimizi yıl içinde belirli bir noktada minimuma sahip bir parabolik eğri ile uydurabilmeliyiz.

Scikit-learn, veri işleme adımlarını birleştirmek için faydalı bir [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) içerir. Bir **pipeline** bir **tahminci** zinciridir. Bizim durumumuzda, önce modele polinom özellikler ekleyen ve ardından regresyonu eğiten bir pipeline oluşturacağız:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)` kullanmak, giriş verisinden tüm ikinci dereceden polinomları dahil edeceğimiz anlamına gelir. Bizim örneğimizde bu yalnızca `DayOfYear`<sup>2</sup> demektir, ancak iki giriş değişkeni X ve Y verilirse, bu X<sup>2</sup>, XY ve Y<sup>2</sup> öğelerini ekler. Dilerseniz daha yüksek dereceli polinomlar da kullanabilirsiniz.

Pipeline'lar, orijinal `LinearRegression` nesnesi gibi kullanılabilir; yani pipeline'ı `fit` edebilir ve ardından tahmin sonuçları almak için `predict` kullanabilirsiniz. İşte test verisi ve yaklaşıklaştırma eğrisini gösteren grafik:

<img alt="Polynomial regression" src="../../../../translated_images/tr/poly-results.ee587348f0f1f60b.webp" width="50%" />

Polinom Regresyon kullanarak MSE biraz daha düşük ve belirleme katsayısı biraz daha yüksek olabilir, ama çok büyük bir fark olmaz. Diğer özellikleri de göz önünde bulundurmamız gerekir!

> Minimum balkabağı fiyatlarının Cadılar Bayramı civarında gözlemlendiğini görebilirsiniz. Bunu nasıl açıklarsınız?

🎃 Tebrikler, balkabağı turta fiyatını tahmin eden bir model oluşturdunuz. Muhtemelen tüm balkabağı türleri için aynı işlemi tekrarlayabilirsiniz, ama bu zahmetli olur. Şimdi modelimizde balkabağı çeşidini nasıl dikkate alacağımızı öğrenelim!

## Kategorik Özellikler

İdeal dünyada, farklı balkabağı çeşitleri için fiyatları aynı modelle tahmin etmek isteriz. Ancak, `Variety` sütunu `Month` gibi sütunlardan farklıdır çünkü sayısal olmayan değerler içerir. Bu tür sütunlara **kategorik** denir.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Kategorik özelliklerin kullanımına dair kısa video özetini izlemek için yukarıdaki resme tıklayın.

Burada ortalama fiyatın çeşitliliğe nasıl bağlı olduğunu görebilirsiniz:

<img alt="Average price by variety" src="../../../../translated_images/tr/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Çeşidi dikkate almak için öncelikle sayısal forma çevirmemiz veya **kodlamamız** gerekir. Bunu yapmanın birkaç yolu vardır:

* Basit **sayısal kodlama**, farklı çeşitlerin bir tablosunu oluşturur ve ardından çeşit adını bu tablodaki bir indeksle değiştirir. Bu lineer regresyon için en iyi yöntem değildir, çünkü lineer regresyon indeksin gerçek sayısal değerini alıp bir katsayı ile çarpar ve sonuca ekler. Bizim durumumuzda, indeks numarası ile fiyat arasındaki ilişki açıkça doğrusal değildir; hatta indekslerin belirli bir şekilde sıralanması garanti edilse bile.
* **One-hot encoding** ile `Variety` sütunu, her bir çeşit için 4 farklı sütuna bölünür. Her sütun, ilgili satır o çeşide aitse `1`, değilse `0` içerir. Bu, lineer regresyonda balkabağı çeşidi başına biri "başlangıç fiyatı" (ya da "ek fiyat") için olmak üzere dört katsayı olacağı anlamına gelir.

Aşağıdaki kod, bir çeşidin one-hot kodlamasını nasıl yapabileceğimizi gösteriyor:

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

One-hot kodlanmış çeşidi giriş olarak kullanarak lineer regresyonu eğitmek için sadece `X` ve `y` verilerini doğru şekilde başlatmamız gerekir:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Geri kalan kod, yukarıda lineer regresyonu eğittiğimiz kod ile aynıdır. Denerseniz, ortalama kare hatasının neredeyse aynı olduğunu ancak belirleme katsayısının (~%77) çok daha yüksek olduğunu görürsünüz. Daha doğru tahminler elde etmek için daha fazla kategorik özellik ile `Month` veya `DayOfYear` gibi sayısal özellikleri de dikkate alabiliriz. Tüm özellikleri tek bir büyük diziye dönüştürmek için `join` kullanabiliriz:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Burada ayrıca `City` ve `Package` türü de dikkate alınmıştır, bu bize 2.84 (%%10) MSE ve 0.94 belirleme katsayısı sağlar!

## Hepsini Bir Araya Getirmek

En iyi modeli yapmak için yukarıdaki örnekten (one-hot kodlanmış kategorik + sayısal veriler) ve Polinom Regresyonu birlikte kullanabiliriz. İşte kolayınız için tam kod:

```python
# eğitim verilerini ayarla
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# eğitim-test bölümü yap
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# pipeline'ı kur ve eğit
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# test verileri için sonuçları tahmin et
pred = pipeline.predict(X_test)

# MSE ve belirleme katsayısını hesapla
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Bu en iyi belirleme katsayısı olan neredeyse %97 ve MSE=2.23 (yaklaşık %8 tahmin hatası) verecektir.

| Model | MSE | Belirleme Katsayısı |
|-------|-----|---------------------|
| `DayOfYear` Lineer | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polinom | 2.73 (17.0%) | 0.08 |
| `Variety` Lineer | 5.24 (19.7%) | 0.77 |
| Tüm özellikler Lineer | 2.84 (10.5%) | 0.94 |
| Tüm özellikler Polinom | 2.23 (8.25%) | 0.97 |

🏆 Çok iyi! Bu derste dört farklı Regresyon modeli oluşturup model kalitesini %97’ye geliştirdiniz. Son Regresyon bölümünde, kategorileri belirlemek için Lojistik Regresyonu öğreneceksiniz.

---
## 🚀Meydan Okuma

Bu not defterinde farklı değişkenlerle deney yaparak korelasyonun model doğruluğuna nasıl karşılık geldiğini test edin.

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme & Kendi Kendine Çalışma

Bu derste Lineer Regresyonu öğrendik. Başka önemli Regresyon türleri de vardır. Stepwise, Ridge, Lasso ve Elasticnet teknikleri hakkında okuyun. Daha fazla öğrenmek için iyi bir kurs [Stanford İstatistiksel Öğrenme kursu](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Ödev

[Bir Model Oluştur](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri servisi [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstermemize rağmen, otomatik çevirilerin hatalar veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlaşılmalardan veya yorum hatalarından sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->