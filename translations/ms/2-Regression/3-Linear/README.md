# Bina model regresi menggunakan Scikit-learn: empat cara regresi

## Nota Pemula

Regresi linear digunakan apabila kita ingin meramalkan **nilai berangka** (contohnya, harga rumah, suhu, atau jualan).
Ia berfungsi dengan mencari garis lurus yang paling mewakili hubungan antara ciri input dan output.

Dalam pelajaran ini, kita fokus pada memahami konsep sebelum meneroka teknik regresi yang lebih maju.
![Infografik regresi linear vs polinomial](../../../../translated_images/ms/linear-polynomial.5523c7cb6576ccab.webp)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kuiz pra kuliah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Pengenalan 

Setakat ini anda telah menerokai apa itu regresi dengan data contoh yang diambil dari set data harga labu yang akan kita gunakan sepanjang pelajaran ini. Anda juga telah memvisualisasikannya menggunakan Matplotlib.

Kini anda sudah bersedia untuk mendalami regresi bagi ML. Walaupun visualisasi membolehkan anda memahami data, kekuatan sebenar Pembelajaran Mesin datang daripada _melatih model_. Model dilatih menggunakan data sejarah untuk menangkap kebergantungan data secara automatik, dan ia membolehkan anda meramalkan hasil untuk data baru, yang mana model belum pernah lihat sebelum ini.

Dalam pelajaran ini, anda akan mempelajari lebih lanjut tentang dua jenis regresi: _regresi linear asas_ dan _regresi polinomial_, bersama dengan sedikit matematik yang mendasari teknik-teknik ini. Model-model tersebut akan membolehkan kita meramalkan harga labu bergantung pada data input yang berbeza. 

[![ML untuk pemula - Memahami Regresi Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML untuk pemula - Memahami Regresi Linear")

> 🎥 Klik imej di atas untuk video ringkas pengenalan kepada regresi linear.

> Sepanjang kurikulum ini, kami menganggap pengetahuan matematik adalah minimum, dan berusaha menjadikannya boleh diakses oleh pelajar daripada bidang lain, jadi perhatikan nota, 🧮 catatan, rajah, dan alat pembelajaran lain sebagai bantuan pemahaman.

### Prasyarat

Anda sepatutnya sudah biasa dengan struktur data labu yang sedang kita periksa. Anda boleh dapati ia telah dimuatkan dan dibersihkan dalam fail _notebook.ipynb_ pelajaran ini. Dalam fail itu, harga labu dipaparkan per bushel dalam bingkai data baru. Pastikan anda boleh menjalankan buku nota ini dalam kernel Visual Studio Code.

### Persiapan

Sebagai peringatan, anda memuatkan data ini supaya dapat bertanya soalan mengenainya.

- Bilakah masa terbaik untuk membeli labu?
- Berapakah harga yang boleh saya jangkakan untuk satu kotak labu kecil?
- Perlukah saya membelinya dalam bakul separuh bushel atau kotak 1 1/9 bushel?
Mari kita terus meneliti data ini.

Dalam pelajaran sebelum ini, anda telah mencipta bingkai data Pandas dan mengisinya dengan sebahagian dataset asal, menetapkan harga mengikut bushel. Dengan berbuat demikian, anda hanya berjaya mengumpul kira-kira 400 titik data dan hanya untuk bulan musim luruh.

Lihat data yang telah dimuatkan dalam buku nota pelajaran ini. Data dimuatkan dan plot taburan awal dipaparkan untuk menunjukkan data bulan. Mungkin kita boleh mendapatkan lebih banyak maklumat tentang sifat data dengan membersihkannya lebih lanjut.

## Garis regresi linear

Seperti yang anda pelajari dalam Pelajaran 1, tujuan latihan regresi linear adalah untuk dapat melukis garis untuk:

- **Menunjukkan hubungan pembolehubah**. Menunjukkan hubungan antara pembolehubah
- **Membuat ramalan**. Membuat ramalan tepat tentang di mana titik data baru akan jatuh berbanding garis itu.

Adalah biasa untuk **Regresi Kuasa Dua Terkecil (Least-Squares Regression)** melukis jenis garis ini. Istilah "Least-Squares" merujuk kepada proses meminimumkan jumlah ralat dalam model kita. Untuk setiap titik data, kita mengukur jarak menegak (dipanggil residual) antara titik sebenar dan garis regresi kita.

Kita kuasakan dua jarak ini atas dua sebab utama:

1. **Magnitud lebih penting daripada Arah:** Kita ingin menganggap ralat -5 sama seperti ralat +5. Kuasa dua menjadikan semua nilai positif.

2. **Menghukum Outlier:** Kuasa dua memberi lebih berat kepada ralat besar, memaksa garis berada lebih dekat dengan titik yang jauh.

Kemudian kami jumlahkan semua nilai kuasa dua ini bersama. Matlamat kami adalah untuk mencari garis khusus di mana jumlah akhir ini adalah paling kecil (nilai terkecil yang mungkin) — sebab itulah namanya "Least-Squares".

> **🧮 Tunjukkan matematiknya**
> 
> Garis ini, dipanggil _garis pemadanan terbaik_ boleh dinyatakan oleh [persamaan](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` adalah 'pembolehubah penerangan'. `Y` adalah 'pembolehubah bergantung'. Cerun garis adalah `b` dan `a` adalah pintasan-y, merujuk kepada nilai `Y` apabila `X = 0`.
>
>![kira cerun](../../../../translated_images/ms/slope.f3c9d5910ddbfcf9.webp)
>
> Pertama, hitung cerun `b`. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Dengan kata lain, dan merujuk kepada soalan asal data labu kita: "meramalkan harga labu per bushel mengikut bulan", `X` merujuk kepada harga dan `Y` merujuk kepada bulan jualan.
>
>![lengkapkan persamaan](../../../../translated_images/ms/calculation.a209813050a1ddb1.webp)
>
> Kira nilai Y. Jika anda membayar sekitar $4, mesti April! Infografik oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Matematik yang mengira garis mesti menunjukkan cerun garis, yang juga bergantung pada pintasan, atau di mana `Y` terletak apabila `X = 0`.
>
> Anda boleh memerhatikan kaedah pengiraan nilai-nilai ini di laman web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Juga lawati [Kalkulator Least-squares ini](https://www.mathsisfun.com/data/least-squares-calculator.html) untuk melihat bagaimana nilai nombor mempengaruhi garis.

## Korelasi

Satu lagi istilah yang perlu difahami adalah **Pekali Korelasi** antara pembolehubah X dan Y tertentu. Menggunakan plot taburan, anda boleh dengan cepat memvisualisasikan pekali ini. Plot dengan titik data bertaburan dalam garis kemas mempunyai korelasi tinggi, tetapi plot dengan titik data bertaburan di mana-mana antara X dan Y mempunyai korelasi rendah.

Model regresi linear yang baik adalah yang mempunyai Pekali Korelasi tinggi (lebih hampir ke 1 berbanding 0) menggunakan kaedah Regresi Least-Squares dengan garis regresi.

✅ Jalankan buku nota yang menyertai pelajaran ini dan lihat plot taburan Bulan ke Harga. Adakah data yang mengaitkan Bulan kepada Harga untuk jualan labu kelihatan mempunyai korelasi tinggi atau rendah menurut tafsiran visual anda tentang plot taburan? Adakah itu berubah jika anda menggunakan ukuran lebih terperinci daripada `Bulan`, contohnya *hari dalam tahun* (iaitu bilangan hari sejak awal tahun)?

Dalam kod di bawah, kita akan menganggap bahawa kita telah membersihkan data, dan memperoleh bingkai data yang dipanggil `new_pumpkins`, serupa seperti berikut:

ID | Bulan | DayOfYear | Varieti | Bandar | Pakej | Harga Rendah | Harga Tinggi | Harga
---|-------|-----------|---------|--------|-------|--------------|--------------|-------
70 | 9 | 267 | JENIS PAI | BALTIMORE | 1 1/9 karton bushel | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | JENIS PAI | BALTIMORE | 1 1/9 karton bushel | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | JENIS PAI | BALTIMORE | 1 1/9 karton bushel | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | JENIS PAI | BALTIMORE | 1 1/9 karton bushel | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | JENIS PAI | BALTIMORE | 1 1/9 karton bushel | 15.0 | 15.0 | 13.636364

> Kod untuk membersihkan data tersedia dalam [`notebook.ipynb`](notebook.ipynb). Kami telah melakukan langkah pembersihan yang sama seperti dalam pelajaran sebelum ini, dan telah mengira lajur `DayOfYear` menggunakan ungkapan berikut:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sekarang bahawa anda sudah memahami matematik di sebalik regresi linear, mari buat model Regresi untuk melihat sama ada kita boleh meramalkan pakej labu mana yang akan mempunyai harga labu terbaik. Seseorang yang membeli labu untuk taman labu percutian mungkin mahu maklumat ini untuk mengoptimumkan pembelian pakej labu mereka untuk taman tersebut.

## Mencari Korelasi

[![ML untuk pemula - Mencari Korelasi: Kunci kepada Regresi Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML untuk pemula - Mencari Korelasi: Kunci kepada Regresi Linear")

> 🎥 Klik imej di atas untuk video ringkas pengenalan kepada korelasi.

Daripada pelajaran sebelum ini anda mungkin telah melihat bahawa harga purata bagi bulan-bulan yang berbeza kelihatan seperti ini:

<img alt="Harga purata mengikut bulan" src="../../../../translated_images/ms/barchart.a833ea9194346d76.webp" width="50%"/>

Ini mencadangkan bahawa harus ada korelasi, dan kita boleh cuba melatih model regresi linear untuk meramalkan hubungan antara `Bulan` dan `Harga`, atau antara `DayOfYear` dan `Harga`. Berikut ialah plot taburan yang menunjukkan hubungan kedua:

<img alt="Plot taburan Harga vs. Hari dalam Tahun" src="../../../../translated_images/ms/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Mari kita lihat jika ada korelasi menggunakan fungsi `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Nampaknya korelasi agak kecil, -0.15 mengikut `Bulan` dan -0.17 mengikut `DayOfMonth`, tetapi boleh jadi ada hubungan penting lain. Nampaknya terdapat kelompok harga yang berbeza mengikut varieti labu yang berlainan. Untuk mengesahkan hipotesis ini, mari plot setiap kategori labu menggunakan warna berlainan. Dengan memasukkan parameter `ax` kepada fungsi plot `scatter`, kita boleh plot semua titik dalam graf yang sama:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Plot taburan Harga vs. Hari dalam Tahun dengan warna" src="../../../../translated_images/ms/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Penyiasatan kami mencadangkan varieti lebih memberi kesan ke atas harga keseluruhan berbanding tarikh jualan sebenar. Kita dapat lihat ini dengan graf bar:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Graf bar harga mengikut varieti" src="../../../../translated_images/ms/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Mari kita fokus buat sementara waktu hanya pada satu varieti labu, 'jenis pai', dan lihat apa kesan tarikh terhadap harga:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Plot taburan Harga vs. Hari dalam Tahun pada labu pai" src="../../../../translated_images/ms/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Jika kita kira korelasi antara `Harga` dan `DayOfYear` menggunakan fungsi `corr`, kita akan dapat nilai sekitar `-0.27` - yang bermakna melatih model ramalan adalah masuk akal.

> Sebelum melatih model regresi linear, penting untuk memastikan data kita bersih. Regresi linear tidak berfungsi dengan baik dengan nilai hilang, jadi masuk akal untuk menghapus semua sel kosong:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Pendekatan lain adalah mengisi nilai kosong itu dengan nilai purata dari lajur yang bersesuaian.

## Regresi Linear Ringkas

[![ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn")

> 🎥 Klik imej di atas untuk video ringkas pengenalan regresi linear dan polinomial.

Untuk melatih model Regresi Linear, kita akan menggunakan perpustakaan **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Kita mulakan dengan memisahkan nilai input (ciri) dan output yang dijangka (label) ke dalam array numpy berasingan:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Perhatikan bahawa kita perlu melakukan `reshape` pada data input supaya pakej Regresi Linear memahaminya dengan betul. Regresi Linear mengharapkan array 2D sebagai input, di mana setiap baris dalam array mewakili vektor ciri input. Dalam kes kita, kerana hanya ada satu input, kita memerlukan array berbentuk N&times;1, di mana N adalah saiz dataset.

Kemudian, kita perlu membahagikan data kepada dataset latihan dan ujian, supaya kita dapat mengesahkan model selepas latihan:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Akhirnya, melatih model Regresi Linear sebenar hanya mengambil dua baris kod. Kita definisikan objek `LinearRegression`, dan fitkan kepada data menggunakan kaedah `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objek `LinearRegression` selepas `fit` mengandungi semua pekali regresi, yang boleh diakses menggunakan sifat `.coef_`. Dalam kes kami, terdapat hanya satu pekali, yang sepatutnya sekitar `-0.017`. Ini bermakna harga nampaknya menurun sedikit dengan masa, tetapi tidak terlalu banyak, sekitar 2 sen sehari. Kita juga boleh mengakses titik persilangan regresi dengan paksi Y menggunakan `lin_reg.intercept_` - ia akan berada sekitar `21` dalam kes kami, menunjukkan harga pada awal tahun.

Untuk melihat betapa tepatnya model kami, kita boleh meramal harga pada set data ujian, dan kemudian mengukur sejauh mana ramalan kami hampir dengan nilai yang dijangka. Ini boleh dilakukan menggunakan metrik ralat kuasa dua min (MSE), yang merupakan purata semua beza kuasa dua antara nilai dijangka dan ramalan.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Ralat kami nampaknya sekitar 2 poin, iaitu ~17%. Tidak terlalu baik. Penunjuk lain kualiti model ialah **koefisien penentuan**, yang boleh diperoleh seperti ini:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Jika nilainya 0, ia bermakna model tidak mengambil kira data input, dan bertindak sebagai *peramal linear terburuk*, yang merupakan nilai purata hasil. Nilai 1 bermakna kita dapat meramalkan semua output yang dijangka dengan sempurna. Dalam kes kami, pekali itu sekitar 0.06, yang agak rendah.

Kita juga boleh plot data ujian bersama garis regresi untuk melihat dengan lebih baik bagaimana regresi berfungsi dalam kes kami:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/ms/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regresi Polinomial

Satu lagi jenis Regresi Linear ialah Regresi Polinomial. Walaupun kadangkala terdapat hubungan linear antara pemboleh ubah - labu yang lebih besar dari segi isipadu, harga yang lebih tinggi - kadangkala hubungan ini tidak boleh diplotkan sebagai satah atau garis lurus.

✅ Berikut adalah [beberapa contoh lagi](https://online.stat.psu.edu/stat501/lesson/9/9.8) data yang boleh menggunakan Regresi Polinomial

Lihat kembali hubungan antara Tarikh dan Harga. Adakah sebaran titik ini perlu dianalisis menggunakan garis lurus? Bukankah harga boleh berfluktuasi? Dalam kes ini, anda boleh cuba regresi polinomial.

✅ Polinomial adalah ekspresi matematik yang mungkin terdiri daripada satu atau lebih pemboleh ubah dan pekali

Regresi polinomial menghasilkan garis melengkung untuk menyesuaikan data bukan linear dengan lebih baik. Dalam kes kami, jika kami menyertakan pemboleh ubah `DayOfYear` kuasa dua dalam data input, kami sepatutnya dapat menyesuaikan data dengan lengkung parabola, yang akan mempunyai minimum pada suatu titik dalam tahun tersebut.

Scikit-learn termasuk [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) yang berguna untuk menggabungkan beberapa langkah pemprosesan data. **Pipeline** adalah rantai **penganggar**. Dalam kes kami, kami akan mencipta pipeline yang terlebih dahulu menambah ciri polinomial ke model kami, dan kemudian melatih regresi:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Menggunakan `PolynomialFeatures(2)` bermakna kami akan memasukkan semua polinomial darjah kedua dari data input. Dalam kes kami ia hanya bermakna `DayOfYear`<sup>2</sup>, tetapi dengan dua pemboleh ubah input X dan Y, ini akan menambah X<sup>2</sup>, XY dan Y<sup>2</sup>. Kami juga boleh menggunakan polinomial darjah lebih tinggi jika mahu.

Pipelines boleh digunakan sama seperti objek `LinearRegression` asal, iaitu kita boleh `fit` pipeline, dan kemudian menggunakan `predict` untuk mendapatkan hasil ramalan. Berikut ialah graf yang menunjukkan data ujian, dan lengkung anggaran:

<img alt="Polynomial regression" src="../../../../translated_images/ms/poly-results.ee587348f0f1f60b.webp" width="50%" />

Menggunakan Regresi Polinomial, kami boleh mendapatkan MSE yang sedikit lebih rendah dan penentuan lebih tinggi, tetapi tidak dengan signifikan. Kita perlu mengambil kira ciri-ciri lain!

> Anda dapat lihat bahawa harga labu paling rendah diperhatikan sekitar Halloween. Bagaimana anda boleh jelaskan ini?

🎃 Tahniah, anda baru sahaja mencipta model yang boleh membantu meramalkan harga labu pai. Anda mungkin boleh mengulangi prosedur yang sama untuk semua jenis labu, tetapi itu akan melecehkan. Mari kita pelajari sekarang bagaimana mengambil kira varieti labu dalam model kita!

## Ciri Kategori

Dalam dunia yang ideal, kita mahu dapat meramalkan harga untuk pelbagai jenis labu menggunakan model yang sama. Namun, lajur `Variety` agak berbeza daripada lajur seperti `Month`, kerana ia mengandungi nilai bukan berangka. Lajur seperti ini dipanggil **kategori**.

[![ML untuk pemula - Ramalan Ciri Kategori dengan Regresi Linear](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML untuk pemula - Ramalan Ciri Kategori dengan Regresi Linear")

> 🎥 Klik gambar di atas untuk video ringkas tentang penggunaan ciri kategori.

Di sini anda dapat lihat bagaimana harga purata bergantung pada varieti:

<img alt="Average price by variety" src="../../../../translated_images/ms/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Untuk mengambil kira varieti, kita perlu menukarkannya ke bentuk berangka terlebih dahulu, atau **mengekodkan** ia. Terdapat beberapa cara kita boleh lakukan:

* **Pengekodan berangka** mudah akan membina jadual pelbagai varieti, dan menggantikan nama varieti dengan indeks dalam jadual tersebut. Ini bukan idea terbaik bagi regresi linear, kerana regresi linear mengambil nilai berangka indeks sebenar, dan menambahkannya ke hasil, didarab dengan pekali tertentu. Dalam kes kami, hubungan antara nombor indeks dan harga jelas bukan linear, walaupun indeks diatur dengan cara tertentu.
* **Pengekodan satu-panas (one-hot encoding)** akan menggantikan lajur `Variety` dengan 4 lajur berbeza, satu untuk setiap varieti. Setiap lajur akan mengandungi `1` jika baris yang sepadan adalah daripada varieti tertentu, dan `0` jika tidak. Ini bermakna akan ada empat pekali dalam regresi linear, satu untuk setiap varieti labu, bertanggungjawab untuk "harga permulaan" (atau lebih tepat "harga tambahan") untuk varieti itu.

Kod di bawah menunjukkan cara kita boleh one-hot encode varieti:

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

Untuk melatih regresi linear menggunakan varieti yang telah one-hot encode sebagai input, kita hanya perlu inisialisasi data `X` dan `y` dengan betul:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Selepas itu, kodnya sama seperti yang kita gunakan sebelum ini untuk melatih Regresi Linear. Jika anda mencubanya, anda akan lihat bahawa ralat kuasa dua min adalah lebih kurang sama, tapi kita dapat koefisien penentuan yang jauh lebih tinggi (~77%). Untuk mendapatkan ramalan yang lebih tepat lagi, kita boleh mengambil lebih banyak ciri kategori serta ciri berangka, seperti `Month` atau `DayOfYear`. Untuk mendapat satu array ciri yang besar, kita boleh gunakan `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Di sini kita turut mengambil kira `City` dan jenis `Package`, yang memberikan MSE 2.84 (10%), dan penentuan 0.94!

## Menggabungkan semua

Untuk mendapatkan model terbaik, kita boleh gunakan data gabungan (kategori yang di-one-hot encode + berangka) dari contoh di atas bersama Regresi Polinomial. Berikut ialah kod lengkap untuk kemudahan anda:

```python
# sediakan data latihan
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# buat pembahagian latih-uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# sediakan dan latih laluan paip
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# ramal keputusan untuk data ujian
pred = pipeline.predict(X_test)

# kira MSE dan penentuan
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Ini sepatutnya memberikan koefisien penentuan terbaik hampir 97%, dan MSE=2.23 (~8% ralat ramalan).

| Model | MSE | Penentuan |
|-------|-----|-----------|
| Linear `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomial `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Linear `Variety` | 5.24 (19.7%) | 0.77 |
| Linear Semua Ciri | 2.84 (10.5%) | 0.94 |
| Polinomial Semua Ciri | 2.23 (8.25%) | 0.97 |

🏆 Tahniah! Anda telah mencipta empat model Regresi dalam satu pelajaran, dan meningkatkan kualiti model kepada 97%. Dalam bahagian akhir mengenai Regresi, anda akan belajar tentang Regresi Logistik untuk menentukan kategori.

---
## 🚀Cabaran

Uji beberapa pemboleh ubah berlainan dalam buku nota ini untuk melihat bagaimana korelasi berkaitan dengan ketepatan model.

## [Kuis Selepas Kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Belajar Sendiri

Dalam pelajaran ini kita belajar tentang Regresi Linear. Terdapat jenis Regresi penting yang lain. Bacalah tentang teknik Stepwise, Ridge, Lasso dan Elasticnet. Kursus yang bagus untuk belajar lebih lanjut ialah [kursus Pembelajaran Statistik Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tugasan

[Bina Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya hendaklah dianggap sebagai sumber yang sahih. Untuk maklumat yang penting, terjemahan profesional oleh manusia adalah disarankan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->