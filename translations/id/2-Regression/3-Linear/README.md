# Membangun model regresi menggunakan Scikit-learn: regresi dengan empat cara

## Catatan Pemula

Regresi linear digunakan ketika kita ingin memprediksi **nilai numerik** (misalnya, harga rumah, suhu, atau penjualan).
Ini bekerja dengan menemukan garis lurus yang paling mewakili hubungan antara fitur input dan output.

Dalam pelajaran ini, kita fokus pada pemahaman konsep sebelum mengeksplorasi teknik regresi yang lebih canggih.
![Infografis regresi linear vs polinomial](../../../../translated_images/id/linear-polynomial.5523c7cb6576ccab.webp)
> Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kuis Pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam Bahasa R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Pendahuluan

Sejauh ini Anda telah menjelajahi apa itu regresi dengan data contoh yang diambil dari dataset harga labu yang akan kita gunakan sepanjang pelajaran ini. Anda juga telah memvisualisasikannya menggunakan Matplotlib.

Sekarang Anda siap untuk menggali lebih dalam regresi untuk ML. Visualisasi memungkinkan Anda memahami data, tetapi kekuatan nyata dari Pembelajaran Mesin datang dari _pelatihan model_. Model dilatih pada data historis untuk secara otomatis menangkap ketergantungan data, dan memungkinkan Anda memprediksi hasil untuk data baru, yang belum pernah dilihat model sebelumnya.

Dalam pelajaran ini, Anda akan belajar lebih banyak tentang dua jenis regresi: _regresi linear dasar_ dan _regresi polinomial_, beserta sebagian matematika yang mendasari teknik-teknik ini. Model-model tersebut akan memungkinkan kita memprediksi harga labu tergantung pada data input yang berbeda.

[![ML untuk pemula - Memahami Regresi Linear](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML untuk pemula - Memahami Regresi Linear")

> 🎥 Klik gambar di atas untuk video singkat tentang regresi linear.

> Sepanjang kurikulum ini, kami mengasumsikan pengetahuan matematika minimal, dan berusaha membuatnya dapat diakses bagi siswa dari bidang lain, jadi perhatikan catatan, 🧮 catatan khusus, diagram, dan alat pembelajaran lain untuk membantu pemahaman.

### Prasyarat

Anda seharusnya sudah familiar dengan struktur data labu yang sedang kita periksa. Data tersebut sudah dimuat dan dibersihkan sebelumnya dalam file _notebook.ipynb_ pelajaran ini. Dalam file tersebut, harga labu ditampilkan per bushel dalam sebuah data frame baru. Pastikan Anda dapat menjalankan notebook ini di kernel Visual Studio Code.

### Persiapan

Sebagai pengingat, Anda memuat data ini untuk mengajukan pertanyaan tentangnya.

- Kapan waktu terbaik untuk membeli labu?
- Harga berapa yang bisa saya harapkan untuk sekotak labu mini?
- Haruskah saya membelinya dalam keranjang setengah bushel atau kotak 1 1/9 bushel?
Mari kita terus menggali data ini.

Dalam pelajaran sebelumnya, Anda telah membuat sebuah data frame Pandas dan mengisinya dengan sebagian dataset asli, menstandarisasi harga berdasarkan bushel. Namun dengan cara itu, Anda hanya dapat mengumpulkan sekitar 400 data dan hanya untuk bulan-bulan musim gugur.

Lihatlah data yang sudah dimuat dalam notebook pelajaran ini. Data sudah dimuat dan sebuah scatterplot awal telah dibuat untuk menunjukkan data bulan. Mungkin kita bisa mendapatkan detail lebih lanjut tentang sifat data dengan membersihkannya lebih lanjut.

## Garis regresi linear

Seperti yang Anda pelajari di Pelajaran 1, tujuan latihan regresi linear adalah untuk memplot sebuah garis guna:

- **Menunjukkan hubungan variabel**. Menunjukkan hubungan antara variabel
- **Membuat prediksi**. Membuat prediksi yang akurat tentang di mana titik data baru jatuh dalam hubungan dengan garis tersebut.

Biasanya **Regresi Kuadrat Terkecil** digunakan untuk menggambar garis seperti ini. Istilah "Kuadrat Terkecil" merujuk pada proses meminimalkan total kesalahan dalam model kita. Untuk setiap titik data, kita mengukur jarak vertikal (disebut residual) antara titik sebenarnya dan garis regresi kita.

Jarak ini dikuadratkan karena dua alasan utama:

1. **Magnitudo, bukan Arah:** Kita ingin memperlakukan kesalahan -5 sama dengan kesalahan +5. Mengkuadratkan membuat semua nilai menjadi positif.

2. **Memberi Bobot pada Outlier:** Mengkuadratkan memberikan bobot lebih pada kesalahan besar, memaksa garis tetap dekat dengan titik yang jauh.

Kemudian kita menjumlahkan semua nilai kuadrat ini. Tujuan kita adalah menemukan garis tertentu di mana jumlah akhir ini paling kecil (nilai terkecil mungkin)—itulah sebabnya dinamakan "Kuadrat Terkecil".

> **🧮 Tunjukkan matematika**
> 
> Garis ini, yang disebut _garis terbaik_ dapat diekspresikan dengan [persamaan](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` adalah 'variabel penjelas'. `Y` adalah 'variabel tergantung'. Kemiringan garis adalah `b` dan `a` adalah intercept-y, yang merujuk pada nilai `Y` ketika `X = 0`.
>
>![menghitung kemiringan](../../../../translated_images/id/slope.f3c9d5910ddbfcf9.webp)
>
> Pertama, hitung kemiringan `b`. Infografis oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Dengan kata lain, dan mengacu pada pertanyaan asli data labu kita: "memprediksi harga labu per bushel berdasarkan bulan", `X` merujuk pada harga dan `Y` merujuk pada bulan penjualan.
>
>![melengkapi persamaan](../../../../translated_images/id/calculation.a209813050a1ddb1.webp)
>
> Hitung nilai Y. Jika Anda membayar sekitar $4, pasti bulan April! Infografis oleh [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika yang menghitung garis harus menunjukkan kemiringan garis, yang juga bergantung pada intercept, atau di mana posisi `Y` saat `X = 0`.
>
> Anda bisa melihat metode perhitungan nilai ini di situs web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Juga kunjungi [Kalkulator kuadrat terkecil ini](https://www.mathsisfun.com/data/least-squares-calculator.html) untuk melihat bagaimana nilai angka mempengaruhi garis.

## Korelasi

Satu istilah lagi yang harus dipahami adalah **Koefisien Korelasi** antara variabel X dan Y yang diberikan. Dengan scatterplot, Anda dapat dengan cepat memvisualisasikan koefisien ini. Plot dengan titik data yang tersebar dalam garis rapi memiliki korelasi tinggi, tetapi plot dengan titik data tersebar di mana-mana antara X dan Y memiliki korelasi rendah.

Model regresi linear yang baik adalah yang memiliki Koefisien Korelasi tinggi (lebih dekat ke 1 daripada 0) menggunakan metode Regresi Kuadrat Terkecil dengan garis regresi.

✅ Jalankan notebook yang menyertai pelajaran ini dan lihat scatterplot Bulan ke Harga. Apakah data yang mengaitkan Bulan ke Harga untuk penjualan labu tampak memiliki korelasi tinggi atau rendah menurut interpretasi visual Anda dari scatterplot? Apakah itu berubah jika Anda menggunakan ukuran yang lebih rinci daripada `Bulan`, misalnya *hari dalam tahun* (yaitu jumlah hari sejak awal tahun)?

Dalam kode berikut, kita akan mengasumsikan bahwa kita telah membersihkan data, dan memperoleh data frame bernama `new_pumpkins`, serupa dengan berikut:

ID | Bulan | HariDalamTahun | Varietas | Kota | Kemasan | Harga Terendah | Harga Tertinggi | Harga  
---|-------|-------------|----------|------|---------|----------------|-----------------|--------
70 | 9 | 267 | PIE TYPE | BALTIMORE | Karton 1 1/9 bushel | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | PIE TYPE | BALTIMORE | Karton 1 1/9 bushel | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | PIE TYPE | BALTIMORE | Karton 1 1/9 bushel | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | PIE TYPE | BALTIMORE | Karton 1 1/9 bushel | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | PIE TYPE | BALTIMORE | Karton 1 1/9 bushel | 15.0 | 15.0 | 13.636364  

> Kode untuk membersihkan data tersedia di [`notebook.ipynb`](notebook.ipynb). Kami telah melakukan langkah pembersihan yang sama seperti pelajaran sebelumnya, dan telah menghitung kolom `DayOfYear` menggunakan ekspresi berikut:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sekarang setelah Anda memahami matematika di balik regresi linear, mari kita buat model Regresi untuk melihat apakah kita bisa memprediksi paket labu mana yang akan memiliki harga labu terbaik. Seseorang yang membeli labu untuk patch labu liburan mungkin ingin informasi ini agar bisa mengoptimalkan pembelian paket labu untuk patch tersebut.

## Mencari Korelasi

[![ML untuk pemula - Mencari Korelasi: Kunci Regresi Linear](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML untuk pemula - Mencari Korelasi: Kunci Regresi Linear")

> 🎥 Klik gambar di atas untuk video singkat tentang korelasi.

Dari pelajaran sebelumnya Anda mungkin sudah melihat bahwa harga rata-rata untuk bulan yang berbeda tampak seperti ini:

<img alt="Harga rata-rata per bulan" src="../../../../translated_images/id/barchart.a833ea9194346d76.webp" width="50%"/>

Ini menunjukkan bahwa seharusnya ada beberapa korelasi, dan kita dapat mencoba melatih model regresi linear untuk memprediksi hubungan antara `Month` dan `Price`, atau antara `DayOfYear` dan `Price`. Berikut adalah scatter plot yang menunjukkan hubungan kedua:

<img alt="Scatter plot Harga vs Hari Dalam Tahun" src="../../../../translated_images/id/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Mari kita lihat apakah ada korelasi menggunakan fungsi `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Terlihat bahwa korelasi cukup kecil, -0.15 berdasarkan `Month` dan -0.17 berdasarkan `DayOfMonth`, tetapi mungkin ada hubungan penting lainnya. Tampak seperti ada kelompok harga yang berbeda yang berhubungan dengan varietas labu yang berbeda. Untuk mengonfirmasi hipotesis ini, mari plot setiap kategori labu dengan warna berbeda. Dengan memberikan parameter `ax` ke fungsi plot `scatter` kita dapat menggambar semua titik dalam satu grafik:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot Harga vs Hari Dalam Tahun dengan warna" src="../../../../translated_images/id/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Investigasi kami menunjukkan bahwa varietas memiliki efek lebih besar pada harga keseluruhan dibandingkan tanggal penjualan sebenarnya. Kita dapat melihat ini dengan grafik batang:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Grafik batang harga vs varietas" src="../../../../translated_images/id/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Mari kita fokus untuk sementara hanya pada satu varietas labu, 'pie type', dan lihat apa efek tanggal terhadap harga:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot Harga vs Hari Dalam Tahun untuk varietas pie" src="../../../../translated_images/id/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Jika sekarang kita hitung korelasi antara `Price` dan `DayOfYear` menggunakan fungsi `corr`, akan didapat nilai sekitar `-0.27`—yang berarti pelatihan model prediktif masuk akal.

> Sebelum melatih model regresi linear, penting untuk memastikan bahwa data kita bersih. Regresi linear tidak bekerja dengan baik dengan nilai yang hilang, jadi masuk akal untuk menghapus semua sel kosong:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Pendekatan lain adalah mengisi nilai kosong tersebut dengan nilai rata-rata dari kolom yang bersangkutan.

## Regresi Linear Sederhana

[![ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML untuk pemula - Regresi Linear dan Polinomial menggunakan Scikit-learn")

> 🎥 Klik gambar di atas untuk video singkat tentang regresi linear dan polinomial.

Untuk melatih model Regresi Linear kita, kita akan gunakan perpustakaan **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Kita mulai dengan memisahkan nilai input (fitur) dan output yang diharapkan (label) ke dalam array numpy terpisah:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Perlu dicatat bahwa kita harus melakukan `reshape` pada data input agar paket Regresi Linear memahaminya dengan benar. Regresi Linear mengharapkan input berupa array 2D, di mana setiap baris array adalah vektor fitur input. Dalam kasus kita, karena hanya ada satu input - kita membutuhkan array berbentuk N&times;1, di mana N adalah ukuran dataset.

Selanjutnya, kita perlu membagi data menjadi dataset latih dan uji, agar kita dapat memvalidasi model setelah pelatihan:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Akhirnya, pelatihan model Regresi Linear sebenarnya hanya memerlukan dua baris kode. Kita definisikan objek `LinearRegression`, dan fit-kan ke data kita menggunakan metode `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objek `LinearRegression` setelah melakukan `fit` berisi semua koefisien regresi, yang dapat diakses menggunakan properti `.coef_`. Dalam kasus kami, hanya ada satu koefisien, yang seharusnya sekitar `-0.017`. Ini berarti bahwa harga tampaknya turun sedikit seiring waktu, tetapi tidak terlalu banyak, sekitar 2 sen per hari. Kita juga dapat mengakses titik potong regresi dengan sumbu Y menggunakan `lin_reg.intercept_` - ini akan sekitar `21` dalam kasus kami, menunjukkan harga di awal tahun.

Untuk melihat seberapa akurat model kami, kita dapat memprediksi harga pada dataset uji, dan kemudian mengukur seberapa dekat prediksi kami dengan nilai yang diharapkan. Ini dapat dilakukan menggunakan metrik mean square error (MSE), yaitu rata-rata dari semua selisih kuadrat antara nilai yang diharapkan dan nilai yang diprediksi.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Kesalahan kami tampaknya sekitar 2 poin, yaitu ~17%. Tidak terlalu baik. Indikator lain dari kualitas model adalah **koefisien determinasi**, yang dapat diperoleh seperti ini:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Jika nilainya 0, itu berarti model tidak mempertimbangkan data input, dan bertindak sebagai *prediktor linier terburuk*, yaitu nilai rata-rata hasil. Nilai 1 berarti kita dapat memprediksi semua hasil yang diharapkan dengan sempurna. Dalam kasus kami, koefisiennya sekitar 0,06, yang cukup rendah.

Kita juga dapat memplot data uji bersama dengan garis regresi untuk melihat lebih baik bagaimana regresi bekerja dalam kasus kami:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/id/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regresi Polinomial

Jenis lain dari Regresi Linear adalah Regresi Polinomial. Walaupun terkadang ada hubungan linier antara variabel - semakin besar labu dalam volume, semakin tinggi harga - terkadang hubungan tersebut tidak dapat diplot sebagai bidang atau garis lurus.

✅ Berikut adalah [beberapa contoh lagi](https://online.stat.psu.edu/stat501/lesson/9/9.8) data yang bisa menggunakan Regresi Polinomial

Lihat lagi hubungan antara Tanggal dan Harga. Apakah scatterplot ini harus dianalisis dengan garis lurus? Apakah harga tidak bisa berfluktuasi? Dalam kasus ini, Anda bisa mencoba regresi polinomial.

✅ Polinomial adalah ekspresi matematika yang mungkin terdiri dari satu atau lebih variabel dan koefisien

Regresi polinomial membuat garis lengkung untuk menyesuaikan data nonlinier dengan lebih baik. Dalam kasus kami, jika kami menambahkan variabel `DayOfYear` kuadrat ke data input, kami harus bisa menyesuaikan data kami dengan kurva parabola, yang akan memiliki titik minimum di suatu titik dalam tahun.

Scikit-learn menyertakan [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) yang membantu untuk menggabungkan berbagai langkah pemrosesan data bersama. Sebuah **pipeline** adalah rantai dari **estimator**. Dalam kasus kami, kami akan membuat pipeline yang pertama menambahkan fitur polinomial ke model, kemudian melatih regresi:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Menggunakan `PolynomialFeatures(2)` berarti kami akan menyertakan semua polinomial derajat dua dari data input. Dalam kasus kami ini berarti `DayOfYear`<sup>2</sup>, tetapi jika ada dua variabel input X dan Y, ini akan menambahkan X<sup>2</sup>, XY dan Y<sup>2</sup>. Kita juga dapat menggunakan polinomial derajat lebih tinggi jika ingin.

Pipeline dapat digunakan dengan cara yang sama seperti objek `LinearRegression` asli, yaitu kita dapat melakukan `fit` pipeline, dan kemudian menggunakan `predict` untuk mendapatkan hasil prediksi. Berikut grafik yang menunjukkan data uji dan kurva aproksimasi:

<img alt="Polynomial regression" src="../../../../translated_images/id/poly-results.ee587348f0f1f60b.webp" width="50%" />

Dengan Regresi Polinomial, kita bisa mendapatkan MSE yang sedikit lebih rendah dan koefisien determinasi yang lebih tinggi, tetapi tidak signifikan. Kita perlu mempertimbangkan fitur lain!

> Anda dapat melihat bahwa harga labu minimal diamati kira-kira di sekitar Halloween. Bagaimana Anda menjelaskannya?

🎃 Selamat, Anda baru saja membuat model yang dapat membantu memprediksi harga labu pai. Anda mungkin dapat mengulangi prosedur yang sama untuk semua jenis labu, tapi itu akan melelahkan. Sekarang mari kita pelajari bagaimana memperhitungkan varietas labu dalam model kita!

## Fitur Kategorikal

Dalam dunia ideal, kita ingin dapat memprediksi harga untuk berbagai varietas labu menggunakan model yang sama. Namun, kolom `Variety` agak berbeda dari kolom seperti `Month`, karena berisi nilai non-numerik. Kolom seperti ini disebut **kategorikal**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Klik gambar di atas untuk melihat video singkat tentang penggunaan fitur kategorikal.

Di sini Anda bisa melihat bagaimana harga rata-rata tergantung pada varietas:

<img alt="Average price by variety" src="../../../../translated_images/id/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Untuk memperhitungkan varietas, pertama kita harus mengubahnya ke bentuk numerik, atau **meng-encode**. Ada beberapa cara yang bisa kita lakukan:

* **Encoding numerik** sederhana akan membuat tabel varietas yang berbeda, kemudian mengganti nama varietas dengan indeks dalam tabel itu. Ini bukan ide terbaik untuk regresi linier, karena regresi linier mengambil nilai numerik indeks itu, dan menambahkan ke hasil, dikalikan dengan koefisien tertentu. Dalam kasus kami, hubungan antara nomor indeks dan harga jelas non-linier, meskipun kami memastikan indeks diurutkan dengan cara tertentu.
* **One-hot encoding** akan mengganti kolom `Variety` dengan 4 kolom berbeda, satu untuk setiap varietas. Setiap kolom akan berisi `1` jika baris terkait adalah varietas tersebut, dan `0` jika tidak. Ini berarti akan ada empat koefisien dalam regresi linier, satu untuk setiap varietas labu, yang bertanggung jawab untuk "harga awal" (atau lebih tepatnya "harga tambahan") untuk varietas itu.

Kode di bawah ini menunjukkan bagaimana kita bisa melakukan one-hot encoding untuk varietas:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | VARIETAS CAMPURAN HEIRLOOM | TIPE PAI
----|-----------|-----------|----------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Untuk melatih regresi linier menggunakan varietas yang telah di-one-hot encode sebagai input, kita hanya perlu menginisialisasi data `X` dan `y` dengan benar:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Sisa kode sama dengan yang kita gunakan sebelumnya untuk melatih Regresi Linier. Jika Anda mencobanya, Anda akan melihat bahwa mean squared error kurang lebih sama, tetapi kita mendapatkan koefisien determinasi yang jauh lebih tinggi (~77%). Untuk mendapatkan prediksi yang lebih akurat lagi, kita dapat mempertimbangkan lebih banyak fitur kategorikal, serta fitur numerik, seperti `Month` atau `DayOfYear`. Untuk mendapatkan satu array fitur besar, kita bisa menggunakan `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Di sini kita juga memperhitungkan `City` dan tipe `Package`, yang memberi kita MSE 2.84 (10%), dan koefisien determinasi 0.94!

## Menggabungkan Semua

Untuk membuat model terbaik, kita dapat menggunakan data gabungan (kategori one-hot encoded + numerik) dari contoh di atas bersama dengan Regresi Polinomial. Berikut adalah kode lengkap untuk kemudahan Anda:

```python
# siapkan data pelatihan
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# buat pemisahan train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# siapkan dan latih pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# prediksi hasil untuk data tes
pred = pipeline.predict(X_test)

# hitung MSE dan determinasi
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Ini harus memberi kita koefisien determinasi terbaik hampir 97%, dan MSE=2.23 (~8% kesalahan prediksi).

| Model | MSE | Koefisien Determinasi |
|-------|-----|-----------------------|
| Regresi Linear `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Regresi Polinomial `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Regresi Linear `Variety` | 5.24 (19.7%) | 0.77 |
| Semua fitur Linear | 2.84 (10.5%) | 0.94 |
| Semua fitur Polinomial | 2.23 (8.25%) | 0.97 |

🏆 Bagus sekali! Anda telah membuat empat model Regresi dalam satu pelajaran, dan meningkatkan kualitas model menjadi 97%. Di bagian terakhir tentang Regresi, Anda akan belajar tentang Regresi Logistik untuk menentukan kategori.

---
## 🚀 Tantangan

Uji beberapa variabel berbeda dalam notebook ini untuk melihat bagaimana korelasi berhubungan dengan akurasi model.

## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Belajar Mandiri

Dalam pelajaran ini kita belajar tentang Regresi Linear. Ada jenis Regresi penting lainnya. Bacalah tentang teknik Stepwise, Ridge, Lasso, dan Elasticnet. Kursus bagus untuk dipelajari lebih jauh adalah [kursus Stanford Statistical Learning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tugas

[Buat Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk mencapai akurasi, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa sumber harus dianggap sebagai sumber yang sahih. Untuk informasi penting, disarankan menggunakan jasa terjemahan manusia profesional. Kami tidak bertanggung jawab atas kesalahpahaman atau kesalahan tafsir yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->