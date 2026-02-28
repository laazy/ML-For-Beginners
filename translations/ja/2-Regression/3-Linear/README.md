# Scikit-learnで回帰モデルを構築する：回帰の4通りの方法

## 初心者向けノート

線形回帰は、**数値を予測したいとき**に使います（例：家の価格、気温、売上など）。
これは、入力特徴と出力の関係を最もよく表現する直線を見つけることで機能します。

このレッスンでは、より高度な回帰手法を探索する前に、概念の理解に焦点を当てます。
![線形回帰と多項式回帰のインフォグラフィック](../../../../translated_images/ja/linear-polynomial.5523c7cb6576ccab.webp)
> インフォグラフィック作成者：[Dasani Madipalli](https://twitter.com/dasani_decoded)
## [事前講義クイズ](https://ff-quizzes.netlify.app/en/ml/)

> ### [このレッスンはRでも利用可能です！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### はじめに

これまで、かぼちゃの価格設定データセットから収集したサンプルデータを使って回帰とは何かを探ってきました。このレッスン全体を通して使うデータです。また、Matplotlibを用いて可視化も行いました。

さて、MLの回帰についてさらに深く掘り下げる準備が整いました。可視化はデータを理解するのに役立ちますが、機械学習の真の力は_モデルの訓練_にあります。モデルは過去のデータで訓練され、データの依存関係を自動的に捉え、新たな未見のデータに対して結果を予測できるようになります。

このレッスンでは、_基本線形回帰_と_多項式回帰_の2種類の回帰についてと、それらのテクニックの背後にある数学を学びます。これらのモデルで異なる入力データに基づいてかぼちゃの価格予測ができるようになります。

[![初心者向け機械学習 - 線形回帰の理解](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "初心者向け機械学習 - 線形回帰の理解")

> 🎥 上の画像をクリックすると、線形回帰の短いビデオ概要が見られます。

> このカリキュラム全体を通じて、数学の知識は最小限に抑え、他分野から来た学生にも理解しやすくすることを目指しています。理解を助けるためにノート、🧮 数学解説、図解、その他の学習ツールに注意してください。

### 前提条件

かぼちゃのデータ構造についてはすでに理解しているはずです。このレッスンの _notebook.ipynb_ ファイル内に、事前に読み込みと整形済みのデータがあります。その中で、かぼちゃの価格が単位「ブッシェル（容量単位）」あたりの価格として新しいデータフレームに表示されています。Visual Studio Codeのカーネル環境でこれらのノートブックが実行できることを確認してください。

### 準備

念のため、データを読み込む目的は「データに質問を投げかける」ことです。

- かぼちゃを買うのに最適な時期はいつ？
- ミニかぼちゃの1箱あたりの価格はどのくらいになる？
- 半ブッシェルのバスケットで買うべきか、それとも1 1/9ブッシェルの箱で買うべきか？
このデータをさらに掘り下げていきましょう。

前のレッスンで、Pandasのデータフレームを作成し、元のデータセットの一部を取り込み、価格をブッシェル単位で標準化しました。ただし、その際約400ポイントのデータで、かぼちゃの季節の月のデータのみを対象にしていました。

今レッスンの添付ノートブックにプリロードされたデータを見てみましょう。データが事前に読み込まれ、月ごとの散布図がプロットされています。もっと詳細にこのデータの性質を知るため、さらにクリーニングをしてみましょう。

## 線形回帰の直線

レッスン1で学んだように、線形回帰のゴールは直線をプロットして、

- **変数の関係を示す**。変数間の関係を示す
- **予測を行う**。その直線に対する新しいデータ点の位置を正確に予測すること

の2つです。

**最小二乗法回帰**では、こうした直線を引くのが典型的です。「最小二乗法」とはモデルの誤差の合計を最小化するプロセスを指します。全てのデータ点について、実際の点と回帰直線の縦方向の距離（残差と呼ばれます）を測ります。

この距離の二乗を取る理由は2つあります：

1. **大きさを重視し符号は無視する**：誤差が-5でも+5でも同じ扱いにするため、すべて正の値にするため。
2. **外れ値へのペナルティ**：二乗することで大きい誤差に重みづけし、離れた点に近づくよう直線を引く。

この二乗誤差を全て足し合わせ、その合計が最も小さくなる直線を求めます。これが「最小二乗」という名前の由来です。

> **🧮 数学を見てみましょう** 
>
> この直線は_最良適合線_とも呼ばれ、[次の式](https://en.wikipedia.org/wiki/Simple_linear_regression)で表されます：  
> 
> ```
> Y = a + bX
> ```

> `X`は“説明変数”、`Y`は“目的変数”です。直線の傾きを`b`、切片を`a`と呼び、`X=0`のときの`Y`の値を指します。
>
>![傾きを計算](../../../../translated_images/ja/slope.f3c9d5910ddbfcf9.webp)
>
> まず傾き`b`を計算します。インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)
>
> 言い換えると、かぼちゃのデータの質問「月ごとのかぼちゃのブッシェル単価を予測する」で考えると、`X`は価格、`Y`は販売月を表します。
>
>![式を完成させる](../../../../translated_images/ja/calculation.a209813050a1ddb1.webp)
>
> `Y`の値を計算します。約4ドルなら、それは4月で間違いありません！ インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)
>
> この式は、`Y`が`X=0`の時どこに位置するか（切片）と、傾きからなる直線の傾斜を計算する数学的な方法です。
>
> これらの計算方法は[Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html)のウェブサイトでも確認できます。また、[この最小二乗計算機](https://www.mathsisfun.com/data/least-squares-calculator.html)で数値の影響を確認してみてください。

## 相関

もう一つ理解すべき用語は、与えられたX変数とY変数間の**相関係数**です。散布図でこの係数を視覚的に簡単に見ることができます。データ点がきれいな直線上に散らばっていると強い相関を持ち、点がばらばらだと弱い相関を示します。

良い線形回帰モデルは、最小二乗法回帰で計算した線形回帰の相関係数が高い（0に近いより1に近い）ものです。

✅ このレッスンに付属のノートブックを実行し、「月」と「価格」の散布図を見てください。かぼちゃの販売では、月と価格の関連は散布図の見た目で「高い」相関でしょうか、それとも「低い」相関でしょうか？　さらに、`Month`を細かくした指標、例えば*年中何日目か*で見るとどうでしょうか？

以下のコードでは、データのクリーニングが済んで、このような`new_pumpkins`というデータフレームがあると仮定します：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> データのクリーニング用コードは[`notebook.ipynb`](notebook.ipynb)にあります。前回のレッスンと同様のクリーニングを行い、`DayOfYear`列は以下の式で計算しています：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

線形回帰の数学的基礎を理解したので、どのパッケージのかぼちゃが一番お得な価格になるかを予測する回帰モデルを作成してみましょう。例えばハロウィンかぼちゃパッチ用にかぼちゃの購入を最適化したい人が参考にできる情報です。

## 相関を探す

[![初心者向け機械学習 - 相関 を探そう：線形回帰のカギ](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "初心者向け機械学習 - 相関 を探そう：線形回帰のカギ")

> 🎥 上の画像をクリックすると、相関の短いビデオ概要が見られます。

前のレッスンで月ごとの平均価格が以下のようだったことを思い出してください：

<img alt="月ごとの平均価格" src="../../../../translated_images/ja/barchart.a833ea9194346d76.webp" width="50%"/>

これを見ると何らかの相関がありそうです。`Month`と`Price`、あるいは`DayOfYear`と`Price`の関係を予測する線形回帰モデルを訓練できます。後者の関係を示す散布図はこちらです：

<img alt="日付（年中の日数）と価格の散布図" src="../../../../translated_images/ja/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

`corr`関数を使って相関を確認すると：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

相関は小さめで、`Month`で-0.15、`DayOfYear`で-0.17のようです。しかし、他に重要な関係性があるようにも見えます。かぼちゃの品種ごとに異なる価格のクラスタができているように見えます。これを確認するため、それぞれの品種を異なる色で散布図にプロットし、`scatter`関数の`ax`パラメータに軸を渡して全ポイントを同一グラフ上に描きます：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="日付と価格の散布図（品種別カラー）" src="../../../../translated_images/ja/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

調査の結果、価格に最も影響を与えるのは販売日ではなく品種である可能性が高いことがわかります。これを棒グラフでも示します：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="品種別価格の棒グラフ" src="../../../../translated_images/ja/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

ここでは一旦「パイタイプ」のかぼちゃのみに注目し、日付が価格に与える影響を見てみましょう：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="日付と価格の散布図（パイタイプのみ）" src="../../../../translated_images/ja/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

`corr`関数で`Price`と`DayOfYear`間の相関を計算すると、およそ `-0.27` となり、予測モデルの訓練が合理的であることがわかります。

> 線形回帰モデルを訓練する前に、データがクリーンであることが大切です。線形回帰は欠損値があるとうまく機能しないため、空のセルは全部除去した方がよいでしょう：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

別の方法は、欠損値を対応する列の平均値で埋める方法です。

## 単純線形回帰

[![初心者向け機械学習 - Scikit-learnを使った線形回帰と多項式回帰](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "初心者向け機械学習 - Scikit-learnを使った線形回帰と多項式回帰")

> 🎥 上の画像をクリックすると、線形回帰と多項式回帰の短いビデオ概要が見られます。

線形回帰モデルを訓練するには、**Scikit-learn** ライブラリを使います。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

入力値（特徴量）と期待される出力（ラベル）を別々のNumPy配列に分けます：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 線形回帰パッケージに正しく認識させるため、入力データに`reshape`を適用したことに注意してください。線形回帰では、2次元配列の入力を期待し、配列の各行が入力特徴量のベクトルを表します。本例では入力が1つだけなので、形状はN×1（Nはデータセットのサイズ）にします。

次に、データを訓練用と検証用に分割して、モデルの検証ができるようにします：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後に、実際に線形回帰モデルの訓練は2行のコードで済みます。まず`LinearRegression`オブジェクトを定義し、`fit`メソッドでデータに当てはめます：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit` メソッド実行後の `LinearRegression` オブジェクトには回帰の全ての係数が含まれており、`.coef_` プロパティを使ってアクセスできます。今回の場合、係数はひとつだけで、約 `-0.017` となるはずです。これは、価格が時間とともに少しずつ（1日あたり約2セントほど）下がる傾向にあることを意味します。回帰線が Y軸と交差する点は `lin_reg.intercept_` でアクセスでき、今回の場合は約 `21` で、年初の価格を示しています。

モデルの精度を確かめるには、テストデータセットで価格を予測し、その予測値が期待値にどれだけ近いかを測ります。これは、平均二乗誤差（MSE）を使って行い、期待値と予測値の差を二乗した値の平均を計算します。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

誤差は約2ポイント、つまり約17%となり、あまり良くありません。モデル品質の別の指標として **決定係数** があり、以下のように取得できます。

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
値が0の場合、モデルは入力データを考慮しておらず、最悪の線形予測子、すなわち単純に結果の平均値を返していることを表します。値が1だと予測が完全に正しいことを示します。今回の決定係数は約0.06で、かなり低い値です。

テストデータと回帰直線を一緒にプロットして、回帰の様子を視覚的に確認することもできます。

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/ja/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式回帰

線形回帰の別のタイプに多項式回帰があります。変数間に線形関係がある場合（例えば、かぼちゃの体積が大きいほど価格が高い）がある一方、関係性が平面や直線で表現できない場合もあります。

✅ こちらに多項式回帰に向く[さらにいくつかの例](https://online.stat.psu.edu/stat501/lesson/9/9.8)があります。

改めて、日付と価格の関係を見てみましょう。この散布図は必ずしも直線で分析すべきでしょうか？価格は変動し得ますよね。この場合、多項式回帰を試すことができます。

✅ 多項式は、1つまたは複数の変数と係数からなる数学的表現です。

多項式回帰は非線形データに合うように曲線を作成します。今回は、入力データに2乗した `DayOfYear` を追加することで、年のある時点で最小値を持つ放物線でデータにフィットできるはずです。

Scikit-learn には、異なるデータ処理ステップを連結するための便利な [パイプライン API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) が含まれています。**パイプライン** は**推定器**の連鎖です。今回は、多項式特徴量を追加してから回帰を学習するパイプラインを作成します。

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)` を使うと、入力データから全ての2次の多項式が含まれます。今回の場合は `DayOfYear`<sup>2</sup> だけですが、X と Y という2つの入力変数がある場合、X<sup>2</sup>、XY、Y<sup>2</sup> が追加されます。必要に応じて、より高次の多項式も使えます。

パイプラインは元の `LinearRegression` オブジェクトと同様に使え、`fit` して `predict` で予測が可能です。以下はテストデータと近似曲線のグラフです。

<img alt="Polynomial regression" src="../../../../translated_images/ja/poly-results.ee587348f0f1f60b.webp" width="50%" />

多項式回帰を使うとMSEがやや下がり、決定係数は上がりますが大きな変化ではありません。より正確な予測には他の特徴も考慮する必要があります！

> 価格が最も低くなるのがハロウィン頃であることがわかります。これをどう説明できますか？

🎃 おめでとうございます、パイ用かぼちゃの価格を予測できるモデルを作成できました。おそらく他のかぼちゃの種類でも同じ手順を繰り返せますが、それは大変です。次はかぼちゃの品種をモデルに取り入れる方法を学びましょう！

## カテゴリ特徴量

理想的には、同じモデルで異なるかぼちゃの品種の価格を予測したいですが、`Variety` 列は `Month` のような数値型の列とは異なり、非数値の値を含みます。こうした列は **カテゴリ特徴量** と呼ばれます。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 上の画像をクリックすると、カテゴリ特徴量を使う簡単なビデオ説明が見られます。

ここでは品種ごとの平均価格の違いがわかります。

<img alt="Average price by variety" src="../../../../translated_images/ja/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

品種を考慮するためには、まず数値形式に変換、つまり **エンコード** が必要です。エンコードにはいくつか方法があります。

* 単純な **数値エンコード** は異なる品種のテーブルを作り、品種名をそのテーブルのインデックスに置き換えます。しかし線形回帰ではこれは良くない方法です。線形回帰はインデックスの数値をそのまま係数と掛け合わせて結果に加えるため、インデックス番号と価格の間の関係が非線形であっても無理に線形として扱ってしまうからです。
* **ワンホットエンコード** は `Variety` 列を各品種ごとに4列に分けます。各列には該当する品種の行だけ `1`、それ以外は `0` が入ります。これにより、線形回帰では各かぼちゃ品種に対して4つの係数があり、それぞれの品種の「ベース価格（あるいは追加分）」を決めることができます。

以下はワンホットエンコードの例です。

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

ワンホットエンコードされた品種を入力にして線形回帰を学習するには、`X` と `y` データを正しく初期化するだけです。

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

残りのコードは先に使った線形回帰の学習と同じです。試してみると平均二乗誤差はほぼ同じですが、決定係数が大幅に向上し約77%になります。さらに正確な予測のためには、今回の例のようにカテゴリー特徴量と `Month` や `DayOfYear` といった数値特徴量の両方を考慮できます。複数の特徴量配列を結合するために `join` を使います。

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

ここではさらに `City` と `Package` の種類も考慮し、MSEは2.84（10%）、決定係数は0.94となります。

## まとめて実装

最も良いモデルを作るため、上記の複合（ワンホットエンコードされたカテゴリ＋数値）データを多項式回帰と組み合わせます。以下は便利のための完全なコードです。

```python
# トレーニングデータを設定する
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# トレイン・テスト分割を行う
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# パイプラインを設定してトレーニングする
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# テストデータの結果を予測する
pred = pipeline.predict(X_test)

# MSEと決定係数を計算する
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

これにより、決定係数は約97%、MSEは2.23（約8%の予測誤差）まで向上します。

| モデル | MSE | 決定係数 |
|-------|-----|---------|
| `DayOfYear` 線形回帰 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式回帰 | 2.73 (17.0%) | 0.08 |
| `Variety` 線形回帰 | 5.24 (19.7%) | 0.77 |
| 全特徴量 線形回帰 | 2.84 (10.5%) | 0.94 |
| 全特徴量 多項式回帰 | 2.23 (8.25%) | 0.97 |

🏆 よくできました！このレッスンで4つの回帰モデルを作成し、モデルの精度を97%まで向上させました。最後の回帰のセクションでは、カテゴリを判定するロジスティック回帰を学びます。

---
## 🚀チャレンジ

このノートブックで異なる変数をいくつか試して、相関関係がモデルの精度にどう影響するか確かめてください。

## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習 & 自習

このレッスンでは線形回帰について学びました。他にも重要な回帰の型があります。部分的な変数選択（ステップワイズ）、リッジ回帰、ラッソ回帰、エラスティックネットなどの技術について読んでみてください。より学ぶのに良いコースは [スタンフォード統計学習コース](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning) です。

## 課題

[モデル作成](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：  
本書類はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を用いて翻訳されました。正確性には努めておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご了承ください。原文の言語による資料を正本としてご参照ください。重要な情報については、専門の人間による翻訳を推奨いたします。本翻訳の利用により生じたいかなる誤解や誤訳についても一切責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->