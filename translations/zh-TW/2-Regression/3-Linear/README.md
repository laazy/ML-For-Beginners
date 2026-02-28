# 使用 Scikit-learn 建立迴歸模型：四種迴歸方式

## 初學者筆記

線性迴歸用於當我們想預測一個**數值**（例如房價、溫度或銷售額）時。
它透過找到一條最佳代表輸入特徵與輸出之間關係的直線來運作。

在本課程中，我們專注於理解基本概念，然後再探討更進階的迴歸技術。
![線性與多項式迴歸資訊圖](../../../../translated_images/zh-TW/linear-polynomial.5523c7cb6576ccab.webp)
> 資訊圖表由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 製作
## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程有 R 版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介紹

迄今為止，你已經探索了什麼是迴歸，並使用本課程將持續使用的南瓜定價資料集的範例數據。你也用 Matplotlib 進行了視覺化。

現在你準備更深入了解機器學習的迴歸。雖然視覺化能幫助你理解資料，機器學習真正的強大之處在於訓練模型。模型在歷史數據上訓練，能自動捕捉數據的依賴關係，並讓你能對新數據（模型未見過的）進行預測。

本課程你將了解兩種迴歸類型：_基本線性迴歸_和_多項式迴歸_，以及這些技術背後的一些數學原理。這些模型將讓我們根據不同輸入數據預測南瓜價格。

[![機器學習初學者－理解線性迴歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "機器學習初學者－理解線性迴歸")

> 🎥 點擊上圖觀看線性迴歸的短片介紹。

> 整個課程假設數學知識最低限度，並試圖讓來自其他領域的學生也能理解，所以請留意筆記、🧮 註解、圖解及其他學習工具，幫助理解。

### 先備知識

你應該已熟悉我們正在檢視的南瓜資料結構。它已預先載入且清理完畢於本課程的 _notebook.ipynb_ 檔案中。在該檔中，南瓜價格以每蒲式耳計算，並顯示於新的資料框中。請確保你可以在 Visual Studio Code 的執行核心中執行這些筆記本。

### 準備

提醒一下，你載入這些資料是為了對它提出問題。

- 什麼時候是買南瓜的最佳時機？
- 我期望一箱小型南瓜的價格是多少？
- 我應該買半蒲式耳的籃子還是一箱 1又1/9蒲式耳的包裝？
讓我們繼續挖掘這些數據。

在上一課中，你建立了 Pandas 資料框並填入原始數據集的一部分，透過蒲式耳將價格標準化。不過這樣你只能取得大約 400 筆資料，且僅限秋季月份。

看看本課程附帶筆記本中預先載入的數據。數據已被載入並繪製了初始的散點圖，展示月份資料。也許我們可以透過更進一步清理，得到關於資料性質的更多細節。

## 線性迴歸線

如同在第 1 課中所學，線性迴歸的目標是繪製一條線來：

- **顯示變數關係**。顯示變數之間的關係
- **進行預測**。準確預測新數據點在該線上的位置。

「最小平方法迴歸」通常用來畫這類線。最小平方法指的是在模型中最小化總誤差的過程。對每個數據點，我們測量實際點與迴歸線之間的垂直距離（稱為殘差）。

我們將距離平方有兩個主要原因：

1. **避免正負抵銷：** 我們希望將 -5 的誤差視同 +5，平方使所有值變成正值。

2. **懲罰離群值：** 平方會給大誤差更大權重，迫使線條更貼近遠離的點。

接著我們會把所有平方值加總起來。目標是找到特定的線，使最終總和達到最小值（可能的最小值），因此稱作「最小平方法」。

> **🧮 答案告訴我數學**  
> 
> 這條線稱為 _最佳擬合線_，可用[方程式](https://en.wikipedia.org/wiki/Simple_linear_regression) 表示：
> 
> ```
> Y = a + bX
> ```
>
> `X` 是「解釋變數」；`Y` 是「應變數」。線的斜率為 `b`，`a` 是 y 截距，表示當 `X = 0` 時，`Y` 的值。
>
>![計算斜率](../../../../translated_images/zh-TW/slope.f3c9d5910ddbfcf9.webp)
>
> 首先計算斜率 `b`。資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 換句話說，將我們南瓜數據的原始問題「按月份預測南瓜每蒲式耳價格」，`X` 代表價格，`Y` 代表銷售月份。
>
>![完成公式](../../../../translated_images/zh-TW/calculation.a209813050a1ddb1.webp)
>
> 計算 Y 的值。如果你付大約 4 美元，一定是 4 月！資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 計算該線的數學必須顯示斜率，斜率也取決於截距，表示 `X = 0` 時 `Y` 在哪裡。
>
> 你可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站觀察計算方法。也可造訪 [此最小平方計算器](https://www.mathsisfun.com/data/least-squares-calculator.html)，觀察數值如何影響線段。

## 相關係數

還有一個要理解的名詞是給定 X 和 Y 變數間的**相關係數**。透過散點圖可以快速將其視覺化。若散點排成一條整齊的線，相關性高；若散點亂散在 X 與 Y 間，相關性低。

好的線性迴歸模型，應該是透過最小平方法計算出迴歸線，且其相關係數數值偏高（靠近 1，而非 0）。

✅ 執行本課程附帶的筆記本，查看「月份對價格」的散點圖。根據你對散點圖的視覺判斷，南瓜銷售的月份與價格關聯性高還是低？若改用更細緻的度量，像是「一年中的第幾天」（例如從年初算起的天數），這個關係會改變嗎？

在下面的程式碼中，我們假設已清理資料，並獲得一個名為 `new_pumpkins` 的資料框，類似如下：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 用於清理資料的程式碼可參考 [`notebook.ipynb`](notebook.ipynb)。我們執行了與前一課相同的清理步驟，並使用以下運算式計算了 `DayOfYear` 欄位：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

既然你已了解線性迴歸背後的數學，讓我們建立一個迴歸模型，看看是否能預測哪種南瓜包裝會有最佳南瓜價格。購買南瓜作為節慶南瓜園的你，可能想利用這些資訊來最佳化購買策略。

## 尋找相關性

[![機器學習初學者－尋找相關性：線性迴歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "機器學習初學者－尋找相關性：線性迴歸的關鍵")

> 🎥 點擊上圖觀看相關性的短片介紹。

從上一課你可能已看到，不同月份的平均價格如下：

<img alt="每月平均價格" src="../../../../translated_images/zh-TW/barchart.a833ea9194346d76.webp" width="50%"/>

這表示應該存在某種相關性，我們可以試著訓練線性迴歸模型，預測 `Month` 與 `Price` 之間，或 `DayOfYear` 與 `Price` 之間的關係。下圖展示了後者的散佈圖：

<img alt="價格 vs. 一年中的日子散點圖" src="../../../../translated_images/zh-TW/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

讓我們使用 `corr` 函數看看是否存在相關：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看起來，按月份的相關性約為 -0.15，按「一年中第幾天」的相關性約為 -0.17，但可能還存在其他重要關係。看起來不同南瓜品種對價格形成了不同的群集。為了驗證這點，我們用不同顏色標示每個南瓜種類，並透過傳遞 `ax` 參數給 `scatter` 繪圖函式，將所有點繪在同一張圖中：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格 vs. 一年中日子散點圖（顏色區分）" src="../../../../translated_images/zh-TW/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

調查顯示，南瓜的品種對價格的影響比實際銷售日期更大。我們可用直條圖視覺化：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="價格 vs. 品種直條圖" src="../../../../translated_images/zh-TW/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

現在先專注於其中一種南瓜品種「派型」，看看日期對價格有什麼影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格 vs. 一年中日子散點圖" src="../../../../translated_images/zh-TW/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

接著用 `corr` 函數計算 `Price` 與 `DayOfYear` 的相關係數，可能會得到約 `-0.27`，這代表訓練預測模型是有意義的。

> 在訓練線性迴歸模型前，務必確保數據清理乾淨。線性迴歸對缺失值不敏感，因此最好去除所有空值：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種選擇是以該欄位的平均值填補空值。

## 簡單線性迴歸

[![機器學習初學者－使用 Scikit-learn 的線性與多項式迴歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "機器學習初學者－使用 Scikit-learn 的線性與多項式迴歸")

> 🎥 點擊上圖觀看線性與多項式迴歸的短片介紹。

我們將使用 **Scikit-learn** 框架來訓練線性迴歸模型。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

我們先將輸入值（特徵）與預期輸出（標籤）分離成不同的 numpy 陣列：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 請注意，我們對輸入資料執行 `reshape`，讓線性迴歸套件能正確理解。線性迴歸期望 2D 陣列作為輸入，陣列的每一列代表一組輸入特徵向量。我們這裡只有一個輸入，因此需要 N×1 形狀的陣列，其中 N 為資料集大小。

接著，我們將數據分割為訓練集與測試集，以便在訓練後驗證模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練真正的線性迴歸模型只要兩行程式碼。我們定義一個 `LinearRegression` 物件，並用 `fit` 方法擬合數據：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 物件在經過 `fit` 訓練後包含了回歸的所有係數，可以透過 `.coef_` 屬性存取。在我們的例子中，只有一個係數，應該大約是 `-0.017`。這代表價格隨時間似乎略微下降，但幅度不大，大約每天跌了兩分錢。我們也可以使用 `lin_reg.intercept_` 取得回歸與 Y 軸的交叉點，這在我們的例子中會是大約 `21`，顯示年初的價格。

為了檢視模型的準確度，我們可以在測試資料集上預測價格，然後衡量預測值與期望值的接近程度。這可以用均方誤差（MSE）來衡量，它是所有預期值與預測值差異平方的平均值。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

我們的誤差大約是 2 點，約為 ~17%。表現並不佳。判斷模型品質的另一個指標是**決定係數**，可用以下方式取得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
若該值為 0，表示模型沒有考慮輸入資料，等同於*最差的線性預測*，也就是輸出結果的平均值。值為 1 意味著我們可以完美地預測所有期望輸出。在我們的例子中，決定係數約為 0.06，相當低。

我們還可以將測試資料與回歸直線繪製在一起，更清楚地了解回歸情況：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-TW/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式回歸

另一種線性回歸是多項式回歸。有時變數之間為線性關係，例如體積越大的南瓜價格越高，但有時這種關係無法用一平面或直線來表示。

✅ 這裡有[更多使用多項式回歸的資料範例](https://online.stat.psu.edu/stat501/lesson/9/9.8)

再看看日期和價格間的關係。這個散佈圖看起來一定適合用直線分析嗎？價格不是會波動嗎？這種情況下，可以嘗試多項式回歸。

✅ 多項式是可能包含一個或多個變數和係數的數學式

多項式回歸會做出曲線，以更適合非線性資料。在我們的例子中，若將平方的 `DayOfYear` 變數加入輸入資料，即可用拋物線去擬合資料，且該曲線會在一年中的某點出現最小值。

Scikit-learn 提供方便的[pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)來串接資料處理的不同步驟。**pipeline** 是一連串的**估計器**。在我們的例子中，我們將建立一個先加入多項式特徵，然後訓練回歸的 pipeline：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 表示會包含所有二次多項式。在我們這裡，只有 `DayOfYear`<sup>2</sup>，但若有兩個輸入變數 X 和 Y，則會加入 X<sup>2</sup>、XY 和 Y<sup>2</sup>。如果想，也可使用更高次數多項式。

Pipeline 使用方法和原本的 `LinearRegression` 物件一樣，我們可以 `fit` pipeline，然後用 `predict` 得到預測結果。下圖是測試資料和擬合曲線：

<img alt="Polynomial regression" src="../../../../translated_images/zh-TW/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多項式回歸，我們可獲得略低的 MSE 和略高的決定係數，但差異不大。還需考慮其他特徵！

> 你可以看到最低的南瓜價格大約出現在萬聖節前後。你怎麼解釋這個現象？

🎃 恭喜你，剛剛建立了一個能幫助預測派用南瓜價格的模型。你或許可以對所有南瓜種類重複這個流程，但那會很繁瑣。現在我們來學習如何讓模型納入南瓜品種差異！

## 類別特徵

理想狀況下，我們希望使用同一個模型預測不同南瓜品種的價格。然而，`Variety` 欄位不同於 `Month` 等欄位，因為它包含非數值資料。這類欄位稱為**類別特徵**。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上方圖片，觀看使用類別特徵的簡短影片介紹。

這裡展示不同品種的平均價格差異：

<img alt="Average price by variety" src="../../../../translated_images/zh-TW/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考慮品種，我們首先需要將它轉換成數值形式，或稱**編碼**。有幾種方法：

* 簡單的**數字編碼**會建立品種列表，然後用該列表的索引取代品種名稱。但對線性回歸而言，這不是好方法，因為線性回歸直接使用編號數值並乘以係數，然而索引與價格間的關係很明顯非線性，即使索引是有序的。
* **One-hot 編碼**會將 `Variety` 欄位拆成四個欄位，每個欄位對應一品種。每一列中對應品種的欄位為 `1`，其餘為 `0`。這樣線性回歸會有四個係數，分別代表各品種的「基礎價格」（或可視為「額外價格」）。

下面程式碼顯示如何以 one-hot 編碼品種：

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

若要使用 one-hot 編碼的品種作為輸入訓練線性回歸，只要正確初始化 `X` 和 `y`：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其他程式碼與之前訓練線性回歸相同。若你嘗試執行，會看到均方誤差差不多，但決定係數大幅提升（約 77%）。想得到更準確的預測，可以加入更多類別特徵以及數值特徵，如 `Month` 或 `DayOfYear`。要獲得一個完整的特徵陣列，可以使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡也納入了 `City` 與 `Package` 類型，得到 MSE 2.84（10%）和決定係數 0.94！

## 綜合應用

為得到最佳模型，我們可將上例中合併的（one-hot 編碼類別 + 數值）資料，搭配多項式回歸。以下是完整程式碼以方便使用：

```python
# 設置訓練資料
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 進行訓練和測試資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設置並訓練流程
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試資料結果
pred = pipeline.predict(X_test)

# 計算均方誤差和判定係數
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

這樣應該可達到最高決定係數接近 97%，MSE=2.23（約 8% 預測誤差）。

| 模型 | MSE | 決定係數 |
|-------|-----|-----------|
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |
| 全特徵 線性 | 2.84 (10.5%) | 0.94 |
| 全特徵 多項式 | 2.23 (8.25%) | 0.97 |

🏆 做得好！你在一堂課中建立了四個回歸模型，並將模型品質提升到 97%。在最後一節「回歸型」中，你將學習用邏輯回歸判斷類別。

---
## 🚀 挑戰

在本筆記本中測試多個不同變數，觀察其相關性與模型準確度的對應關係。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

本課程介紹線性回歸。還有許多重要的回歸型態。可閱讀邁進式回歸、嶺回歸、套索回歸與彈性網路技巧。深入了解建議學習[史丹佛統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作業

[建立一個模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們力求準確，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件之母語版本應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤譯負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->