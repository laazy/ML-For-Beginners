# 使用 Scikit-learn 建立回歸模型：回歸的四種方法

## 初學者筆記

線性回歸用於當我們想要預測一個**數值**（例如，房價、溫度或銷售額）時。  
它通過尋找一條最好地代表輸入特徵與輸出之間關係的直線來工作。

在這堂課中，我們專注於理解概念，然後再探索更高級的回歸技術。
![線性與多項式回歸資訊圖](../../../../translated_images/zh-MO/linear-polynomial.5523c7cb6576ccab.webp)
> 資訊圖由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 製作
## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [這堂課另有 R 語言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介紹

迄今為止，你已經透過我們將在此課程中使用的南瓜價格資料集瞭解了什麼是回歸。你也使用 Matplotlib 做了視覺化。

現在你已準備好深入了解機器學習中的回歸。雖然視覺化讓你能理解資料，但機器學習的真正威力來自於_訓練模型_。模型在歷史資料上訓練以自動捕捉資料依賴關係，並能預測新資料的結果，這些新資料是模型之前未見過的。

在這個課程中，你將學習兩種類型的回歸：_基礎線性回歸_和_多項式回歸_，以及這些技術背後的一些數學。這些模型將允許我們根據不同的輸入資料來預測南瓜價格。

[![機器學習初學者 - 了解線性回歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "機器學習初學者 - 了解線性回歸")

> 🎥 點擊上述圖片觀看線性回歸的短片概述。

> 在整個課程中，我們假設數學知識最低限度，並力求讓來自其他領域的學生易於理解，因此請留意筆記、🧮 提示、圖示及其他學習工具以輔助理解。

### 先備知識

你現在應該熟悉我們正在檢視的南瓜資料結構。課程中的 _notebook.ipynb_ 檔案已預先載入且清理過該資料。該檔案中，南瓜的價格以每蒲式耳計算並展示在新的資料框中。確保你能在 Visual Studio Code 的 kernel 中執行這些 notebook。

### 準備工作

提醒你，載入資料是為了提出問題。

- 什麼時間買南瓜最好？
- 一箱迷你南瓜的價格大概要多少？
- 我該以半蒲式耳的籃子買還是用 1 1/9 蒲式耳的箱子買？
讓我們繼續深挖這些資料。

在上一堂課中，你建立了一個 Pandas 資料框，並填入來自原始資料集的一部分數據，統一以蒲式耳為單位計價。這樣做，只能取得約 400 筆資料，且只針對秋季幾個月。

請看看這堂課隨附 notebook 中預先載入的資料。我們載入資料後，繪製了月份的初始散點圖。或許透過更多清理，我們能更細緻地了解資料的特性。

## 一條線性回歸線

如你在第一課中所學，線性回歸的目標是能夠繪製一條線：

- **顯示變數關係**。展示變數間的關係
- **做出預測**。準確預測新數據點相對於該線會落在哪裡

典型的**最小平方法回歸（Least-Squares Regression）**會畫出這類線。"最小平方法"一詞指的是最小化模型中總誤差的過程。對每個數據點，我們測量該點和回歸線之間的垂直距離（稱為殘差）。

我們平方這些距離有兩個主要原因：

1. **大小超過方向**：我們要將錯誤 -5 和 +5 同等看待，平方可以令所有值變為正數。

2. **懲罰異常值**：平方會給較大誤差更高的權重，迫使回歸線更貼近遠離的點。

接著，我們將所有平方後的值相加。目標是找到使該和最小的那條具體直線，這也是「最小平方法」的名稱由來。

> **🧮 給我看數學！**  
> 
> 這條稱為_最佳擬合線_的線可用[方程式](https://en.wikipedia.org/wiki/Simple_linear_regression)表示：
> 
> ```
> Y = a + bX
> ```
>
> `X`是「解釋變數」，`Y`是「應變數」。線的斜率為`b`，`a`是 y 截距，指的是當`X=0`時，`Y`的值。  
>
>![計算斜率](../../../../translated_images/zh-MO/slope.f3c9d5910ddbfcf9.webp)
>
> 首先計算斜率 `b`。資訊圖由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 換句話說，並回到我們南瓜資料的原始問題：「預測每蒲式耳南瓜價格與月份的關係」，`X` 代表價格，`Y` 代表銷售月份。
>
>![完成方程式](../../../../translated_images/zh-MO/calculation.a209813050a1ddb1.webp)
>
> 計算 Y 的值。如果你付約 4 美元，那一定是 4 月！資訊圖由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 計算該線的數學方法必須展現線的斜率，同時受截距影響，即`X=0`時的`Y`位置。
>
> 你可以參考 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站來觀察這些計算方法。也可訪問[此最小平方法計算器](https://www.mathsisfun.com/data/least-squares-calculator.html)來看數值如何影響線。

## 相關性

還有一個必須了解的詞是給定 X 和 Y 變數間的**相關係數**。利用散點圖，你可以快速視覺化此係數。點散佈成很整齊一條線的圖有高相關，但點散佈在 X 和 Y 間各處的圖則低相關。

良好線性回歸模型會有高（靠近 1 而非 0）的相關係數，使用最小平方回歸法畫出回歸線。

✅ 執行本課程附帶的 notebook，查看「月份對價格」的散點圖。根據你對散點圖的視覺判斷，南瓜銷售中「月份對價格」的數據似乎是高相關還是低相關？換用更細微的度量，比如 *一年的第幾天*（即從年初開始算的天數）情況會改變嗎？

以下程式碼中，我們假設已經清理過資料，得到名為 `new_pumpkins` 的資料框，類似如下：

ID | 月份 | 一年中的第幾天 | 品種 | 城市 | 包裝 | 最低價 | 最高價 | 價格
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | 派型 | BALTIMORE | 1 1/9 蒲式耳紙箱 | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | 派型 | BALTIMORE | 1 1/9 蒲式耳紙箱 | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | 派型 | BALTIMORE | 1 1/9 蒲式耳紙箱 | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | 派型 | BALTIMORE | 1 1/9 蒲式耳紙箱 | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | 派型 | BALTIMORE | 1 1/9 蒲式耳紙箱 | 15.0 | 15.0 | 13.636364

> 清理資料的程式碼可在 [`notebook.ipynb`](notebook.ipynb) 中看到。我們已經執行與先前課程相同的清理步驟，並透過以下表達式計算了 `DayOfYear` 欄：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

現在你已了解線性回歸背後的數學，讓我們建立回歸模型，看看是否能預測哪種南瓜包裝的價格最優惠。想要開設假日南瓜園的人可能會需要這個資訊，來優化他們南瓜包裝的採購。

## 尋找相關性

[![機器學習初學者 - 尋找相關性：線性回歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "機器學習初學者 - 尋找相關性：線性回歸的關鍵")

> 🎥 點擊圖片觀看相關性的短片概述。

從上一堂課，你可能已看到不同月份的平均價格大致如下：

<img alt="各月平均價格" src="../../../../translated_images/zh-MO/barchart.a833ea9194346d76.webp" width="50%"/>

這表明應該存在某些相關性，我們可以嘗試訓練線性回歸模型來預測`月份`與`價格`之間的關聯，或者`一年中的第幾天`與`價格`的關係。以下散點圖顯示後者的關係：

<img alt="價格與一年中天數的散點圖" src="../../../../translated_images/zh-MO/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

我們用 `corr` 函數來看是否存在相關性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看起來以 `月份` 計算的相關性約為 -0.15，`DayOfMonth` 大約是 -0.17，但另有可能存在另一個重要關係。價格似乎依南瓜品種分成不同群集。要確認此假設，我們用不同顏色繪製每個南瓜品種。透過傳遞 `ax` 參數給 `scatter` 繪圖函數，我們可以將所有點畫在同一張圖上：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格與一年中天數的散點圖（按品種著色）" src="../../../../translated_images/zh-MO/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

調查結果顯示品種對整體價格比實際銷售日期影響較大。我們用長條圖也可觀察到：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="不同品種價格長條圖" src="../../../../translated_images/zh-MO/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

暫時只聚焦單一品種——「派型」，看看日期對價格的影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格與一年中天數的散點圖" src="../../../../translated_images/zh-MO/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

現在如果用 `corr` 函數計算 `價格` 與 `DayOfYear` 的相關性，大約會是 `-0.27`——這意味著訓練預測模型是合理的。

> 在訓練線性回歸模型前，重要的是確保資料是乾淨的。線性回歸不適用於存在缺值的情況，因此刪除所有空白欄位是合理做法：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種方法是將空值以該欄的平均值填補。

## 簡單線性回歸

[![機器學習初學者 - 使用 Scikit-learn 的線性與多項式回歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "機器學習初學者 - 使用 Scikit-learn 的線性與多項式回歸")

> 🎥 點擊上面圖片觀看線性與多項式回歸短片介紹。

為了訓練線性回歸模型，我們將使用 **Scikit-learn** 函式庫。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

我們先將輸入數值（特徵）與期望輸出（標籤）分別放入不同的 numpy 陣列：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意，我們必須將輸入資料做 `reshape`，讓線性回歸套件正確理解它。線性回歸預期輸入是一個二維陣列，每行對應一組輸入特徵的向量。由於我們只有一個輸入，因此需要的是形狀為 N×1 的陣列，其中 N 是資料集大小。

接著，我們需要將資料拆分成訓練集及測試集，以便訓練後驗證模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練真正的線性回歸模型只需要兩行程式。先定義 `LinearRegression` 物件，然後利用 `fit` 方法將它擬合至資料：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit` 後的 `LinearRegression` 物件包含所有回歸係數，可以使用 `.coef_` 屬性存取。在我們的例子中，只有一個係數，大約是 `-0.017`。這代表價格似乎隨時間略為下降，但幅度不大，大約每天降2仙。我們也可以使用 `lin_reg.intercept_` 取得回歸與 Y 軸的交點，這在我們的例子中大約是 `21`，表示年初的價格。

為了檢查模型的準確度，我們可以在測試資料集上預測價格，然後測量預測結果與期望值的接近程度。這可以使用均方誤差（MSE）指標完成，即期望值與預測值之間所有平方差的平均值。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

我們的誤差似乎約為 2 點，約 17%。不算太好。模型品質的另一個指標是**決定係數**，可以這樣取得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
若值為 0，表示模型完全不考慮輸入資料，並且充當*最差線性預測器*，即預測為結果的平均值。值為 1 表示我們能完全完美地預測所有期望輸出。在我們的例子中，決定係數約為 0.06，相當低。

我們也可以將測試資料與回歸線同時繪圖，更直觀地看回歸的表現：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-MO/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式回歸

另一種線性回歸是多項式回歸。有時，變數間存在線性關係—體積較大的南瓜價格較高—但有時這些關係無法用平面或直線描述。

✅ 這裡有 [更多範例](https://online.stat.psu.edu/stat501/lesson/9/9.8) 適合用多項式回歸的資料

再看看日期與價格的關係。這散點圖似乎一定要用直線分析嗎？價格不會波動嗎？在這種情況下，可以嘗試多項式回歸。

✅ 多項式是包含一個或多個變量與係數的數學表達式

多項式回歸會建立曲線以更好擬合非線性資料。在我們的例子中，若將平方的 `DayOfYear` 變項加入輸入資料，我們應能用拋物線擬合資料，曲線會在年內某一點有極小值。

Scikit-learn 包含便利的 [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)，用來串接資料處理步驟。**pipeline** 是一連串的**估計器**。在我們例子中，我們將建立一條 pipeline，先加入多項式特徵，再訓練回歸：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 意味著我們將包含輸入資料的所有二階多項式。在我們這裡，只有 `DayOfYear`<sup>2</sup>，但若輸入有兩變量 X 及 Y，則會加上 X<sup>2</sup>、XY 及 Y<sup>2</sup>。當然，我們也可以使用更高階的多項式。

Pipeline 可以像原本的 `LinearRegression` 物件一樣使用，例如我們可以 `fit` pipeline，再用 `predict` 取得預測結果。下圖顯示測試資料與擬合曲線：

<img alt="Polynomial regression" src="../../../../translated_images/zh-MO/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多項式回歸，我們可以獲得略低的 MSE 與較高的決定係數，但差異不大。我們還需要考慮其他特徵！

> 你可以看到最低的南瓜價格似乎出現在萬聖節前後。你怎麼解釋這現象？

🎃 恭喜，剛剛你建立了一個能幫助預測派南瓜價格的模型。或許你可以對其他所有南瓜類型重複此程序，但那會很繁瑣。現在讓我們了解如何將南瓜品種納入模型！

## 類別特徵

在理想狀況下，我們希望使用同一模型預測不同南瓜品種的價格。不過，`Variety` 欄與 `Month` 等欄不同，因它包含非數字值。這類欄稱為**類別特徵**。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上圖觀看使用類別特徵的短片介紹。

這裡展示了平均價格與品種的關係：

<img alt="Average price by variety" src="../../../../translated_images/zh-MO/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考慮品種，我們首先需要將它轉成數值型態，或稱**編碼**。我們可以用幾種方式做到這點：

* 簡單的**數字編碼**會建立一個品種清單，然後以該清單中索引取代品種名稱。這對線性回歸不太適合，因為線性回歸會使用索引的數值，乘以某係數加入結果之中。在我們的例子，索引與價格的關係明顯非線性，即使我們保證索引有特定排序。
* **一熱編碼**會將 `Variety` 欄拆成四個欄，每個品種一欄。每欄對應列若屬於該品種則為 `1`，否則為 `0`。這代表線性回歸會有四個係數，分別對應四個南瓜品種的「起始價格」（或更準確說是「額外價格」）。

以下程式碼示範如何對品種進行一熱編碼：

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

要用一熱編碼品種作為輸入訓練線性回歸，只要正確初始化 `X` 與 `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其餘程式碼與之前用來訓練線性回歸的相同。實驗結果會顯示均方誤差約相當，但決定係數大幅提升（約 77%）。若想更精確預測，可加入更多類別特徵，或數值特徵，如 `Month` 和 `DayOfYear`。可以用 `join` 合併成一個特徵陣列：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡我們還考慮了 `City` 和 `Package` 類型，使 MSE 降為 2.84（10%），決定係數升至 0.94！

## 綜合應用

為了打造最佳模型，我們可以使用上述範例中合併的（類別一熱編碼 + 數值）資料，搭配多項式回歸。以下為完整程式碼方便使用：

```python
# 設置訓練數據
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 進行訓練-測試拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設置並訓練流程
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試數據結果
pred = pipeline.predict(X_test)

# 計算均方誤差及決定係數
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

預期將得到近 97% 的最佳決定係數，以及 MSE=2.23（約8%預測誤差）。

| 模型 | MSE | 決定係數 |
|-------|-----|---------|
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |
| 所有特徵 線性 | 2.84 (10.5%) | 0.94 |
| 所有特徵 多項式 | 2.23 (8.25%) | 0.97 |

🏆 做得好！你在一課中建立了四個回歸模型，將模型品質提升至 97%。回歸章節最後會介紹用於分類的邏輯回歸。

---
## 🚀挑戰

在這個筆記本中測試不同變數，以觀察相關程度如何影響模型準確度。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

本課介紹線性回歸。還有其他重要的回歸類型，請閱讀逐步回歸、Ridge、Lasso 與 Elasticnet 技術。推薦的深入課程是 [史丹佛統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)。

## 作業

[建構模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於確保準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原文文件以其母語版本為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或誤釋負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->