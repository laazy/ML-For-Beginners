# 使用 Scikit-learn 建立回歸模型：回歸的四種方式

## 初學者注意

線性回歸用於我們想要預測**數值**（例如房價、溫度或銷售額）時。  
它的原理是找到一條最能代表輸入特徵與輸出關係的直線。

在本課程中，我們著重於理解概念，之後將探索更進階的回歸技術。  
![線性回歸與多項式回歸資訊圖](../../../../translated_images/zh-HK/linear-polynomial.5523c7cb6576ccab.webp)  
> 資訊圖由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 製作
## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程亦以 R 語言提供！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介紹

到目前為止，你已經透過使用本課程中將持續使用的南瓜定價資料集，探索了回歸是什麼。你也使用 Matplotlib 將資料視覺化。

現在你已準備好更深入探討機器學習中的回歸。視覺化讓你了解資料的意義，但機器學習的真正威力來自於_訓練模型_。模型會根據歷史資料自動捕捉資料的依賴關係，並允許你對從未見過的新資料進行預測。

本課程將介紹兩種回歸模型：_基本線性回歸_及_多項式回歸_，以及部分支撐這些技術的數學原理。這些模型將幫助我們根據不同的輸入資料預測南瓜價格。

[![ML 新手指南 - 理解線性回歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML 新手指南 - 理解線性回歸")

> 🎥 點擊上方圖片觀看線性回歸的短片介紹。

> 在整個課程中，我們假設學員的數學知識基礎不多，致力於讓不同背景的學生都能理解，因此會有筆記、🧮 計算輔助說明、圖示與其他教學工具，幫助理解。

### 先決條件

到現在你應該已熟悉我們目前分析的南瓜資料結構。你可以在本課程附帶的_notebook.ipynb_ 檔案中找到已預載且預處理過的資料。此檔案中，每蒲式耳南瓜價格會顯示在一個新的資料框內。請確保你能在 Visual Studio Code 的 Kernel 中執行這些筆記本。

### 準備

提醒你，我們讀取這些資料是為了提出問題。

- 什麼時候是買南瓜的最好時機？
- 我可以預期迷你南瓜一箱的價格是多少？
- 我該買半蒲式耳籃裝還是一箱 1 1/9蒲式耳盒裝？
我們繼續深入挖掘這些資料。

在前一堂課中，你建立了一個 Pandas 資料框，並以部分原始資料填充，價格以蒲式耳標準化。不過，這樣只能取得約 400 筆資料，而且只包含秋季的數據。

看看本課程附帶筆記本中預先載入的資料。資料已預載，且初步散點圖顯示每個月的資料。也許藉由更深入清理，我們能對資料特性得到更細緻的了解。

## 線性回歸線

如你在第一課學到，線性回歸的目標是繪製一條線用以：

- **顯示變數間關係**。展現變數之間的關聯性  
- **做出預測**。準確預測新資料點在該線上的位置

**最小平方法回歸**常用於繪製這類線。名詞「最小平方法」指的是我們試圖使模型的總誤差最小化的過程。對每筆資料，我們測量實際點與回歸線之間的垂直距離（稱為殘差）。

我們平方這些距離主要有兩個原因：

1. **忽略方向，只看大小**：想要將 -5 與 +5 的誤差視為相同，平方會讓所有數值變正數。

2. **懲罰異常值**：平方使較大的誤差得到加重，促使回歸線更貼近那些遠離的點。

最後，我們將所有平方誤差加總，我們的目標是尋找一條令這個和最小（最低值）的直線，因此稱為「最小平方法」。

> **🧮 數學說明**  
>  
> 這條稱為_最佳擬合直線_的線可用[公式](https://en.wikipedia.org/wiki/Simple_linear_regression)表示：  
> 
> ```
> Y = a + bX
> ```
>
> `X` 是「解釋變數」，`Y` 是「依賴變數」。線的斜率為 `b`，`a` 是 y 截距，即當 `X=0` 時 `Y` 的值。  
>
>![計算斜率](../../../../translated_images/zh-HK/slope.f3c9d5910ddbfcf9.webp)
>
> 首先計算斜率 `b`。資訊圖由 [Jen Looper](https://twitter.com/jenlooper) 製作。  
>
> 換句話說，對應我們南瓜資料的原始問題：「預測某月每蒲式耳的南瓜價格」，`X` 指的是價格，`Y` 是銷售月份。  
>
>![完成公式](../../../../translated_images/zh-HK/calculation.a209813050a1ddb1.webp)
>
> 計算 Y 值。如果你付約 4 美元，肯定是四月！資訊圖由 [Jen Looper](https://twitter.com/jenlooper) 製作。  
>
> 該方程式需展示斜率，斜率也依賴截距，或說 `X=0` 時 `Y` 所在的位置。
> 
> 你可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站看到計算方法，也可造訪 [此最小平方法計算機](https://www.mathsisfun.com/data/least-squares-calculator.html) 觀察數值如何影響直線。

## 相關性

另一個要理解的名詞是給定 X 和 Y 變數間的**相關係數**。透過散點圖，你可以快速看出相關係數的大小。若點畫出一條整齊的線，則相關性高；若散布在 X 與 Y 之間散亂，相關性則低。

理想的線性回歸模型是那些用最小平方法計算時，相關係數接近 1（遠大於 0）且有回歸線的模型。

✅ 執行本課程附帶的筆記本，觀察「月份對價格」散點圖。依你觀察散點圖的判斷，南瓜銷售月份與價格之間的資料是否顯示高度或低度相關？使用更細緻測量（例如年月日中第幾天）代替「月份」會改變結果嗎？

下面的程式碼假設我們已清理資料，並獲得一個名為 `new_pumpkins` 的資料框，如下：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price  
---|-------|-----------|---------|------|---------|-----------|------------|-------  
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  

> 清理資料的程式碼可參閱 [`notebook.ipynb`](notebook.ipynb) 檔案。我們已同前一課執行相同的清理步驟，並用以下表達式計算出 `DayOfYear` 欄位：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

了解線性回歸背後的數學原理後，我們來建立一個回歸模型，看看是否能預測哪種南瓜包裝能獲得最佳價格。假如有人想買南瓜做節慶南瓜園，這資訊有助於他們優化採購計畫。

## 尋找相關性

[![ML 新手指南 - 尋找相關性：線性回歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML 新手指南 - 尋找相關性：線性回歸的關鍵")

> 🎥 點擊上方圖片觀看關於相關性的短片介紹。

從前一課你可能看過，不同月份的平均價格長這樣：

<img alt="各月份平均價格" src="../../../../translated_images/zh-HK/barchart.a833ea9194346d76.webp" width="50%"/>

這暗示出可能存在某種相關性，我們可以嘗試訓練線性回歸模型，預測 `Month` 與 `Price` 之間，或 `DayOfYear` 與 `Price` 之間的關係。以下為後者的散點圖：

<img alt="價格 vs 年中天數 散點圖" src="../../../../translated_images/zh-HK/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />

我們用 `corr` 函數看看是否存在相關性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看來相關性很小，對`Month`是 -0.15，對 `DayOfMonth` 為 -0.17，但可能存在另一重要關係。不同南瓜品種對應於不同價格群組。為證實此假設，讓我們用不同顏色畫出不同南瓜類別。將 `ax` 參數傳給 `scatter` 函數，可將所有點畫於同一張圖：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格 vs 年中天數 散點圖（多色）" src="../../../../translated_images/zh-HK/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

調查顯示品種對總價格的影響比實際銷售日期大，我們可以用條形圖觀察：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="價格 vs 品種條形圖" src="../../../../translated_images/zh-HK/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

目前暫時聚焦於單一品種「派用型」，看看日期對價格有何影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格 vs 年中天數 散點圖" src="../../../../translated_images/zh-HK/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

若用 `corr` 函數計算 `Price` 和 `DayOfYear` 的相關係數，約為 `-0.27`，這表示訓練預測模型是有意義的。

> 訓練線性回歸模型前，務必確保資料已清理乾淨。線性回歸對遺漏值不適用，故應刪除所有空白單元格：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種作法是用對應欄位的平均值填補空白。

## 簡單線性回歸

[![ML 新手指南 - 使用 Scikit-learn 的線性與多項式回歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML 新手指南 - 使用 Scikit-learn 的線性與多項式回歸")

> 🎥 點擊上方圖片觀看線性與多項式回歸的短片介紹。

要訓練線性回歸模型，我們將使用**Scikit-learn**函式庫。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

我們先把輸入值（特徵）與預期輸出（標籤）分別存入不同的 numpy 陣列中：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意，我們必須對輸入資料進行 `reshape`，才能讓線性回歸函式庫正確認識它。線性回歸期望輸入為二維陣列，每一列為一組特徵向量。由於我們只有一個輸入特徵，需將陣列調整為 Nx1 的形狀，N 是資料集大小。

接著，我們需將資料切成訓練和測試集，以便於訓練後驗證模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練線性回歸模型只需兩行程式碼。我們先定義 `LinearRegression` 物件，再用 `fit` 方法套用到資料上：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit` 後的 `LinearRegression` 物件包含所有回歸的係數，可以使用 `.coef_` 屬性訪問。在我們的例子中，只有一個係數，大約為 `-0.017`。這意味著價格隨時間略微下降，但不多，大約每天下降兩仙。我們也可以使用 `lin_reg.intercept_` 訪問回歸與 Y 軸的交點 — 在我們的例子中約為 `21`，表示年初的價格。

為了查看我們模型的準確度，我們可以在測試數據集上預測價格，然後測量預測值與期望值的接近程度。這可以使用均方誤差（MSE）指標進行，該指標是所有期望與預測值平方差的平均。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

我們的誤差約為 2 點，約為 17%。不太好。另一個模型質量指標是**決定係數**，可以這樣獲取：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
若值為 0，表示模型沒考慮輸入資料，充當*最差線性預測器*，即輸出結果的均值。值為 1 表示我們能完全準確地預測所有期望輸出。在我們的例子中，係數約為 0.06，較低。

我們也可以繪製測試數據與回歸線一起，更好地看回歸在我們案例中的效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-HK/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式回歸

另一種線性回歸類型是多項式回歸。雖然有時候變數間有線性關係——例如南瓜體積越大，價格越高——有時這些關係不能用平面或直線表示。

✅ 這裡有[更多例子](https://online.stat.psu.edu/stat501/lesson/9/9.8)展示可能使用多項式回歸的數據

再看看日期和價格的關係。這個散點圖是否必須用直線分析？價格不可能會波動嗎？在這種情況下，你可以試試多項式回歸。

✅ 多項式是可能由一個或多個變數及係數組成的數學表達式

多項式回歸會創建曲線，更好地擬合非線性資料。在我們的例子中，如果將平方的 `DayOfYear` 變數加入輸入資料，我們應該能用一條拋物線來擬合數據，該曲線在年底某點有最低點。

Scikit-learn 包含一個有用的[pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)，用來組合不同的數據處理步驟。**pipeline** 是一連串的 **estimators**。在我們的例子中，我們將創建一個 pipeline，先加入多項式特徵，然後訓練回歸：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 表示我們會包含所有二次多項式。在我們的例子中，這只是 `DayOfYear`<sup>2</sup>，不過給定兩個變數 X 和 Y，會加入 X<sup>2</sup>、XY 和 Y<sup>2</sup>。當然也可以使用更高次多項式。

pipeline 可像原始的 `LinearRegression` 物件一樣使用，即可以 `fit` pipeline，然後用 `predict` 取得預測結果。下面的圖顯示測試數據與擬合曲線：

<img alt="Polynomial regression" src="../../../../translated_images/zh-HK/poly-results.ee587348f0f1f60b.webp" width="50%" />

利用多項式回歸，我們可以得到稍低的 MSE 和較高的決定係數，但差距不大。我們還需考慮其他特徵！

> 你可以看到南瓜價格的最低點出現在萬聖節前後。你怎麼解釋這現象？

🎃 恭喜！你剛剛創建了一個可以幫助預測派用南瓜價格的模型。你大概也可以用相同方法，對其他南瓜品種做預測，但那會很繁瑣。現在，我們來學習如何讓模型考慮南瓜品種！

## 類別特徵

理想狀況下，我們希望用同一模型預測不同南瓜品種的價格。但 `Variety` 欄位有別於 `Month` 之類的欄位，因為它包含非數值。這類欄位稱為**類別特徵**。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上圖觀看短視頻概述類別特徵的使用。

以下展示價格與品種的平均關係：

<img alt="Average price by variety" src="../../../../translated_images/zh-HK/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考慮品種，我們首先需要將它轉為數字形式，或稱**編碼**。有幾種方法：

* 簡單的**數字編碼**會建立品種表，然後用該表索引替代品種名稱。但這不適合線性回歸，因為線性回歸會將編碼數值直接乘以某係數加入結果中，若索引與價格間明顯非線性，即使你特定排序索引，也不合適。
* **獨熱編碼（One-hot encoding）**會把 `Variety` 欄位換成 4 個不同欄位，每個品種一欄。如果該行為該品種，該欄為 1，否則為 0。這意味著線性回歸將有四個係數，分別對應每種南瓜品種，代表該品種的「起始價格」（或更確切說是「額外價格」）。

下面程式碼展示如何使用獨熱編碼：

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

要用獨熱編碼的品種作為輸入來訓練線性回歸，只要正確初始化 `X` 和 `y`：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其餘程式碼與之前訓練線性回歸的相同。若你試試看，會發現均方誤差差不多，但決定係數大幅提高（約 77%）。想要更準確預測，可以考慮更多類別特徵，也加入數字特徵，如 `Month` 或 `DayOfYear`。要組合成一個大特徵陣列，可以使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡也考慮了 `City` 和 `Package` 類型，使 MSE 降至 2.84（10%），決定係數達 0.94！

## 結合所有

要打造最佳模型，可以用上述例子中組合（獨熱編碼類別 + 數字）資料搭配多項式回歸。這是完整程式碼供你方便使用：

```python
# 設置訓練數據
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 進行訓練和測試劃分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設置並訓練流程
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試數據結果
pred = pipeline.predict(X_test)

# 計算均方誤差和決定係數
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

這將給我們最佳決定係數接近 97%，MSE=2.23（約 8% 預測誤差）。

| 模型 | MSE | 決定係數 |
|-------|-----|---------------|
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |
| 所有特徵 線性 | 2.84 (10.5%) | 0.94 |
| 所有特徵 多項式 | 2.23 (8.25%) | 0.97 |

🏆 幹得好！你在一課中創建了四個回歸模型，並將模型質量提升至97%。在回歸的最後一章，你將學習邏輯回歸來判斷分類。

---
## 🚀挑戰

在這個筆記本中測試幾個不同變數，看看相關性如何對模型準確度產生影響。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

本課我們學習了線性回歸。還有其他重要的回歸類型。請閱讀逐步回歸、嶺回歸、套索回歸和彈性網回歸技術。有個優秀課程可深入學習：[史丹佛統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作業

[建立模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件為使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於確保準確性，但請注意自動翻譯可能包含錯誤或不準確之處。文件的原文版本應被視為權威來源。對於重要資訊，建議使用專業人工翻譯。我們不會對因使用本翻譯所引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->