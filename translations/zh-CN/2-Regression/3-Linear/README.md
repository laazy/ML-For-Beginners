# 使用 Scikit-learn 构建回归模型：四种回归方法

## 初学者提示

线性回归用于我们想预测**数值型**的场景（例如房价、温度、销售额）。  
它通过找到一条最能代表输入特征和输出关系的直线来工作。

本课侧重于理解概念，之后再探讨更高级的回归技术。  
![线性回归与多项式回归信息图](../../../../translated_images/zh-CN/linear-polynomial.5523c7cb6576ccab.webp)  
> 信息图作者：[Dasani Madipalli](https://twitter.com/dasani_decoded)

## [课前测试](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课也提供 R 语言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)

### 介绍

到目前为止，你已经通过南瓜定价数据集探讨了回归是什么。该数据集将在本课中持续使用。你还通过 Matplotlib 可视化了数据。

现在你准备深入了解机器学习中的回归。虽然可视化有助于理解数据，但机器学习的真正力量在于_训练模型_。模型基于历史数据训练，自动捕捉数据之间的依赖关系，从而能够预测模型未见过的新数据结果。

本课将介绍两种回归类型：_基础线性回归_ 和 _多项式回归_，以及这些技术背后的部分数学原理。利用这些模型，我们可以根据不同输入数据预测南瓜价格。

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 点击上方图片观看线性回归简介短视频。

> 在整个课程中，我们假设数学基础较少，并努力让来自其它领域的学生也能理解，所以请注意文中的提示、🧮 计算说明、图示及其它辅助学习工具。

### 先决条件

你现在应该熟悉我们正在检查的南瓜数据结构。该数据已预加载并预先清理，可在本课的 _notebook.ipynb_ 文件中找到。文件中，南瓜价格以每蒲式耳计算并显示在一个新的数据框架中。请确保能在 Visual Studio Code 的内核中运行这些笔记本文件。

### 准备工作

提醒一下，你加载这些数据是为了对它提出问题。  

- 什么时候买南瓜最合适？  
- 一箱迷你南瓜的价格会是多少？  
- 我应该买半蒲式耳篮装，还是 1又1/9蒲式耳盒装？  
让我们继续挖掘这些数据。

在上节课，你创建了一个 Pandas 数据框，并从原始数据集中提取了一部分数据，按蒲式耳标准化了价格。这样做只能获取约400个数据点，且只涉及秋季月份。

看看本课随附笔记本中预加载的数据。数据已预加载，并绘制了初步的散点图以显示月份数据。也许通过进一步清理，我们可以更详细地了解数据的性质。

## 线性回归线

正如你在第一课中学到的一样，线性回归的目标是绘制一条线，用来：

- **显示变量关系**。展示变量间的关系  
- **做出预测**。准确预测新数据点相对于该线的位置  

通常通过**最小二乘回归**来绘制这类线。术语“最小二乘”指的是最小化模型的总误差的过程。对于每个数据点，我们测量实际点到回归线的垂直距离（称为残差）。

我们对这些距离做平方处理，主要出于两个原因：

1. **大小而非方向**：我们希望将 -5 的误差与 +5 的误差视为相同。平方使所有值均为正数。  

2. **惩罚异常值**：平方使得较大的误差权重更大，促使回归线更贴近远离的点。  

然后我们将这些平方值加总。目标是找到这条最终平方和最小的线——这就是“最小二乘”的由来。

> **🧮 给我看数学过程**  
>  
> 这条被称作_最佳拟合直线_的线可以用[以下方程表示](https://en.wikipedia.org/wiki/Simple_linear_regression)：  
>  
> ```
> Y = a + bX
> ```
>  
> 其中 `X` 是“解释变量”，`Y` 是“因变量”。斜率是 `b`，`a` 是 y 轴截距，表示当 `X = 0` 时 `Y` 的值。  
>  
>![计算斜率](../../../../translated_images/zh-CN/slope.f3c9d5910ddbfcf9.webp)  
>  
> 首先计算斜率 `b`。信息图作者：[Jen Looper](https://twitter.com/jenlooper)  
>  
> 换句话说，结合我们的南瓜数据原始问题：“按月份预测每蒲式耳的南瓜价格”，`X` 是价格，`Y` 是销售月份。  
>  
>![完成方程](../../../../translated_images/zh-CN/calculation.a209813050a1ddb1.webp)  
>  
> 计算 `Y` 的值。如果你支付大约 4 美元，那应该是四月！信息图作者：[Jen Looper](https://twitter.com/jenlooper)  
>  
> 计算该直线的数学过程必须呈现斜率，同时也依赖于截距，即当 `X = 0` 时 `Y` 的位置。  
>  
> 你可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 网站查看此计算方法。也可访问[最小二乘计算器](https://www.mathsisfun.com/data/least-squares-calculator.html)，观察数值变化如何影响回归线。

## 相关性

还有一个术语需要理解：给定 X 和 Y 变量间的**相关系数**。借助散点图，你可以快速直观地看出这个系数。数据点沿一条整齐线分布的散点图显示高相关性，数据点杂乱无序分布的则显示低相关性。

一个好的线性回归模型通常会在线性回归方程下，利用最小二乘法，具有较高（接近 1 远离 0）的相关系数。

✅ 运行本课配套的笔记本，观察“月份”与“价格”的散点图。根据你对散点图的视觉判断，“月份”与南瓜销售价格的关联度是高还是低？如果用更细粒度的度量替代`Month`，例如“一年中的天数”（即从年初开始的天数数量），结果会有变化吗？

下面的代码中，我们假设已清理数据，并获得一个叫做 `new_pumpkins` 的数据框，如下示例：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理数据的代码可在 [`notebook.ipynb`](notebook.ipynb) 找到。我们按照上一课进行了相同的数据清理步骤，并使用以下表达式计算了 `DayOfYear` 列：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
现在你已经理解了线性回归背后的数学，我们来创建一个回归模型，看看是否能够预测哪种南瓜包装有最优价格。购买者想为假期制作南瓜田，这些信息可以帮助他们优化购买南瓜包装方案。

## 寻找相关性

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 点击上方图片观看相关性介绍短视频。

从上一课你大概看到了各月平均价格大致如下：

<img alt="按月平均价格" src="../../../../translated_images/zh-CN/barchart.a833ea9194346d76.webp" width="50%"/>

这说明可能存在一定程度的相关性，我们可以尝试训练线性回归模型来预测`Month`与`Price`之间，或者`DayOfYear`与`Price`之间的关系。下图展示了后者的散点图：

<img alt="价格 vs. 一年中的天数 散点图" src="../../../../translated_images/zh-CN/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />

用 `corr` 函数检查相关性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
相关性看起来很小，`Month` 为 -0.15，`DayOfYear` 为 -0.17，但可能存在其它重要的关系。看起来价格群聚对应不同南瓜品种。为了验证这个假设，我们用不同颜色为每个品种绘制散点图。通过向 `scatter` 绘图函数传递 `ax` 参数，所有点可绘制在同一张图上：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="价格 vs. 一年中的天数 散点图（彩色）" src="../../../../translated_images/zh-CN/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />

调查显示品种对总价影响大于实际销售日期。条形图如下：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="按品种分类的价格条形图" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

暂时只关注南瓜品种 'pie type'，观察日期对价格的影响：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
  
<img alt="价格 vs. 一年中的天数 散点图（派南瓜）" src="../../../../translated_images/zh-CN/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />

如果用 `corr` 函数计算 `Price` 与 `DayOfYear` 的相关系数，大约是 `-0.27` ——这意味着训练预测模型是合理的。

> 在训练线性回归模型前，确保数据干净非常重要。线性回归对缺失值表现不好，因此建议删除所有空单元格：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
另一种方法是用对应列的均值填充空值。

## 简单线性回归

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 点击上方图片观看线性回归与多项式回归介绍短视频。

训练线性回归模型时，我们将使用**Scikit-learn**库。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
首先把输入值（特征）和预期输出（标签）分成两个 numpy 数组：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> 注意，我们必须对输入数据使用 `reshape`，让线性回归模块能正确理解输入。线性回归期望输入是一个二维数组，数组的每一行是一个特征向量。我们这里只有一个输入，所以需要一个形状为 N×1 的数组，其中 N 是数据集大小。

接着需要将数据拆分成训练集和测试集，以便训练后验证模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
最后，训练线性回归模型仅需两行代码。定义一个 `LinearRegression` 对象，用 `fit` 方法使其拟合数据：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 对象在 `fit` 之后包含了所有回归系数，可以通过 `.coef_` 属性访问。在我们的例子中，只有一个系数，约为 `-0.017`。这意味着价格似乎随着时间略微下降，但幅度不大，大约每天下降2美分。我们还可以使用 `lin_reg.intercept_` 访问回归与 Y 轴的交点 —— 对我们而言，它大约是 `21`，表示年初时的价格。

为了查看模型的准确度，我们可以对测试数据集进行价格预测，然后测量预测结果与真实值的接近程度。这可以通过均方误差（MSE）指标完成，即所有预测值与期望值差的平方的平均值。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
  
我们的误差大约是2点，约为17%。并不算好。模型质量的另一个指标是**决定系数**，可通过如下方式获得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
  
如果该值为0，意味着模型不考虑输入数据表现，仅作为*最差的线性预测器*，即预测结果为结果的均值。值为1意味着我们可以完美预测所有期望输出。我们这里的决定系数约为0.06，非常低。

我们还可以将测试数据和回归直线一起绘制，以更好地观察回归效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```
  
<img alt="Linear regression" src="../../../../translated_images/zh-CN/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多项式回归

另一种线性回归形式是多项式回归。虽然变量之间有时呈线性关系——比如南瓜体积越大，价格越高——但有时这些关系不能用平面或直线表示。

✅ 这里有[一些更多示例](https://online.stat.psu.edu/stat501/lesson/9/9.8)适合使用多项式回归的数据。

再看看日期和价格间的关系。这个散点图一定要用直线分析吗？价格难道不会波动？在这种情况下，可以尝试多项式回归。

✅ 多项式是一种可能包含一个或多个变量与系数的数学表达式。

多项式回归创建曲线，更好地拟合非线性数据。在本例中，如果我们加入平方的 `DayOfYear` 变量作为输入，应该能够用抛物线拟合数据，该曲线将在年内某一点达到最小值。

Scikit-learn 提供了一个有用的 [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) 来组合不同的数据处理步骤。**管道**是一个**估计器**链。在这里，我们先为模型添加多项式特征，然后训练回归：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```
  
使用 `PolynomialFeatures(2)` 意味着将包含所有输入数据的二阶多项式。在我们的例子中，这仅表示 `DayOfYear`<sup>2</sup>，但若有两个输入变量 X 和 Y，则会增加 X<sup>2</sup>、XY 和 Y<sup>2</sup>。如果需要，也可以使用更高次的多项式。

管道可以像原始 `LinearRegression` 对象一样使用，即先 `fit` 管道，再用 `predict` 获取预测结果。下面的图显示了测试数据及其拟合曲线：

<img alt="Polynomial regression" src="../../../../translated_images/zh-CN/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多项式回归，我们可以获得稍低的 MSE 和更高的决定系数，但提升有限。我们需要考虑其他特征！

> 你可以看到最低的南瓜价格大约出现在万圣节附近。你能解释为什么吗？

🎃 恭喜，你刚刚创建了一个能预测馅饼南瓜价格的模型。你或许也能用相同方法处理所有南瓜品种，但那会很繁琐。现在让我们学习如何在模型中考虑南瓜品种！

## 类别特征

理想情况下，我们希望用同一个模型预测不同南瓜品种的价格。然而，`Variety` 列与 `Month` 类似的列不同，因为它包含非数字值。这样的列称为**类别特征**。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 点击上图观看关于使用类别变量的简短视频介绍。

这是平均价格如何随品种变化的示意图：

<img alt="Average price by variety" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

为了考虑品种，我们首先需要将其转换为数字形式，即**编码**。有几种方式：

* 简单的**数值编码**会建立一个品种表，然后用该表中的索引替换品种名。这对线性回归不是最佳，因为线性回归会把索引作为数值处理，乘以系数累加到结果中。在我们例子中，索引与价格的关系明显非线性，即使确保索引按特定顺序排列。
* **独热编码**会把 `Variety` 列替换为4个独立的列，每列代表一个品种。对应品种的行该列为 `1`，其他为 `0`。这意味着在线性回归中会有四个系数 ，分别对应每个南瓜品种，代表该品种的“起始价格”（或更准确说是“附加价格”）。

下面代码演示如何对品种进行独热编码：

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

要用独热编码的品种作为输入训练线性回归，只需正确初始化 `X` 和 `y` 数据：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```
  
其余代码与之前训练线性回归的一样。尝试后可以看到均方误差大致相同，但决定系数大幅提升（约77%）。为了获得更准确的预测，我们可以同时考虑更多类别特征和数值特征，如 `Month` 或 `DayOfYear`。使用 `join` 可以得到一个大的特征数组：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```
  
这里我们同时考虑了 `City` 和 `Package` 类型，得到 MSE 为 2.84（10%），决定系数为 0.94！

## 综合应用

为了拟合最优模型，我们可以将上述的组合数据（独热编码类别 + 数值）和多项式回归结合。方便起见，完整代码如下：

```python
# 设置训练数据
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 做训练-测试拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 设置并训练流水线
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 预测测试数据结果
pred = pipeline.predict(X_test)

# 计算均方误差和确定系数
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```
  
这应获得接近97%的最佳决定系数，MSE=2.23（约8%的预测误差）。

| 模型 | MSE | 决定系数 |  
|-------|-----|---------------|  
| `DayOfYear` 线性模型 | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` 多项式模型 | 2.73 (17.0%) | 0.08 |  
| `Variety` 线性模型 | 5.24 (19.7%) | 0.77 |  
| 全特征 线性模型 | 2.84 (10.5%) | 0.94 |  
| 全特征 多项式模型 | 2.23 (8.25%) | 0.97 |  

🏆 做得好！在本节课中你创建了四个回归模型，并将模型质量提升至97%。回归的最后一节你将学习用于确定类别的逻辑回归。

---
## 🚀挑战

在此笔记本中测试多个不同变量，观察相关性如何对应模型准确度。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

本课学习了线性回归。还有其他重要的回归类型。请阅读逐步回归、岭回归、套索回归和弹性网络技术。推荐学习的优秀课程是 [斯坦福统计学习课程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作业

[构建模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：  
本文件使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们力求准确，但请注意自动翻译可能包含错误或不准确之处。原始文档的母语版本应被视为权威来源。对于重要信息，建议采用专业人工翻译。我们不对因使用此翻译而引起的任何误解或误释承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->