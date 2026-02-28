# Build a regression model using Scikit-learn: regression four ways

## Beginner Note

Linear regression dey used wen we want predict **numerical value** (like house price, temperature, or sales).
E dey work by finding straight line way best represent di relationship between input features and di output.

For dis lesson, we go focus on understanding di concept before we explore more advanced regression techniques.
![Linear vs polynomial regression infographic](../../../../translated_images/pcm/linear-polynomial.5523c7cb6576ccab.webp)
> Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Dis lesson dey available for R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduction 

So far, you don explore wetin regression be with sample data wey we collect from di pumpkin pricing dataset wey we go use for dis lesson. You don also visualize am using Matplotlib.

Now you fit dive deeper into regression for ML. While visualization dey help you make sense of data, real power of Machine Learning na from _training models_. Models dey train on historic data to automatically catch data dependencies, and dem allow you predict outcomes for new data wey di model never see before.

For dis lesson, you go learn more about two types of regression: _basic linear regression_ and _polynomial regression_, plus some math way dey behind these techniques. Dem models go allow us predict pumpkin prices based on different input data. 

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Click di image wey dey above for short video overview of linear regression.

> Throughout dis curriculum, we assume say person no get much knowledge for math before, and we dey try make am easy for students wey dey comot from other fields, so watch for notes, 🧮 callouts, diagrams, and other learning tools to help you understand.

### Prerequisite

You suppose sabi di structure of di pumpkin data wey we dey check by now. You fit find am preloaded and pre-cleaned inside dis lesson's _notebook.ipynb_ file. Inside di file, di pumpkin price dey show per bushel for new dataframe. Make sure say you fit run these notebooks inside Visual Studio Code kernels.

### Preparation

As reminder, you dey load dis data make you fit ask questions about am.

- When na di best time to buy pumpkins? 
- Wetin price fit be for one case of miniature pumpkins?
- I suppose buy dem for half-bushel baskets or buy by 1 1/9 bushel box?
Make we continue to look into dis data.

For di previous lesson, you create Pandas dataframe and put small part of di original dataset for am, standardizing di pricing by di bushel. But by doing that, you only fit collect about 400 datapoints and na only for di fall months.

Check di data way we preload for dis lesson notebook. Di data dey preloaded and initial scatterplot dey show month data. Maybe we fit get small better detail about di nature of di data if we clean am more.

## A linear regression line

As you learn for Lesson 1, di aim of linear regression exercise na to fit plot one line to:

- **Show variable relationships**. Show di relationship between variables
- **Make predictions**. Make correct predictions on where new datapoint for fit fall inside relation to dat line.
 
E common for **Least-Squares Regression** to draw dis kain line. Di term "Least-Squares" mean say you want minimize total error for our model. For every data point, we dey measure vertical distance (we call am residual) between di real point and our regression line.

We dey square these distances for two main reasons:  

1. **Magnitude over Direction:** We wan make error of -5 equal to error of +5. Squaring go make all values positive.  

2. **Penalizing Outliers:** Squaring dey put more weight on bigger errors, e go make di line stay closer to points wey far away.  

Then we add all these squared values together. Our goal na to find di line where dis total sum go minimum (smallest value)—na why dem call am "Least-Squares".  

> **🧮 Show me di math** 
> 
> Dis line wey we call _line of best fit_ fit be expressed by [one equation](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` na 'explanatory variable'. `Y` na 'dependent variable'. Di slope of di line na `b` and `a` na di y-intercept, way mean di value of `Y` when `X = 0`. 
>
>![calculate di slope](../../../../translated_images/pcm/slope.f3c9d5910ddbfcf9.webp)
>
> First, calculate di slope `b`. Infographic by [Jen Looper](https://twitter.com/jenlooper)
>
> To talk am another way, and referring to di pumpkin data original question: "predict price of pumpkin per bushel by month", `X` na price and `Y` na month of sale. 
>
>![complete di equation](../../../../translated_images/pcm/calculation.a209813050a1ddb1.webp)
>
> Calculate di value of Y. If you dey pay around $4, e be say na April! Infographic by [Jen Looper](https://twitter.com/jenlooper)
>
> Di math wey calculate di line must show di slope of di line, way still depend on di intercept, or where `Y` dey when `X = 0`.
>
> You fit check di way to calculate dis values on di [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) website. Also visit [dis Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) to see how di numbers value affect di line.

## Correlation

Another term to sabi na di **Correlation Coefficient** between X and Y variables. Using scatterplot, you fit quickly visualize dis coefficient. If plot get datapoints scatter for neat line, e get high correlation, but if datapoints scatter everywhere between X and Y, e get low correlation.

Good linear regression model na di one way get high (near 1 pass near 0) Correlation Coefficient using Least-Squares Regression method and regression line.

✅ Run di notebook wey join dis lesson and check di Month to Price scatterplot. Di data way associate Month to Price for pumpkin sales get high or low correlation according to your visual interpretation? E go change if you use more detailed measure instead of `Month`, like *day of di year* (meaning number of days since year start)?

For di code below, we go assume say we don clean di data, and get data frame named `new_pumpkins`, similar to dis:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Di code wey clean di data dey inside [`notebook.ipynb`](notebook.ipynb). We do di same cleaning steps like before, and we calculate `DayOfYear` column with dis expression: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Now wey you understand di math behind linear regression, make we create Regression model to see if we fit predict which pumpkin package go get di best pumpkin prices. Anybody buying pumpkins for holiday pumpkin patch fit want dis info to optimize how dem go buy pumpkin packages for di patch.

## Looking for Correlation

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Click di image above to watch short video overview about correlation.

From di last lesson, you probably don see dat average price for different months look like dis:

<img alt="Average price by month" src="../../../../translated_images/pcm/barchart.a833ea9194346d76.webp" width="50%"/>

Dis show say should get some correlation, and we fit try train linear regression model to predict di relationship between `Month` and `Price`, or between `DayOfYear` and `Price`. Dis na di scatter plot wey show di latter one:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pcm/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Make we check if correlation dey using `corr` function:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

E be like say correlation small, -0.15 by `Month` and -0.17 by `DayOfMonth`, but fit be say another important relationship dey. E be like say different clusters of prices dey waka with different pumpkin varieties. To confirm dis hypothesis, make we plot each pumpkin category with different color. By passing `ax` parameter to `scatter` plotting function, we fit plot all points for one graph:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pcm/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Our check show say variety get more effect on overall price pass di actual selling date. We fit see dis with bar graph:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/pcm/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Make we focus for now on one pumpkin variety, di 'pie type', and see wetin di date do to di price:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/pcm/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

If we calculate correlation between `Price` and `DayOfYear` using `corr` function, we go get like `-0.27` - mean say e make sense to train predictive model.

> Before train linear regression model, e important to make sure our data clean well. Linear regression no dey work well if values dey miss, so e make sense to clear all empty cells:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Another way na to fill empty values with mean values from corresponding column.

## Simple Linear Regression

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Click di image above for short video overview of linear and polynomial regression.

To train our Linear Regression model, we go use **Scikit-learn** library.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

We start by separating input values (features) and expected output (label) into separate numpy arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Note say we gatz do `reshape` on input data so Linear Regression package go fit understand am well well. Linear Regression dey expect 2D-array as input, where each row of di array represent vector of input features. Our case, we get only one input - so we need array with shape N&times;1, where N na size of dataset.

Then, we need to split data into train and test datasets so we fit validate our model after training:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finally, training actual Linear Regression model na just two lines of code. We define `LinearRegression` object, and fit am to our data using `fit` method:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Di `LinearRegression` object afta `fit`-ting get all di coefficients of di regression, we fit fit access using `.coef_` property. For our case, na only one coefficient wey dey, wey suppose dey around `-0.017`. E mean sey prices fit dey drop small with time, but e no too much, around 2 cents per day. We fit access di intersection point of di regression with Y-axis using `lin_reg.intercept_` - e go dey around `21` for our case, wey mean di price for beginning of di year.

To see how correct our model be, we fit predict prices for test dataset, then measure how close our predictions be to di expected values. Dis one fit use mean square error (MSE) metrics, wey be di mean of all squared differences between expected and predicted value.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Our error dey around 2 points, wey be ~17%. E no too good. Another indicator of model quality na di **coefficient of determination**, wey fit get like dis:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

If di value na 0, e mean sey di model no dey consider input data, e just dey act as di *worst linear predictor*, wey na mean value of di result. If di value na 1, e mean sey we fit perfectly predict all expected outputs. For our case, di coefficient na around 0.06, wey low shaa.

We fit also plot di test data join with di regression line so dat we go fit see correct how regression dey work for our case:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pcm/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomial Regression

Another kin of Linear Regression na Polynomial Regression. Sometimes, e for be say the relationship between variables be straight line - like di bigger di pumpkin volume, di higher di price - but sometimes, dis kain relationship no fit plot as plane or straight line. 

✅ Here na [some more examples](https://online.stat.psu.edu/stat501/lesson/9/9.8) of data wey polynomial regression fit work for

Look di relationship between Date and Price again. Dis scatterplot seem like e suppose analyze by straight line? Prices no dey fluctuate sef? For dis situation, you fit try polynomial regression.

✅ Polynomials na mathematical expressions wey fit get one or more variables and coefficients

Polynomial regression dey create curved line to fit nonlinear data better. For our case, if we add squared `DayOfYear` variable enter input data, we suppose fit our data with parabolic curve, wey get minimum for one point inside di year.

Scikit-learn get beta [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) wey dey help combine different data processing steps together. **pipeline** na chain of **estimators**. For our case, we go create pipeline wey first add polynomial features to our model, then train di regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Use `PolynomialFeatures(2)` mean sey we go include all second-degree polynomials from di input data. For us, e mean say na just `DayOfYear`<sup>2</sup>, but if we get two input variables X and Y, e go add X<sup>2</sup>, XY and Y<sup>2</sup>. We fit also use higher degree polynomials if we want.

Pipelines fit act like di original `LinearRegression` object, e.g. we fit `fit` di pipeline, then use `predict` to get prediction results. Dis na di graph wey show test data and approximation curve:

<img alt="Polynomial regression" src="../../../../translated_images/pcm/poly-results.ee587348f0f1f60b.webp" width="50%" />

Using Polynomial Regression, we fit get small lower MSE and higher determination, but e no too much. We need consider other features!

> You fit see sey minimal pumpkin prices dey around Halloween. How you fit explain dat? 

🎃 Congrats, you just create model wey fit help predict pie pumpkins price. You fit fit repeat di same process for all pumpkin types, but dat one go dey tiring. Make we learn now how to take pumpkin variety enter our model!

## Categorical Features

For correct world, we want fit predict prices for different pumpkin varieties using same model. But di `Variety` column differ from columns like `Month`, because e get non-numeric values. Dis kind columns na **categorical**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Click di image wey dey above for short video wey summarize how to use categorical features.

Here you fit see how average price depend on variety:

<img alt="Average price by variety" src="../../../../translated_images/pcm/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

To take variety enter account, we first need to convert am to numeric form, or **encode** am. We get different ways to do am:

* Simple **numeric encoding** go build table of different varieties, then replace variety name with index for that table. Dis one no too good for linear regression, because linear regression go take di numeric value of di index, add am to result, multiply by some coefficient. For our case, di relationship between index number and price no linear, even if we arrange indices some specific way.
* **One-hot encoding** go replace `Variety` column with 4 different columns, one for each variety. Each column go get `1` if that row na dat variety, else `0`. Mean sey four coefficients go dey for linear regression, one for each pumpkin variety, wey dey responsible for "starting price" (or "additional price") for dat variety.

Dis code below show how we fit one-hot encode variety:

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

To train linear regression using one-hot encoded variety as input, we just need to initialize `X` and `y` data well:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Di rest of di code na di same wey we use above for Linear Regression training. If you try am, you go see say mean squared error dey about di same, but coefficient of determination go much higher (~77%). To get better predictions, we fit take more categorical features enter account, plus numeric features, like `Month` or `DayOfYear`. To get one large features array, we fit use `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Here we also take `City` and `Package` type enter account, wey give MSE 2.84 (10%), plus determination 0.94!

## Putting it all together

To make di best model, we fit combine (one-hot encoded categorical + numeric) data from di example above join Polynomial Regression. Dis na di complete code for your ease:

```python
# prepare training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# divide train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# prepare and train di pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data dem
pred = pipeline.predict(X_test)

# calculate MSE and determination score
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

This one suppose give us best determination coefficient near 97%, and MSE=2.23 (~8% prediction error).

| Model | MSE | Determination |
|-------|-----|---------------|
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |
| `Variety` Linear | 5.24 (19.7%) | 0.77 |
| All features Linear | 2.84 (10.5%) | 0.94 |
| All features Polynomial | 2.23 (8.25%) | 0.97 |

🏆 Well done! You create four Regression models for one lesson, plus you improve model quality to 97%. For di last part of Regression, you go learn about Logistic Regression to determine categories.

---
## 🚀Challenge

Test different variables for dis notebook to see how correlation link to model accuracy.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

For dis lesson, we learn about Linear Regression. Get other important Regression types too. Read about Stepwise, Ridge, Lasso and Elasticnet techniques. One good course to study be di [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Assignment 

[Build a Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis document don translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am correct, abeg sabi say automated translation fit get errors or no too correct. Di original document wey dey dia in im own language na di correct one. If na serious matter, better make person wey sabi do proper human translation do am. We no go hold ourselves responsible if pesin misunderstand or misinterpret any tin wey come from dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->