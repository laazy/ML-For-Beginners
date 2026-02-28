# Scikit-learn을 사용하여 회귀 모델 구축하기: 회귀의 네 가지 방법

## 초보자 주의사항

선형 회귀는 **수치 값**을 예측하고자 할 때 사용합니다(예: 주택 가격, 온도, 판매량).
입력 변수와 출력 간의 관계를 가장 잘 나타내는 직선을 찾는 방식으로 작동합니다.

이 수업에서는 좀 더 고급 회귀 기법을 탐구하기 전에 개념 이해에 집중합니다.
![선형 대 다항 회귀 인포그래픽](../../../../translated_images/ko/linear-polynomial.5523c7cb6576ccab.webp)
> 인포그래픽 출처: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

> ### [이 강의는 R 버전으로도 제공됩니다!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 소개

지금까지 당신은 우리 수업 전반에 걸쳐 사용할 호박 가격 데이터로부터 회귀가 무엇인지 탐구했습니다. 또한 Matplotlib를 사용해 시각화해 보았습니다.

이제 머신러닝을 위한 회귀를 더 깊게 학습할 준비가 되었습니다. 시각화는 데이터를 이해하는 데 도움을 주지만, 머신러닝의 진정한 힘은 _모델 학습_에 있습니다. 모델은 과거 데이터를 학습해 데이터 간의 관계를 자동으로 포착하며, 학습하지 않은 새로운 데이터에 대한 결과를 예측할 수 있게 해줍니다.

이번 수업에서는 _기본 선형 회귀_와 _다항 회귀_ 두 가지 유형과 이 기법들의 수학적 배경에 관해 배웁니다. 이런 모델을 통해 다양한 입력 변수에 따라 호박 가격을 예측할 수 있게 됩니다.

[![초보자용 머신러닝 - 선형 회귀 이해하기](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "초보자용 머신러닝 - 선형 회귀 이해하기")

> 🎥 위 이미지를 클릭하면 선형 회귀에 관한 짧은 동영상을 볼 수 있습니다.

> 이 교육 과정 전반에서 최소한의 수학 지식을 가정하며, 다른 분야에서 온 학생들도 이해하기 쉽도록 노트, 🧮 주석, 도표 및 기타 학습 도구들을 활용합니다.

### 선수 지식

당신은 현재 우리가 다루는 호박 데이터 구조에 익숙할 것입니다. 해당 데이터는 이번 수업의 _notebook.ipynb_ 파일에 미리 불러와지고 전처리되어 있습니다. 이 파일에서는 한 단위(bushel) 당 호박 가격이 새로운 데이터 프레임에 표시되어 있습니다. Visual Studio Code의 커널에서 해당 노트북을 실행할 수 있는지 확인하세요.

### 준비 사항

이 데이터를 불러오는 이유는 질문을 던지기 위해서입니다.

- 호박을 가장 싸게 살 수 있는 시기는 언제일까요?
- 미니어처 호박 하나 가격은 대략 얼마일까요?
- 반 bushel 바구니 단위로 사는 게 좋을까요, 아니면 1 1/9 버셜 박스로 사는 게 좋을까요?

계속해서 이 데이터를 자세히 탐색해 봅시다.

이전 수업에서는 Pandas 데이터 프레임을 만들고 원본 데이터 일부를 기준으로 단위를 버셜별 가격으로 표준화했습니다. 하지만 그 경우 가을철 데이터 약 400개만 수집할 수 있었습니다.

이번 수업에 첨부된 노트북에 미리 불러온 데이터를 확인하세요. 데이터가 미리 불러와져 있으며, 기본 산점도가 월별 데이터를 보여 줍니다. 데이터를 더 정제하면 더 구체적인 데이터 특성을 알 수 있을지도 모릅니다.

## 선형 회귀선

1강에서 배운 것처럼, 선형 회귀 연습의 목표는 다음을 수행할 수 있는 선을 그리는 것입니다.

- **변수 관계 표시.** 변수 간 관계를 시각화합니다.
- **예측 수행.** 새로운 데이터가 이 선과 어떤 관계인지 정확히 예측합니다.

이런 선을 그릴 때 흔히 쓰는 방식이 **최소제곱법 회귀(Least-Squares Regression)**입니다. "최소제곱"이란 모델 오차의 총합을 최소화하는 과정을 뜻합니다. 각 데이터 점과 회귀선 사이 수직 거리(잔차)를 계산하며,

이 거리 값들을 제곱하는 데는 다음 두 가지 이유가 있습니다:

1. **크기만 고려, 방향 무시:** -5와 +5의 오차를 동일하게 취급하기 위해 모든 값을 양수로 만듭니다.

2. **이상치 가중:** 큰 오차에는 제곱값이 훨씬 커지므로 외곽값에 대해 더 강한 패널티를 줍니다.

이 제곱 거리들을 모두 더한 값이 가장 작은 선을 찾는 것이 목표입니다. 이 때문에 "최소제곱"이라는 이름이 붙었습니다.

> **🧮 수학으로 보기**  
>  
> 최적의 선을 의미하는 _최적선_은 [다음 방정식](https://en.wikipedia.org/wiki/Simple_linear_regression)으로 표현할 수 있습니다:  
>  
> ```
> Y = a + bX
> ```
>
> `X`는 독립 변수(설명 변수)이며, `Y`는 종속 변수입니다. 기울기를 나타내는 `b`와 y절편 `a`는 `X=0`일 때 `Y`의 값입니다.  
> 
>![기울기 계산](../../../../translated_images/ko/slope.f3c9d5910ddbfcf9.webp)
> 
> 먼저 기울기 `b`를 계산합니다. 인포그래픽 출처: [Jen Looper](https://twitter.com/jenlooper)
>
> 다시 말해 호박 데이터 문제에서 "한 달 단위로 버셜당 호박 가격을 예측"하는 것으로, `X`는 가격, `Y`는 판매 월을 나타낼 수 있습니다.
>
>![방정식 완성](../../../../translated_images/ko/calculation.a209813050a1ddb1.webp)
>
> `Y` 값을 계산하세요. 만약 가격이 약 4달러라면 4월일 겁니다! 인포그래픽 출처: [Jen Looper](https://twitter.com/jenlooper)
>
> 선의 기울기를 구하는 수학은 단절점(절편)에도 의존하며, `X=0`일 때 `Y`가 어디에 위치하는지 보여 줍니다.
>
> [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 사이트에서 이 계산법을 확인할 수 있습니다. 또한 [이 최소제곱 계산기](https://www.mathsisfun.com/data/least-squares-calculator.html)에서 수치 변화가 선에 미치는 영향을 볼 수 있습니다.

## 상관관계

다음으로 알아야 할 용어는 주어진 X와 Y 변수 간의 **상관계수(Correlation Coefficient)**입니다. 산점도를 통해 빠르게 상관계수를 시각화할 수 있습니다. 데이터 점들이 한 줄로 정렬돼 있으면 상관관계가 높고, 산점도가 흩어져 있으면 상관관계가 낮습니다.

좋은 선형 회귀 모델은 최소제곱 회귀를 사용해 상관계수가 높게(0에 가까운 것보다 1에 가까운) 나오는 모델입니다.

✅ 강의 첨부 노트북을 실행하여 월별-가격 산점도를 살펴보세요. 호박 판매 월별 가격 데이터는 시각적으로 보았을 때 상관관계가 높아 보이나요, 낮아 보이나요? `Month` 대신 더 세분화한 척도(예: *연중 일수*, 즉 1년 시작일부터 지난 일수)를 사용하면 결과가 달라지나요?

다음 코드는 데이터를 정제하여 `new_pumpkins`라는 데이터 프레임을 얻었다고 가정합니다. 예시는 아래와 같습니다:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 데이터 정제 코드는 [`notebook.ipynb`](notebook.ipynb)에 있습니다. 이전 강의와 동일한 정제 단계를 수행했고, `DayOfYear` 열은 다음 식으로 계산했습니다: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

선형 회귀의 수학적 배경을 이해했다면, 이제 어느 호박 패키지가 가장 좋은 가격을 보일지 예측하는 회귀 모델을 만들어 봅시다. 휴일 호박 패치용 호박을 구매하는 사람에게 필요한 정보를 제공하는 것이 목표입니다.

## 상관관계 찾기

[![초보자용 머신러닝 - 상관관계 찾기: 선형 회귀의 핵심](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "초보자용 머신러닝 - 상관관계 찾기: 선형 회귀의 핵심")

> 🎥 위 이미지를 클릭하면 상관관계에 관한 짧은 동영상을 볼 수 있습니다.

앞선 강의에서 월별 평균 가격이 다음과 같음을 봤을 겁니다:

<img alt="월별 평균 가격" src="../../../../translated_images/ko/barchart.a833ea9194346d76.webp" width="50%"/>

이는 상관관계가 다소 있음을 시사하며, `Month`와 `Price` 또는 `DayOfYear`와 `Price` 간의 관계를 예측하려 선형 회귀 모델을 훈련해볼 수 있음을 보여줍니다. 아래 산점도는 후자를 나타냅니다:

<img alt="가격 vs 연중 일수 산점도" src="../../../../translated_images/ko/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

`corr` 함수를 이용하면 상관관계 정도를 알 수 있습니다:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

상관관계는 다소 작아서 `Month` 기준 -0.15, `DayOfMonth` 기준 -0.17 정도지만, 중요한 또 다른 관계가 있을지도 모릅니다. 서로 다른 호박 품종 별 가격 클러스터가 형성되어 보입니다. 이를 확인하려면 각 호박 품종을 다른 색상으로 표시해 봅시다. `scatter` 함수에 `ax` 파라미터를 넘겨 여러 데이터를 한 그래프에 겹쳐 그릴 수 있습니다:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="가격 vs 연중 일수 산점도 - 품종별 색상" src="../../../../translated_images/ko/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

조사 결과 품종이 실제 판매 날짜보다 가격에 더 큰 영향을 미치는 것 같습니다. 다음 막대 그래프로도 볼 수 있습니다:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="품종별 가격 막대 그래프" src="../../../../translated_images/ko/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

지금은 한 가지 품종인 'pie type'에만 집중해 가격에 날짜가 미치는 영향을 살펴봅시다:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="가격 vs 연중 일수 산점도 - pie 타입" src="../../../../translated_images/ko/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

`corr` 함수를 써서 `Price`와 `DayOfYear` 간 상관관계를 계산하면 -0.27 정도로, 예측 모델 훈련하기에 적합함을 알 수 있습니다.

> 선형 회귀 모델을 학습하기 전에 데이터가 깨끗한지 확인해야 합니다. 선형 회귀는 결측치가 있는 데이터에 취약해 결측값이 있으면 데이터를 제거하는 편이 좋습니다:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

또 다른 방법으로, 해당 열의 평균값으로 결측값을 채우는 방법도 있습니다.

## 단순 선형 회귀

[![초보자용 머신러닝 - Scikit-learn으로 선형 및 다항 회귀](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "초보자용 머신러닝 - Scikit-learn으로 선형 및 다항 회귀")

> 🎥 위 이미지를 클릭하면 선형 및 다항 회귀에 관한 간단한 영상 개요를 볼 수 있습니다.

선형 회귀 모델을 훈련하기 위해, **Scikit-learn** 라이브러리를 사용할 것입니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

먼저 입력값(특징)과 예상 출력(레이블)을 별도의 numpy 배열로 분리합니다:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 입력 데이터를 선형 회귀 패키지가 올바르게 이해하도록 `reshape` 처리를 해야 했던 점을 유의하세요. 선형 회귀는 각 행이 입력 특징 벡터인 2차원 배열을 요구합니다. 현재 입력이 하나이므로, 데이터 크기 N에 대해 N×1 배열이 필요합니다.

그다음 데이터를 훈련용과 테스트용으로 나누어 훈련 후 모델을 검증할 수 있도록 합니다:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

마지막으로, 실제 선형 회귀 모델 훈련은 두 줄의 코드만 필요합니다. `LinearRegression` 객체를 정의하고, `fit` 메서드로 데이터를 학습합니다:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit` 후의 `LinearRegression` 객체는 회귀의 모든 계수를 포함하며, `.coef_` 속성을 통해 접근할 수 있습니다. 이 예제에서는 하나의 계수만 있으며, 대략 `-0.017` 정도입니다. 이는 시간이 지남에 따라 가격이 약간 떨어지는 경향이 있음을 의미하며, 하루에 약 2센트 정도 하락하는 셈입니다. 또한 회귀의 Y축과의 교차점을 `lin_reg.intercept_`로 접근할 수 있는데, 이 값은 대략 `21` 정도로, 연초의 가격을 나타냅니다.

모델의 정확도를 확인하려면, 테스트 데이터셋에서 가격을 예측하고 예측값과 실제값이 얼마나 가까운지 측정할 수 있습니다. 이는 평균 제곱 오차(MSE) 지표를 사용해 할 수 있는데, 이는 예상값과 예측값의 제곱 차이의 평균입니다.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

오차는 대략 2포인트로, 약 17% 정도입니다. 그리 좋지 않은 결과입니다. 모델 품질의 또 다른 지표는 **결정 계수(coefficient of determination)**로, 다음과 같이 얻을 수 있습니다:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
값이 0이라면 모델이 입력 데이터를 전혀 고려하지 않고, 단순히 결과의 평균값을 예측하는 *최악의 선형 예측자*임을 의미합니다. 1이라는 값은 모든 출력값을 완벽하게 예측할 수 있음을 뜻합니다. 우리의 경우, 결정 계수는 약 0.06으로 꽤 낮은 편입니다.

또한 회귀선과 테스트 데이터를 함께 그려서 회귀가 어떻게 작동하는지 더 잘 볼 수 있습니다:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/ko/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 다항 회귀 (Polynomial Regression)

선형 회귀의 또 다른 유형은 다항 회귀입니다. 때로 변수들 간에 선형 관계가 존재하기도 하지만 — 예를 들어 부피가 큰 호박일수록 가격이 높음 — 때때로 이런 관계를 평면이나 직선으로 그릴 수 없을 수도 있습니다.

✅ [여기에 다항 회귀를 사용할 수 있는 더 많은 예시](https://online.stat.psu.edu/stat501/lesson/9/9.8)가 있습니다.

날짜와 가격 간 관계를 다시 보세요. 이 산점도가 반드시 직선으로 분석되어야 할까요? 가격이 변동할 수는 없을까요? 이런 경우 다항 회귀를 시도할 수 있습니다.

✅ 다항식은 하나 이상의 변수와 계수로 이루어진 수학적 표현입니다.

다항 회귀는 곡선을 만들어 비선형 데이터를 더 잘 맞춥니다. 우리 경우 입력 데이터에 제곱된 `DayOfYear` 변수를 포함하면, 연 중 특정 지점에서 최솟값을 가지는 포물선 형태로 데이터를 적합할 수 있습니다.

Scikit-learn은 데이터 처리를 여러 단계로 결합할 수 있는 유용한 [파이프라인 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)를 제공합니다. **파이프라인**은 여러 **추정기(estimators)**가 연결된 체인입니다. 이 예제에서는 먼저 다항 특성을 추가하고, 그 다음 회귀를 학습하는 파이프라인을 만듭니다:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)`는 입력 데이터에서 모든 2차 다항식을 포함한다는 의미입니다. 우리 경우에는 `DayOfYear`<sup>2</sup> 뿐이지만, 만약 X와 Y 두 변수가 있다면 X<sup>2</sup>, XY, Y<sup>2</sup>가 추가됩니다. 더 높은 차수의 다항식도 사용할 수 있습니다.

파이프라인은 기본 `LinearRegression` 객체와 동일하게 사용할 수 있어, `fit`으로 학습시키고 `predict`로 결과를 얻을 수 있습니다. 다음은 테스트 데이터와 근사 곡선을 보여주는 그래프입니다:

<img alt="Polynomial regression" src="../../../../translated_images/ko/poly-results.ee587348f0f1f60b.webp" width="50%" />

다항 회귀를 사용하면 평균 제곱 오차(MSE)가 약간 낮아지고 결정 계수가 높아지긴 하지만 크게 개선되지는 않습니다. 다른 특성들을 더 고려할 필요가 있습니다!

> 최소 호박 가격이 할로윈 즈음에 관찰됩니다. 이것을 어떻게 설명할 수 있을까요?

🎃 축하합니다! 파이 호박의 가격을 예측하는 모델을 성공적으로 만들었습니다. 동일한 절차로 모든 호박 종류에 대해 반복할 수 있지만, 번거로울 수 있습니다. 이제 모델에 호박 품종을 반영하는 방법을 배워봅시다!

## 범주형 특성 (Categorical Features)

이상적인 상황에서 우리는 같은 모델로 다양한 호박 품종의 가격을 예측할 수 있기를 원합니다. 하지만 `Variety` 열은 `Month`와 같은 열과 다르게 숫자가 아닌 값을 포함하고 있습니다. 이러한 열을 **범주형(categorical)** 이라고 부릅니다.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 위 이미지를 클릭하면 범주형 특성 사용에 대한 짧은 동영상 개요를 볼 수 있습니다.

아래 그래프는 품종에 따른 평균 가격 변화를 보여줍니다:

<img alt="Average price by variety" src="../../../../translated_images/ko/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

품종을 반영하려면 먼저 이를 숫자형으로 변환하거나, 즉 **인코딩** 해야 합니다. 방법은 여러 가지가 있습니다:

* 간단한 **숫자 인코딩**은 서로 다른 품종 목록을 만들고, 품종 이름을 그 목록의 인덱스로 대체합니다. 이는 선형 회귀에는 최적의 방법이 아닙니다. 선형 회귀는 인덱스의 숫자 값을 그대로 사용해 결과에 계수와 곱하여 더하기 때문입니다. 예를 들어, 우리 예에서 인덱스 숫자와 가격 간 관계는 명백히 비선형적입니다.
* **원-핫 인코딩(one-hot encoding)**은 `Variety` 열을 품종별로 4개의 서로 다른 열로 분리합니다. 각 열은 해당 행이 그 품종이면 `1`, 그렇지 않으면 `0` 값을 가집니다. 이렇게 되면 선형 회귀에는 각 품종마다 하나씩 4개의 계수가 추가되어, 해당 품종의 "기본 가격"(또는 "추가 가격")을 결정합니다.

다음 코드는 품종을 원-핫 인코딩하는 방법을 보여줍니다:

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

원-핫 인코딩된 품종을 입력값으로 하여 선형 회귀를 학습하려면, 단지 `X`와 `y` 데이터를 올바르게 초기화하면 됩니다:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

나머지 코드는 위에서 했던 선형 회귀 학습과 동일합니다. 시도해 보면 평균 제곱 오차는 거의 비슷하지만, 결정 계수가 훨씬 높아져 약 77%가 됩니다. 더 정밀한 예측을 위해서는 더 많은 범주형 특성과 더불어 `Month`나 `DayOfYear` 같은 숫자형 특성도 함께 고려할 수 있습니다. 여러 특성을 하나의 큰 배열로 합치려면 `join`을 사용하면 됩니다:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

여기서는 `City`와 `Package` 유형도 함께 고려하며, 이 경우 MSE는 2.84 (약 10%), 결정 계수는 0.94에 달합니다!

## 모두 함께 사용하기

최적의 모델을 만들기 위해 위 예제에서 범주형(원-핫 인코딩) + 숫자형 데이터를 다항 회귀와 함께 사용할 수 있습니다. 편의를 위해 전체 코드를 여기에 제시합니다:

```python
# 훈련 데이터 설정
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 학습-테스트 분할 수행
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 파이프라인 설정 및 훈련
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 테스트 데이터에 대한 결과 예측
pred = pipeline.predict(X_test)

# MSE 및 결정 계수 계산
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

이렇게 하면 거의 97%의 결정 계수와 MSE=2.23 (약 8% 예측 오차)을 얻을 수 있습니다.

| 모델 | MSE | 결정 계수 |
|-------|-----|---------------|
| `DayOfYear` 선형 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 다항 | 2.73 (17.0%) | 0.08 |
| `Variety` 선형 | 5.24 (19.7%) | 0.77 |
| 모든 특성 선형 | 2.84 (10.5%) | 0.94 |
| 모든 특성 다항 | 2.23 (8.25%) | 0.97 |

🏆 훌륭합니다! 한 강의에서 네 개의 회귀 모델을 만들고 모델 품질을 97%까지 향상시켰습니다. 다음 회귀 강의의 마지막 부분에서 분류를 위한 로지스틱 회귀를 배우게 됩니다.

---
## 🚀도전

이 노트북에서 여러 변수들을 테스트해보고 상관관계가 모델 정확도에 어떻게 영향을 미치는지 확인해 보세요.

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자율학습

이번 강의에서는 선형 회귀에 대해 배웠습니다. 그 외에도 중요한 회귀 유형들이 있습니다. 스텝와이즈, 리지, 라쏘, 엘라스틱넷 기법을 공부해 보세요. 더 배우기에 좋은 강의로는 [스탠포드 통계 학습 강의](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)가 있습니다.

## 과제

[모델 만들기](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역은 오류나 부정확성이 포함될 수 있음을 유의하시기 바랍니다. 원래 문서의 원문이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 본 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->