# Xây dựng mô hình hồi quy sử dụng Scikit-learn: hồi quy bốn cách

## Ghi chú dành cho người mới bắt đầu

Hồi quy tuyến tính được sử dụng khi chúng ta muốn dự đoán một **giá trị số** (ví dụ: giá nhà, nhiệt độ hoặc doanh số bán hàng).
Nó hoạt động bằng cách tìm một đường thẳng đại diện tốt nhất cho mối quan hệ giữa các đặc trưng đầu vào và đầu ra.

Trong bài học này, chúng ta tập trung vào việc hiểu khái niệm trước khi khám phá các kỹ thuật hồi quy nâng cao hơn.
![Infographic so sánh hồi quy tuyến tính và đa thức](../../../../translated_images/vi/linear-polynomial.5523c7cb6576ccab.webp)
> Infographic bởi [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Trắc nghiệm trước bài giảng](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bài học này có sẵn bằng R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Giới thiệu

Cho đến nay bạn đã khám phá hồi quy là gì với dữ liệu mẫu lấy từ bộ dữ liệu giá bí ngô mà chúng ta sẽ sử dụng xuyên suốt bài học này. Bạn cũng đã trực quan hóa nó bằng Matplotlib.

Bây giờ bạn đã sẵn sàng để đi sâu hơn vào hồi quy trong ML. Trong khi việc trực quan hóa giúp bạn hiểu dữ liệu, sức mạnh thực sự của Machine Learning đến từ _việc huấn luyện mô hình_. Các mô hình được huấn luyện trên dữ liệu lịch sử để tự động nắm bắt các phụ thuộc dữ liệu, và cho phép bạn dự đoán kết quả cho dữ liệu mới mà mô hình chưa từng thấy trước đó.

Trong bài học này, bạn sẽ học thêm về hai loại hồi quy: _hồi quy tuyến tính cơ bản_ và _hồi quy đa thức_, cùng với một số phép toán nền tảng của các kỹ thuật này. Những mô hình đó sẽ giúp chúng ta dự đoán giá bí ngô dựa trên các dữ liệu đầu vào khác nhau.

[![Học máy cho người mới - Hiểu về hồi quy tuyến tính](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Học máy cho người mới - Hiểu về hồi quy tuyến tính")

> 🎥 Nhấn vào hình ảnh trên để xem video giới thiệu ngắn về hồi quy tuyến tính.

> Xuyên suốt chương trình học này, chúng tôi giả định kiến thức toán học tối thiểu, và cố gắng làm cho nó dễ tiếp cận với học viên đến từ các lĩnh vực khác, vì vậy hãy chú ý đến các ghi chú, 🧮 phần tóm tắt, sơ đồ và các công cụ học tập khác để hỗ trợ hiểu bài.

### Yêu cầu trước

Bạn nên đã quen thuộc với cấu trúc dữ liệu bí ngô mà chúng ta đang xem xét. Bạn có thể tìm thấy nó được tải sẵn và đã được làm sạch trong tập _notebook.ipynb_ của bài học này. Trong tập đó, giá bí ngô được hiển thị theo từng bushel trong một bảng dữ liệu mới. Đảm bảo bạn có thể chạy các notebook này trong các kernel của Visual Studio Code.

### Chuẩn bị

Như lời nhắc nhở, bạn đang tải dữ liệu này để đặt câu hỏi cho nó.

- Khi nào là thời điểm tốt nhất để mua bí ngô?
- Tôi có thể kỳ vọng giá bao nhiêu cho một thùng bí ngô mini?
- Tôi có nên mua chúng theo giỏ nửa bushel hay theo hộp 1 1/9 bushel?
Hãy tiếp tục khám phá dữ liệu này.

Trong bài học trước, bạn đã tạo một Pandas data frame và điền dữ liệu từ một phần của bộ dữ liệu gốc, chuẩn hóa giá theo bushel. Tuy nhiên, bằng cách đó bạn chỉ có thể lấy được khoảng 400 điểm dữ liệu và chỉ cho các tháng mùa thu.

Hãy xem dữ liệu mà chúng tôi đã tải sẵn trong notebook kèm bài học này. Dữ liệu được tải sẵn và một biểu đồ phân tán ban đầu được vẽ để hiển thị dữ liệu theo tháng. Có thể chúng ta sẽ có thêm thông tin chi tiết về bản chất của dữ liệu bằng cách làm sạch nó kỹ hơn.

## Đường hồi quy tuyến tính

Như bạn đã học trong Bài 1, mục tiêu của bài tập hồi quy tuyến tính là có thể vẽ được một đường để:

- **Hiển thị mối quan hệ giữa các biến.** Hiển thị mối quan hệ giữa các biến
- **Dự đoán.** Dự đoán chính xác vị trí của một điểm dữ liệu mới so với đường đó.

Điều phổ biến trong **Hồi quy bình phương tối thiểu** là vẽ kiểu đường như thế này. Thuật ngữ "Bình phương tối thiểu" đề cập đến quá trình giảm thiểu tổng sai số trong mô hình. Với mỗi điểm dữ liệu, chúng ta đo khoảng cách theo chiều dọc (gọi là phần dư) giữa điểm thực tế và đường hồi quy của chúng ta.

Chúng ta bình phương các khoảng cách này vì hai lý do chính:

1. **Độ lớn hơn Hướng:** Chúng ta muốn coi sai số -5 giống như sai số +5. Việc bình phương biến tất cả giá trị thành số dương.

2. **Phạt các ngoại lệ:** Việc bình phương làm tăng trọng số cho các sai số lớn hơn, buộc đường phải nằm gần những điểm nằm xa hơn.

Sau đó, chúng ta cộng tất cả các giá trị bình phương này lại với nhau. Mục tiêu là tìm đường cụ thể mà tổng này đạt giá trị nhỏ nhất (giá trị nhỏ nhất có thể) — do đó gọi là "Bình phương tối thiểu".

> **🧮 Cho tôi xem toán học** 
> 
> Đường này, gọi là _đường phù hợp tốt nhất_, có thể biểu diễn bằng [một phương trình](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` là biến 'giải thích'. `Y` là biến 'phụ thuộc'. Độ dốc của đường là `b` và `a` là giao điểm y, tức giá trị của `Y` khi `X = 0`. 
>
>![tính độ dốc](../../../../translated_images/vi/slope.f3c9d5910ddbfcf9.webp)
>
> Đầu tiên, tính độ dốc `b`. Infographic bởi [Jen Looper](https://twitter.com/jenlooper)
>
> Nói cách khác, và liên quan đến câu hỏi gốc của dữ liệu bí ngô: "dự đoán giá bí ngô theo bushel theo tháng", `X` sẽ là giá và `Y` sẽ là tháng bán. 
>
>![hoàn thành phương trình](../../../../translated_images/vi/calculation.a209813050a1ddb1.webp)
>
> Tính giá trị của Y. Nếu bạn trả khoảng 4 đô la, chắc chắn là tháng Tư rồi! Infographic bởi [Jen Looper](https://twitter.com/jenlooper)
>
> Phép toán tính đường phải thể hiện độ dốc đường, cũng phụ thuộc vào giao điểm, hay vị trí của `Y` khi `X = 0`.
>
> Bạn có thể xem phương pháp tính các giá trị này trên trang web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Cũng hãy ghé thăm [máy tính bình phương tối thiểu này](https://www.mathsisfun.com/data/least-squares-calculator.html) để xem cách các giá trị số gây ảnh hưởng cho đường như thế nào.

## Tương quan

Một thuật ngữ nữa cần hiểu là **Hệ số tương quan** giữa các biến X và Y cho trước. Sử dụng biểu đồ phân tán, bạn có thể nhanh chóng hình dung hệ số này. Một biểu đồ với các điểm dữ liệu nằm gần một đường thẳng thể hiện tương quan cao, còn biểu đồ với các điểm phân tán khắp nơi giữa X và Y thì thể hiện tương quan thấp.

Một mô hình hồi quy tuyến tính tốt sẽ có Hệ số tương quan cao (gần 1 hơn 0) sử dụng phương pháp Hồi quy bình phương tối thiểu với một đường hồi quy.

✅ Hãy chạy notebook kèm theo bài học này và xem biểu đồ phân tán Tháng với Giá. Dữ liệu liên kết Tháng với Giá cho doanh số bí ngô có vẻ có tương quan cao hay thấp, theo cách bạn quan sát biểu đồ phân tán? Điều này có thay đổi nếu bạn dùng phép đo chi tiết hơn thay vì `Month`, ví dụ *ngày của năm* (tức số ngày kể từ đầu năm)?

Trong đoạn mã dưới đây, chúng ta giả định đã làm sạch dữ liệu, và thu được data frame gọi là `new_pumpkins`, tương tự như sau:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Mã để làm sạch dữ liệu có trong [`notebook.ipynb`](notebook.ipynb). Chúng tôi đã thực hiện các bước làm sạch tương tự như bài học trước, và đã tính cột `DayOfYear` bằng biểu thức sau:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Bây giờ bạn đã hiểu toán học phía sau hồi quy tuyến tính, hãy tạo một mô hình hồi quy để xem liệu chúng ta có thể dự đoán gói bí ngô nào sẽ có giá tốt nhất không. Ai đó mua bí ngô cho mùa lễ hội có thể muốn thông tin này để tối ưu hóa việc mua các gói bí ngô cho mùa lễ.

## Tìm kiếm tương quan

[![Học máy cho người mới - Tìm kiếm tương quan: Chìa khóa của hồi quy tuyến tính](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Học máy cho người mới - Tìm kiếm tương quan: Chìa khóa của hồi quy tuyến tính")

> 🎥 Nhấn vào hình ảnh trên để xem video tóm tắt về tương quan.

Từ bài học trước bạn có thể đã thấy giá trung bình theo các tháng trông như sau:

<img alt="Giá trung bình theo tháng" src="../../../../translated_images/vi/barchart.a833ea9194346d76.webp" width="50%"/>

Điều này cho thấy có thể có một số tương quan, và chúng ta có thể thử huấn luyện mô hình hồi quy tuyến tính để dự đoán mối quan hệ giữa `Month` và `Price`, hoặc giữa `DayOfYear` và `Price`. Dưới đây là biểu đồ phân tán cho thấy mối quan hệ sau:

<img alt="Biểu đồ phân tán Giá so với Ngày trong năm" src="../../../../translated_images/vi/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Hãy xem có tương quan bằng cách sử dụng hàm `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Có vẻ như tương quan khá nhỏ, -0.15 theo `Month` và -0.17 theo `DayOfMonth`, nhưng có thể có mối quan hệ quan trọng khác. Có vẻ như có các cụm giá khác nhau tương ứng với các loại bí ngô khác nhau. Để xác nhận giả thuyết này, hãy vẽ từng loại bí ngô với màu khác nhau. Bằng cách truyền tham số `ax` cho hàm `scatter` chúng ta có thể vẽ tất cả điểm trên cùng một biểu đồ:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Biểu đồ phân tán Giá so với Ngày trong năm với màu sắc" src="../../../../translated_images/vi/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Cuộc điều tra cho thấy loại bí ngô ảnh hưởng nhiều hơn đến giá tổng thể so với ngày bán thực tế. Ta có thể thấy điều này với biểu đồ thanh:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Biểu đồ thanh giá theo loại bí ngô" src="../../../../translated_images/vi/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Chúng ta hãy tập trung tạm thời chỉ vào một loại bí ngô, loại 'pie type', và xem tác động của ngày đến giá như thế nào:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Biểu đồ phân tán Giá so với Ngày trong năm - loại pie pumpkins" src="../../../../translated_images/vi/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Nếu chúng ta tính tương quan giữa `Price` và `DayOfYear` sử dụng hàm `corr`, ta sẽ thu được giá trị khoảng `-0.27` - nghĩa là việc huấn luyện một mô hình dự đoán là có ý nghĩa.

> Trước khi huấn luyện mô hình hồi quy tuyến tính, điều quan trọng là đảm bảo dữ liệu sạch. Hồi quy tuyến tính không hoạt động tốt với các giá trị thiếu, nên nên loại bỏ tất cả ô trống:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Một cách khác là điền các giá trị thiếu bằng giá trị trung bình của cột tương ứng.

## Hồi quy tuyến tính đơn giản

[![Học máy cho người mới - Hồi quy tuyến tính và đa thức sử dụng Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Học máy cho người mới - Hồi quy tuyến tính và đa thức sử dụng Scikit-learn")

> 🎥 Nhấn vào hình ảnh trên để xem video giới thiệu ngắn về hồi quy tuyến tính và đa thức.

Để huấn luyện mô hình hồi quy tuyến tính, chúng ta sẽ sử dụng thư viện **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Chúng ta bắt đầu bằng cách tách giá trị đầu vào (đặc trưng) và đầu ra dự kiến (nhãn) thành các mảng numpy riêng biệt:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Lưu ý rằng chúng ta phải thực hiện thao tác `reshape` trên dữ liệu đầu vào để gói Linear Regression hiểu đúng. Linear Regression yêu cầu đầu vào là mảng 2 chiều, trong đó mỗi hàng tương ứng với một vector đặc trưng đầu vào. Trường hợp của chúng ta chỉ có một đầu vào, nên cần mảng có kích thước N&times;1, với N là số lượng data.

Sau đó, chúng ta cần chia dữ liệu thành tập huấn luyện và tập kiểm thử để có thể kiểm tra mô hình sau khi huấn luyện:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Cuối cùng, việc huấn luyện mô hình hồi quy tuyến tính thực sự chỉ cần hai dòng lệnh. Chúng ta tạo đối tượng `LinearRegression`, rồi điều chỉnh nó với dữ liệu bằng phương pháp `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Đối tượng `LinearRegression` sau khi thực hiện `fit` chứa tất cả các hệ số của hồi quy, có thể được truy cập thông qua thuộc tính `.coef_`. Trong trường hợp của chúng ta, chỉ có một hệ số, khoảng `-0.017`. Điều này có nghĩa là giá cả có vẻ giảm một chút theo thời gian, nhưng không quá nhiều, khoảng 2 xu mỗi ngày. Chúng ta cũng có thể truy cập điểm cắt của đường hồi quy với trục Y bằng cách sử dụng `lin_reg.intercept_` - nó sẽ khoảng `21` trong trường hợp của chúng ta, cho thấy giá vào đầu năm.

Để xem mô hình của chúng ta chính xác như thế nào, ta có thể dự đoán giá trên một tập dữ liệu kiểm tra, sau đó đo lường mức độ gần với giá trị mong đợi. Điều này có thể được thực hiện bằng cách sử dụng chỉ số lỗi bình phương trung bình (MSE), là trung bình của tất cả các hiệu số bình phương giữa giá trị mong đợi và giá trị dự đoán.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Lỗi của chúng ta có vẻ khoảng 2 điểm, tức khoảng ~17%. Không quá tốt. Một chỉ số khác về chất lượng mô hình là **hệ số xác định**, có thể được lấy như sau:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Nếu giá trị là 0, điều đó có nghĩa mô hình không xem xét dữ liệu đầu vào, và hoạt động như *bộ dự đoán tuyến tính tệ nhất*, là giá trị trung bình của kết quả. Giá trị 1 có nghĩa là chúng ta có thể dự đoán chính xác tất cả các kết quả mong đợi. Trong trường hợp của chúng ta, hệ số này khoảng 0.06, khá thấp.

Chúng ta cũng có thể vẽ dữ liệu kiểm tra cùng với đường hồi quy để thấy rõ cách hồi quy hoạt động trong trường hợp này:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/vi/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Hồi Quy Đa Thức (Polynomial Regression)

Một loại khác của Hồi Quy Tuyến Tính là Hồi Quy Đa Thức. Trong khi đôi khi có mối quan hệ tuyến tính giữa các biến - quả bí ngô càng to về thể tích thì giá càng cao - đôi khi những mối quan hệ này không thể biểu diễn bằng mặt phẳng hoặc đường thẳng.

✅ Đây là [một vài ví dụ khác](https://online.stat.psu.edu/stat501/lesson/9/9.8) về dữ liệu có thể sử dụng Hồi Quy Đa Thức

Hãy nhìn lại mối quan hệ giữa Date và Price. Biểu đồ phân tán này có nhất thiết phải được phân tích bằng một đường thẳng không? Giá cả có thể dao động phải không? Trong trường hợp này, bạn có thể thử hồi quy đa thức.

✅ Đa thức là biểu thức toán học có thể bao gồm một hoặc nhiều biến và hệ số

Hồi quy đa thức tạo ra một đường cong để phù hợp hơn với dữ liệu phi tuyến. Trong trường hợp của chúng ta, nếu thêm biến `DayOfYear` bình phương vào dữ liệu đầu vào, ta có thể phù hợp dữ liệu bằng một đường parabol, sẽ có điểm cực tiểu tại một thời điểm nhất định trong năm.

Scikit-learn bao gồm một [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) hữu ích để kết hợp các bước xử lý dữ liệu khác nhau với nhau. Một **pipeline** là chuỗi các **estimator**. Trong trường hợp này, ta sẽ tạo pipeline trước tiên thêm các đặc trưng đa thức vào mô hình, sau đó đào tạo hồi quy:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Sử dụng `PolynomialFeatures(2)` có nghĩa là ta sẽ bao gồm tất cả các đa thức bậc hai từ dữ liệu đầu vào. Trong trường hợp của chúng ta chỉ có `DayOfYear`<sup>2</sup>, nhưng nếu có hai biến đầu vào X và Y, nó sẽ thêm X<sup>2</sup>, XY và Y<sup>2</sup>. Ta cũng có thể sử dụng đa thức bậc cao hơn nếu muốn.

Pipeline có thể được sử dụng giống như đối tượng `LinearRegression` gốc, tức là có thể `fit` pipeline, sau đó dùng `predict` để có kết quả dự đoán. Dưới đây là đồ thị thể hiện dữ liệu kiểm tra và đường cong xấp xỉ:

<img alt="Polynomial regression" src="../../../../translated_images/vi/poly-results.ee587348f0f1f60b.webp" width="50%" />

Sử dụng Hồi Quy Đa Thức, ta có thể thu được MSE thấp hơn một chút và hệ số xác định cao hơn, nhưng không đáng kể. Ta cần xem xét các đặc trưng khác!

> Bạn có thể thấy giá bí ngô thấp nhất được quan sát vào khoảng dịp Halloween. Bạn giải thích điều này như thế nào?

🎃 Chúc mừng, bạn vừa tạo một mô hình có thể dự đoán giá bí ngô làm bánh. Bạn có thể làm tương tự cho tất cả các loại bí ngô, nhưng điều đó sẽ khá tẻ nhạt. Bây giờ hãy học cách xem xét loại bí ngô trong mô hình của chúng ta!

## Đặc Trưng Phân Loại (Categorical Features)

Trong thế giới lý tưởng, chúng ta muốn dự đoán giá cho các giống bí ngô khác nhau dùng cùng một mô hình. Tuy nhiên, cột `Variety` hơi khác với các cột như `Month`, vì nó chứa các giá trị không phải số. Những cột như vậy được gọi là **categorical**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Nhấp vào hình ảnh trên để xem video tổng quan ngắn về cách dùng đặc trưng phân loại.

Ở đây bạn có thể thấy giá trung bình phụ thuộc vào giống:

<img alt="Average price by variety" src="../../../../translated_images/vi/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Để xem xét giống, trước tiên ta cần chuyển nó sang dạng số, hay còn gọi là **mã hóa**. Có vài cách để làm điều này:

* **Mã hóa số đơn giản** sẽ xây dựng bảng các giống khác nhau, rồi thay thế tên giống bằng chỉ số trong bảng đó. Đây không phải ý tưởng tốt cho hồi quy tuyến tính vì hồi quy tuyến tính lấy giá trị số của chỉ số, và cộng nó vào kết quả nhân với hệ số tương ứng. Trong trường hợp này, mối quan hệ giữa số chỉ mục và giá là phi tuyến rõ ràng, ngay cả khi chúng ta sắp xếp chỉ mục theo 1 cách cụ thể.
* **Mã hóa one-hot** sẽ thay thế cột `Variety` bằng 4 cột khác nhau, mỗi cột dành một giống. Mỗi cột sẽ chứa `1` nếu dòng tương ứng thuộc giống đó, và `0` nếu không. Điều này có nghĩa hồi quy tuyến tính sẽ có bốn hệ số, mỗi hệ số ứng với một giống bí ngô, đại diện cho "giá khởi điểm" (hay chính xác hơn là "giá cộng thêm") của giống đó.

Đoạn mã dưới đây cho thấy cách ta mã hóa one-hot cho giống:

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

Để huấn luyện hồi quy tuyến tính sử dụng dạng one-hot mã hóa giống làm đầu vào, ta chỉ cần khởi tạo dữ liệu `X` và `y` đúng cách:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Phần còn lại của mã giống như đã dùng trên để huấn luyện hồi quy tuyến tính. Nếu bạn thử, sẽ thấy lỗi bình phương trung bình (MSE) gần như không đổi, nhưng hệ số xác định tăng lên đáng kể (~77%). Để có dự đoán chính xác hơn nữa, ta có thể thêm các đặc trưng phân loại khác cũng như các đặc trưng số, như `Month` hay `DayOfYear`. Để có một mảng đặc trưng lớn, ta có thể dùng `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tại đây ta cũng xem xét `City` và loại `Package`, cho kết quả MSE 2.84 (10%), và hệ số xác định 0.94!

## Kết Hợp Toàn Bộ

Để có mô hình tốt nhất, ta có thể dùng dữ liệu kết hợp (mã hóa one-hot các đặc trưng phân loại + đặc trưng số) từ ví dụ trên cùng với Hồi Quy Đa Thức. Dưới đây là đoạn mã hoàn chỉnh để thuận tiện sử dụng:

```python
# thiết lập dữ liệu huấn luyện
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# thực hiện chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# thiết lập và huấn luyện pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# dự đoán kết quả cho dữ liệu kiểm tra
pred = pipeline.predict(X_test)

# tính MSE và hệ số xác định
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Điều này sẽ cho hệ số xác định tốt nhất gần 97%, và MSE=2.23 (~8% lỗi dự đoán).

| Mô hình | MSE | Hệ số xác định |
|---------|-----|----------------|
| Hồi Quy Tuyến Tính với `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Hồi Quy Đa Thức với `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Hồi Quy Tuyến Tính với `Variety` | 5.24 (19.7%) | 0.77 |
| Hồi Quy Tuyến Tính với tất cả đặc trưng | 2.84 (10.5%) | 0.94 |
| Hồi Quy Đa Thức với tất cả đặc trưng | 2.23 (8.25%) | 0.97 |

🏆 Làm tốt lắm! Bạn đã tạo bốn mô hình hồi quy trong một bài học, và cải thiện chất lượng mô hình lên 97%. Trong phần cuối cùng về hồi quy, bạn sẽ học về Hồi Quy Logistic để phân loại.

---
## 🚀Thách Thức

Thử nghiệm nhiều biến khác nhau trong sổ tay này để xem mức độ tương quan ảnh hưởng thế nào đến độ chính xác mô hình.

## [Bài kiểm tra sau bài giảng](https://ff-quizzes.netlify.app/en/ml/)

## Ôn tập & Tự học

Trong bài học này, chúng ta học về Hồi Quy Tuyến Tính. Có những loại hồi quy quan trọng khác. Hãy đọc về các kỹ thuật Stepwise, Ridge, Lasso và Elasticnet. Một khóa học tốt để học thêm là [Khóa học Học thống kê của Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Bài tập

[Xây dựng một Mô hình](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố từ chối trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, vui lòng lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc sai sót. Văn bản gốc bằng ngôn ngữ nguyên bản nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, chúng tôi khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm đối với bất kỳ sự hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->