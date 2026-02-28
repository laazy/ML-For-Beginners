# Scikit-learn အသုံးပြု၍ regression မော်ဒယ် တည်ဆောက်ခြင်း - regression လေးနည်းလမ်း

## စတားသူမှတ်ချက်

Linear regression ကို **ဂဏန်းတန်ဖိုး** (ဥပမာ- အိမ်ဈေးနှုန်း၊ အပူချိန်၊ သို့မဟုတ် ရောင်းအား) ခန့်မှန်းလိုသည့်အခါ အသုံးပြုသည်။
ဒါဟာ input အင်အားများနဲ့ output အကြား ဆက်နွယ်မှုကို အကောင်းဆုံး ကိုယ်စားပြုသော တတ်တရာဟောင်းတစ်ခုကို ရှာဖွေခြင်းဖြင့် အလုပ်လုပ်သည်။

ဒီသင်ခန်းစာမှာတော့ အဆင့်မြင့် regression နည်းများကို ရှာဖွေမတတ်ခင် แนวคิดကို နားလည်ခြင်းကို အာရုံစိုက်သွားမှာဖြစ်ပါတယ်။
![Linear vs polynomial regression infographic](../../../../translated_images/my/linear-polynomial.5523c7cb6576ccab.webp)
> infographic ကို [Dasani Madipalli](https://twitter.com/dasani_decoded) ဖန်တီးသည်။
## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [ဒီသင်ခန်းစာကို R မှာလည်း ရနိုင်ပါတယ်!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### နိဒါန်း

အခုထိ pumpkin ဈေးနှုန်းဒေတာအရေအတွက်နဲ့ regression ဆိုတာ ဘာလဲဆိုတာကို လေ့လာခဲ့ပြီး Matplotlib ဖြင့် ဓာတ်ပုံဆွဲပြီးဖြစ်ပြီဖြစ်ပါသည်။

ယခုအခါ ML အတွက် regression ကို နက်နဲစွာ လေ့လာရန် အသင့်ဖြစ်ပါပြီ။ Visualization က ဒေတာကို နားလည်ဖို့ အကူအညီဖြစ်ပေမယ့် Machine Learning ရဲ့ အဓိကအားသာချက်က _မော်ဒယ်များကို လေ့ကျင့်ခြင်း_ မှာ ဖြစ်သည်။ မော်ဒယ်များကို ရှေးဟောင်းဒေတာပေါ် မူတည်၍ အလိုအလျောက် ဒေတာအချိုးအစားတွေကို အတိအကျဖမ်းယူသည်၊ ထို့ကြောင့် မူလမမြင်ဖူးသော ဒေတာအတွက် ခန့်မှန်းချက်များ ဖန်တီးနိုင်သည်။

ဒီသင်ခန်းစာမှာ၊ _မူလ linear regression_ နဲ့ _polynomial regression_ ဆိုတဲ့ regression နှစ်မျိုးနဲ့ အခြေခံသင်္ချာနည်းများကို တက်တူးလေ့လာပြီး၊ ဤမော်ဒယ်များဖြင့် pumpkin အမျိုးအစားအလိုက် ဈေးနှုန်းခန့်မှန်းမှု ပြုလုပ်နိုင်မှာ ဖြစ်ပါတယ်။

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 ဒေါက်ဗိုင်းလင်ကို နှိပ်၍ linear regression အကြောင်း နိဒါန်းဗီဒီယို ကြည့်ရှုနိုင်ပါသည်။

> ဒီ သင်ရိုးအတန်းများအတွက် သင်္ချာနည်းပညာဆိုင်ရာ အသိပညာနည်းပါးမှသာ ယူဆောင်ထားပြီး၊ အခြားကဏ္ဍမှလာသော ကျောင်းသားများအတွက် လွယ်ကူစေရန် မှတ်စု၊ 🧮 ဟောပြောချက်များ၊ ပုံများနှင့် အခြားသင်ယူရေးကိရိယာများ ထည့်သွင်းထားသည်။

### ဆော့ရွတ်

ယခုအခါ စိမ်းလန်းသစ်ဆန်းသော pumpkin data ဖွဲ့စည်းမှုနဲ့ ရင်းနှီးပြီး ဖြစ်နေသည်။ သင်တို့အတွက် ဒီသင်ခန်းစာ _notebook.ipynb_ ဖိုင်ထဲတွင် ကြိုတင်တင်ထားပြီး သန့်ရှင်းထားပါသည်။ ဖိုင်အတွင်းမှာ pumpkin ဈေးနှုန်းကို bushel အလိုက် ပြသထားပါသည်။ Visual Studio Code ၏ kernel တွင် ဒီ notebook များကို တည်ဆောက်၍ စမ်းသပ်နိုင် ဖို့ သေချာစေပါ။

### ပြင်ဆင်မှု

သတိပေးချက်အဖြစ်, ဒီဒေတာကို လာယူတယ်ဆိုတာ ဒေတာအပေါ် မေးခွန်းတွေ ထုတ်ဖို့ပါပဲ။

- Pumpkin များ ဝယ်ဖို့ အချိန်ကောင်းဆုံး ဘယ်လိုဆိုတာ?
- Miniature pumpkin များ case တစ်ခုမှာ ဘယ်နှစ်နှုန်းထိ အရောင်းရှိမလဲ?
- Pumpkin half-bushel အိတ်လား ဒါမှမဟုတ် 1 1/9 bushel ပုံးလား ဝယ်သင့်မလား?
ဒီဒေတာကို ဆက်လက် ရှာဖွေကြမယ်။

ယခင်သင်ခန်းစာတွင် pandas data frame တစ်ခု တည်ဆောက်ပြီး မူလ dataset မှအချိုးအစားကို bushel အလိုက် တူညီအောင် ပြင်ဆင်ထားသည်။ သို့သော် အဲ့ဒီနည်းဖြင့် datapoint ၄ ရာခိုင်နှုန်းတိုင်သာ ရခဲ့ပြီး ရာသီဆောင်းလများအတွက်သာ ရှိခဲ့သည်။

ဒီသင်ခန်းစာအတွက် အသစ်တင်ထားသော notebook ထဲရှိ ဒေတာကို ကြည့်ပါ။ ဒေတာကို ကြိုတင်တင်ထားပြီး အစပိုင်း scatterplot တစ်ခုကို month data ပြရန် ဖော်ပြထားသည်။ ဒေတာအတွက် ပိုမိုအသေးစိတ် ရယူဖို့ သန့်ရှင်းမှုပိုပြီး လုပ်သင့်သလို ဖြစ်နိုင်သည်။

## Linear regression လိုင်းတစ်ခု

သင် သင်ခန်းစာ ၁ တွင် သင်တန်းရဲ့ ရည်ရွယ်ချက်မှာ ပြောင်းလဲမှု များကိုတစ်ခုချင်းစီ ဘယ်လို ဆက်နွယ်ကြောင်း ပြရန် တိုက်ရိုက်လမ်းကြောင်းဖော်လိုသည်ဆိုတာ ဖြစ်ကြောင်း သိရသည်။

- **ဇာတ်ကောင်များ ဆက်နွယ်မှု ပြသခြင်း**။ မတူညီသော ဇာတ်ကောင်များအကြား ဆက်နွယ်မှု ပြသခြင်း။
- **ခန့်မှန်းခြေပြုလုပ်ခြင်း**။ ပြီးခဲ့တဲ့ ဒေတာဇယားအပေါ် ခန့်မှန်းမှုပြုလုပ်၍ ဒေတာအသစ်အပေါ် ရသရွယ်မှု ပြုခြင်း။

**Least-Squares Regression** သေချာမှုအတွက် ဒီလိုလိုင်းကို ဆွဲတယ်။ "Least-Squares" ဆိုတာ သက်ဆိုင်ရာ မော်ဒယ်အတွင်း ဒေတာမှားယွင်းမှုတစ်ခုချင်းစီအတွက် မူလိုင်ရာမှားယွင်းမှုကို လျှော့ချဖို့ လုပ်ငန်းစဉ်တစ်ခု ဖြစ်ပါသည်။ ဒေတာအချက်လေးတစ်ခုချင်းစီအတွက် စစ်မှန်တဲ့အချက်နဲ့ regression လိုင်းအကြား လျှောက်လွှာဖြင့်ခွဲခြမ်းဖြတ်တောက်မှု (residual) ကိုတိုင်းတာပါသည်။

ဒီအကွာအဝေးကို အောက်ပါ အကြောင်းညီညွတ်မှု နှစ်မျိုးကြောင့် ရေတွက်ပါသည်-

1. **အချက်အလက် ဒေသအလိုက် မျှတခြင်း**: -5 မှာ အမှားတစ်ခုနဲ့ +5 မှာ အမှားတစ်ခုကို တူညီသဘောထားစေဖို့ ရည်ရွယ်သည်။ ကွေးသွားခြင်းအားဖြင့် တန်ဖိုးများအားလုံး မရှိမဖြစ် အပေါင်းလိုက် ဖြစ်စေသည်။

2. **အပြင်မှာရှိနေသောမှားယွင်းမှုများကို ဒဏ်ရာပေးခြင်း**: အကြီးမားဆုံးအမှားများကို ပိုပြီး အာရုံစိုက်စေပြီး၊ လိုင်းသည် အဲ့ဒီဒေတာများနီးပါးမှာ ပိုနီးစပ်အောင် ဖိအားပေးသည်။

ခုနောက်ဆုံး အခြေခံထားတဲ့ squared error သုံးပြီး ပြီးဆုံးရာ တန်ဖိုး၏ စုစုပေါင်းကို တွက်ချက်သည်။ အဲ့ဒီစုစုပေါင်းတန်ဖိုး အနည်းဆုံး ရှိတဲ့ လိုင်းကို ရှာဖွေပါတယ်။ဒါ့ကြောင့် "Least-Squares" လို့ခေါ်ကြတာဖြစ်ပါတယ်။

> **🧮 သင်္ချာအပိုင်း ပြပါ** 
> 
> ဒီလိုင်းကို _line of best fit_ လို့ခေါ်ပြီး [တစ်ခုတည်းသောဆွဲဆန့် regression ဆိပ်ကမ်း](https://en.wikipedia.org/wiki/Simple_linear_regression) ဖြင့်ဖော်ပြနိုင်ပါတယ်။
> 
> ```
> Y = a + bX
> ```
>
> `X` သည် 'အကြောင်းပြချက် ပြောင်းလဲမှု' ကို ဆိုလိုသည်။ `Y` သည် 'ဘယ်လိုပြောင်းလဲမှု ထိခိုက်သည်' ကို ဆိုလိုသည်။ လိုင်း၏ အလျားညီသည် `b` ဖြစ်ပြီး `a` သည် y-intercept ဖြစ်ပြီး `X = 0` ဖြစ်သောအခါ `Y` တန်ဖိုးဖြစ်သည်။
>
>![calculate the slope](../../../../translated_images/my/slope.f3c9d5910ddbfcf9.webp)
>
> ပထမဦးဆုံး slope `b` ကိုတွက်ချက်ပါ။ infographic ကို [Jen Looper](https://twitter.com/jenlooper) ဖန်တီးသည်။
>
> နောက်ထပ် ရှင်းလင်းရန်၊ pumpkin data အရ "လစဉ် တစ် bushel အတွက် pumpkin ဈေးဖော်ပြချက် ခန့်မှန်းပါ" ဆိုသည်မှာ `X` သည် ဈေးနှုန်းကို ဆိုလိုပြီး `Y` သည် အရောင်းလကို ဆိုလိုသည်။
>
>![complete the equation](../../../../translated_images/my/calculation.a209813050a1ddb1.webp)
>
> Y တန်ဖိုးကိုတွက်ချက်ပါ။ သင် $4 လောက်ပေးထားရင် April လဖြစ်မယ်! infographic ကို [Jen Looper](https://twitter.com/jenlooper) ဖန်တီးသည်။
>
> ဒီလိုင်းကိုတွက်ချက်ရာတွင် slope ကို ဖော်ထုတ်ပြီး၊ intercept (သို့) `X = 0` ဖြစ်စဉ် Y တန်ဖိုးရှိရာနေရာတိုက်ရိုက် ကိုဖော်ပြရပါမယ်။
>
> ဤတန်ဖိုးများထုတ်ယူနည်းကို [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) ဆိုက်မှာ တွေ့နိုင်ပြီး၊ [Least-squares ကိရိယာ](https://www.mathsisfun.com/data/least-squares-calculator.html) မှာလည်း နံပါတ်များ၏ သက်ရောက်မှုကို ကြည့်ရှုနိုင်သည်။

## ဆက်စပ်မှု

နားလည်ရန် လိုအပ်သော အခြား အသုံးအနှုန်းမှာ given X နဲ့ Y အကြားရှိ **Correlation Coefficient** ဖြစ်သည်။ Scatterplot တစ်ခုဖြင့် ဒီ coefficient ကို မြန်ဆန်စွာ ဓာတ်ပုံဆွဲကြည့်နိုင်သည်။ တိကျသွားသော အချက်များ စီစဉ်ရိုက်ထားသော plot တွင် Correlation အမြင့်ရှိပြီး၊ အချက်များကြားကွဲဝေးနေသော scatterplot တွင် Correlation နည်းပါးသည်။

Linear regression မော်ဒယ်က Correlation Coefficient မြင့်မား (0 ထက် 1 နီးစပ်သော) ဖြစ်စေရန် Least-Squares Regression နည်းပြုလုပ်ခြင်းဖြင့် တည်ဆောက်သည်။

✅ ဒီသင်ခန်းစာ ပါ Notebook ကို Run လိုက်ပြီး Month နဲ့ Price ရဲ့ scatterplot ကို ကြည့်ပါ။ Pumpkin ရောင်းအားတွင် Month နဲ့ Price ဆက်နွယ်မှုမြင့်မား/နည်းပါးသလား? ဒါမှမဟုတ် `Month` ကျော်ပြီး *နှစ်အတွင်းရက်* နဲ့ ခန့်မှန်းပါက ဘယ်လို ပြောင်းလဲသလဲ?

အောက်ပါ code မှာ `new_pumpkins` data frame ကို ရယူထားပြီး ဆက်သွယ်မှု ရှိတယ်လိုထင်ရတယ်။

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> ဒေတာသန့်ရှင်းမှု ကိစ္စအတွက် code ကို [`notebook.ipynb`](notebook.ipynb) တွင် ရနိုင်သည်။ ယခင်သင်ခန်းစာကဲ့သို့ သန့်ရှင်းမှု လုပ်ဆောင်ပြီး `DayOfYear` column ကို အောက်ပါ အထူးဖော်ပြချက်ဖြင့် တွက်ချက်ထားသည်။

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Linear regression တွင် သင်္ချာနည်းပညာနောက်ခံကို နားလည်လျှင်၊ pumpkin package များအတွက် အကောင်းဆုံး ဈေးနှုန်း ခန့်မှန်းရန် Regression မော်ဒယ်ကို ဖန်တီးကြမယ်။ အစပျိုး pumpkin patch အတွက် pumpkin ဝယ်သူများအတွက် package ဝယ်ယူမှု တိုးတက်အောင် အကောင်းဆုံး ကိုယ်စားလှယ်ဖြစ်စေခဲ့ပါလိမ့်မယ်။

## ဆက်စပ်မှု ရှာခြင်း

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 correlation ကိုအကျဉ်းသွားသော ဗီဒီယိုအတွက် ဓာတ်ပုံကို နှိပ်ပါ။

ယခင်သင်ခန်းစာမှ ဖော်ပြသည့်အတိုင်း လစဉ် ပျမ်းမျှဈေးနှုန်းသည် အောက်ပါအတိုင်း ပုံဖော်ထားသည်။

<img alt="Average price by month" src="../../../../translated_images/my/barchart.a833ea9194346d76.webp" width="50%"/>

ဒါက ဆက်စပ်မှုရှိစေသင့်ကြောင်း ပြထားပြီး `Month` နဲ့ `Price` အကြား သို့မဟုတ် `DayOfYear` နဲ့ `Price` အကြား ဆက်နွယ်မှု မော်ဒယ်တည်ဆောက်ဖို့ ကြိုးစားကြည့်နိုင်သည်။ အောက်တွင် ယခုဆက်နွယ်မှုကို ပြသသည့် scatterplot ပါ -

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/my/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

`corr` function သုံး၍ ဆက်စပ်မှုကြည့်ပါ -

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

အခြေအနေသည် correlation နည်းပါးသည်ကို လေ့လာရသည်၊ `Month` အတွက် -0.15၊ `DayOfMonth` အတွက် -0.17 ဖြစ်သော်လည်း pumpkin အမျိုးအစားအလိုက် group လည်းရှိသည်။ ဒီ hypothesis မှန်ကန်ရန် တစ်ခုချင်းစီကို အခြားစိတ်ကြည့်အရောင်အားဖြင့် scatter ပြကြမည်။ `scatter` function ထဲမှာ `ax` parameter ပေးပြီး နေရာတူမှန်တဲ့ ဂရပ် ဖော်ရန် -

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/my/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

စစ်ဆေးမှုချင်းရန် အမျိုးအစားပေါ် မူတည်၍ ဈေးနှုန်းထက် ရောင်းတော့မည့်နေ့ရက် ပိုမိုသက်ရောက်မှုရှိသည်ဟု တွေ့ရှိသည်။ အပြည့်အစုံ ကြည့်ရန် -

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/my/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

နေ့ရက် သက်ရောက်မှုကို တစ်မျိုးတည်း pumpkin အမျိုးအစား 'pie type' အပေါ် ဦးတည်လျှောက်လဲကြည့်ပါ။

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/my/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

`Price` နှင့် `DayOfYear` အကြား correlation ကို `corr` function ဖြင့်တွက်လျှင် `-0.27` သို့မဟုတ် အနီးပါး ဖြစ်ပြီး ဖြစ်လာသည်မှာ သင်တန်းတက်ရန် ပြုလုပ်ယုံကြည်ရပါသည်။

> Linear regression မော်ဒယ် လေ့လာဖို့ မတိုင်မီ ဒေတာ သန့်ရှင်းမှုအတိုင်းအတာ ပြုလုပ်ပါ။ Linear regression သည် information ကျန်ရှိမှုများ (missing value) ဖြင့် မကောင်းဘူး၊ ထို့ကြောင့် ခြိမ်းခြောက်မှုရှိသော cell များကို ဖယ်ရှားပါ။

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

တစ်ခြားနည်းဖြင့် အတားအဆီးရှိသော ထောင့်များကို အထူးအတွက် များ၏ ပျမ်းမျှတန်ဖိုးဖြင့် ဖြည့်တင်းနိုင်သည်။

## ရိုးရှင်းသော Linear Regression

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 ဖြစ်ပေါ်သူများအတွက် Linear နဲ့ Polynomial Regression အကြောင်း နိဒါန်းဗီဒီယိုကြည့်ရန်ဓာတ်ပုံကို နှိပ်ပါ။

Linear Regression မော်ဒယ် သင်တန်းတက်ဖို့ **Scikit-learn** library ကို အသုံးပြုမည်။

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

input တန်ဖိုးများ (features) နှင့် အလားအလာ output (label) ကို numpy array တွဲ ဘာသာပြန်ထားပါတယ်။

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Linear Regression ကို သေချာနားလည်ဖို့ input data ကို `reshape` ပြောင်းတာ ဆိုင်းလိုင်းတစ်ခုသည် 2D-array ဖြစ်ရမည်ဖြစ်သည်။ input feature vector တစ်ခုချင်းစီသည်တန်းတစ်ကြောင်းတွင် ရှိရပါမည်။ ဒါကြောင့်၊ input တွေက တစ်ခုတည်းပါ ကနေ N×1 ရောဂါရှိသော array ဖြစ်အောင် ပြင်ထားရတယ်။

ထို့နောက် data ကို train နှင့် test dataset များခွဲခြားမယ်၊ သင်တန်းပြီးတဲ့နောက် မော်ဒယ်ကို စစ်ဆေးနိုင်ရန် ဖြစ်ပါတယ်။

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

နောက်ဆုံး Linear Regression မော်ဒယ် အတွက် training ကို နှစ်ကြောင်း ကုတ်ဖြင့် ပြုလုပ်နိုင်ပါသည်။ `LinearRegression` object ကိုသတ်မှတ်ပြီး `fit` method ဖြင့် data ကို လေ့ကျင့်သည်။

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit` လုပ်ပြီးနောက် `LinearRegression` အရာဝတ္ထုတွင် regression ၏ coefficient အားလုံးပါရှိပြီး၊ ပြီးတော့ `.coef_` ဆက်တင်မှတဆင့် ဝင်ရောက်မြင်နိုင်သည်။ ကျွန်ုပ်တို့ကိစ္စတွင် coefficient တစ်ခုသာရှိပြီး၊ ယင်းသည် အကြမ်းအားဖြင့် `-0.017` လောက် ဖြစ်ရမည်။ ၎င်းသည် စျေးနှုန်းများသည် အချိန်နှင့်အမျှ သေးငယ်စွာ လျော့နည်းသွားသော ဖြစ်ရပ်ဖြစ်ပြီး၊ တစ်ရက်လျော့တန်ဖိုးဟု ၂ ဆင့်ခန့်သာရှိသည်ဟု ဆိုလိုသည်။ အနည်းငယ် များစွာ လျော့နည်းသော်လည်း များစွာမဟုတ်ပါ။ ဒါ့အပြင် regression ၏ Y-axis နှင့် ဖြတ်သွားသော အချက်ကို `lin_reg.intercept_` မှတဆင့် ဝင်ရောက်ကြည့်ရှုနိုင်ပြီး၊ ကျွန်ုပ်တို့ကိစ္စတွင် ၂၁ အနီးဖြစ်ပြီး၊ ၎င်းသည် နှစ်အစောပိုင်းတွင် စျေးနှုန်းကို ဖော်ပြသည်။

ကျွန်ုပ်တို့၏ မော်ဒယ်တိကျမှု မည်မျှရှိသည်ကို ကြည့်ရန်၊ စမ်းသပ်ဒေတာတွင် စျေးနှုန်းများကို ခန့်မှန်းပြီး၊ ထိုခန့်မှန်းချက်များနှင့် မျှော်မှန်းထားသောတန်ဖိုးများမှာ မည်မျှနီးစပ်သည်ကို တိုင်းတတ်နိုင်သည်။ ၎င်းသည် mean square error (MSE) စဉ်တန်းကို အသုံးပြု၍ ပြုလုပ်နိုင်ပြီး၊ ယင်းသည် မျှော်မှန်းထားသောတန်ဖိုးနှင့် ခန့်မှန်းထားသောတန်ဖိုးကြားရှိ စတုရန်းကွာခြားမှုအားလုံး၏ ပျမ်းမျှတန်ဖိုး ဖြစ်သည်။

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
  
ကျွန်ုပ်တို့၏ အမှားသည် ၂ အမှတ်လောက်ဖြစ်ပြီး၊ ၁၇% ဆန်လောက်ဖြစ်သည်။ မမှန်ကန်ပါ။ မော်ဒယ်အရည်အသွေး၏ အခြားညွှန်ပြချက်ကတော့ **coefficient of determination** ဖြစ်ပြီး၊ အောက်ပါအတိုင်း ရယူနိုင်သည် -

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
  
တန်ဖိုးသည် ၀ ဖြစ်လျှင်၊ မော်ဒယ်သည် input data ကိုမထည့်သွင်းသောဖြစ်ပြီး၊ *အဆိုးဆုံး လျင်မြန်စွာ ခန့်မှန်းသူ* အဖြစ် လုပ်ဆောင်နေသည်။ ၎င်းသည် ရလဒ်၏ ပျမ်းမျှတန်ဖိုးတစ်ခုသာဖြစ်သည်။ တန်ဖိုး ၁ ဖြစ်လျှင်၊ မျှော်မှန်းထားသော output အားလုံးကို တိကျမှန်ကန်စွာ ခန့်မှန်းနိုင်ခြင်းကို ဆိုလိုသည်။ ကျွန်ုပ်တို့ကိစ္စတွင် coefficient သည် ၀.၀၆ အနီးဖြစ်ပြီး၊ သက်သက်မနည်းပါ။

ကျွန်ုပ်တို့သည် စမ်းသပ်ဒေတာကို regression လိုင်းနှင့်ပေါင်းပြီး ဖြတ်ပေါ်ပုံကို ကြည့်ရှုနိုင်သည် -

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```
  
<img alt="Linear regression" src="../../../../translated_images/my/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomial Regression

Linear Regression ၏ အမျိုးအစားတစ်ခုမှာ Polynomial Regression ဖြစ်သည်။ အချို့အခါတွင် 변수များအကြား လျှင်မြန်သော ဆက်စပ်မှုရှိသည် - ပုံမှန်အားဖြင့် pumpkin ၏ အရွယ်အစားပိုကြီးသလောက် စျေးနှုန်းပိုများသွားကြသည်။ သို့သော် အချို့အခါ၌ ယင်းဆက်စပ်မှုများကို စတိတ်တန်းသို့မဟုတ် ရှည်လျားသော လိုင်းအဖြစ် ပုံဖော်၍ မရနိုင်ပါ။

✅ ဒီထဲမှာ [Polynomial Regression အသုံးပြုနိုင်သော နမူနာများ](https://online.stat.psu.edu/stat501/lesson/9/9.8) တချို့ ရှိပါသည်။

Date နှင့် Price နှင့် ဆက်စပ်မှုကို ထပ်မံကြည့်ပါ။ ၎င်း scatterplot သည် လျှင်မြန်သော လိုင်းဖြင့် လေ့လာသင့်သည်ဟု သင်ထင်ပါသလား? စျေးနှုန်းတွေ အပြောင်းအလဲများ မဖြစ်နိုင်ပါသလား? ဒီကိစ္စတွင် polymer regression ကို စမ်းသုံးနိုင်သည်။

✅ Polynomials ဆိုသည်မှာ များစွာသော variables နှင့် coefficient များပါဝင်နိုင်သော ဂဏန်းတွက်ချက်မှုဖော်ပြချက်များ ဖြစ်သည်။

Polynomial regression သည် nonlinear data များကို ပိုမိုသင့်လျော်စွာသိမ်းဆည်းရန် လိမ္မော်လိုင်းမျဉ်း curvature တစ်ခုကို ဖန်တီးပေးသည်။ ကျွန်ုပ်တို့ကိစ္စတွင် `DayOfYear` variable ကို အချက်အလက်ထဲတွင် ထပ်ထည့်ပါက data ကို parabolic curve ဖြင့် သိုလှောင်နိုင်ပြီး၊ တစ်နှစ်အတွင်း အနည်းဆုံးတန်ဖိုးရှိသောနေရာ တစ်ခုရှိမည်ဖြစ်သည်။

Scikit-learn သည် အချက်အလက်စီမံခန့်ခွဲမှု အဆင့်များကို ပေါင်းစပ်နိုင်သည့် 유용한 [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) ကို ပါရှိသည်။ **pipeline** ဆိုသည်မှာ **estimators** များ လှိုင်းလိုက်မှု ဖြစ်သည်။ ကျွန်ုပ်တို့ကိစ္စတွင် သင်္ချာစနစ်ကို ပထမဆုံး polynomial features ထပ်ထည့်ပြီး၊ ထို့နောက် regression ကို လေ့ကျင့်သွားမည် -

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```
  
`PolynomialFeatures(2)` အသုံးပြုခြင်းသည် input data ထဲမှ ဒုတိယအဆင့် polynomials အားလုံးကို ထည့်သွင်းသည့် အဓိပ္ပာယ်ဖြစ်သည်။ ကျွန်ုပ်တို့ကိစ္စတွင် `DayOfYear`<sup>2</sup> ဖြစ်မည်၊ ဒါပေမဲ့ variable နှစ်ခုအတွက် X နှင့် Y ရှိပါက X<sup>2</sup>, XY, Y<sup>2</sup> ကို ထည့်သွင်းမည်။ အဆင့်မြင့် polynomials များကိုလည်း အသုံးပြုနိုင်သည်။

Pipeline များကို မူရင်း `LinearRegression` အရာဝတ္ထုပုံစံနဲ့ လိုက်ဖက်၍ သုံးနိုင်ပြီး၊ `fit` လုပ်ပြီးနောက် `predict` ကို သုံး၍ ခန့်မှန်းချက်များရယူနိုင်သည်။ စမ်းသပ်ဒေတာ နှင့် ပြီးရော curve ကို ပြသလိုက်ပါသည် -

<img alt="Polynomial regression" src="../../../../translated_images/my/poly-results.ee587348f0f1f60b.webp" width="50%" />

Polynomial Regression အသုံးပြုကာ နည်းနည်း သိပ်သာသော MSE နှင့် determination ကိုရရှိနိုင်သော်လည်း အရေးကြီးထိရောက်မှုမရှိပါ။ အခြား features များကိုပါ ထည့်သွင်းစဉ်းစားရန် လိုအပ်သည်။

> 🎃 နောက်တစ်ချက် - Halloween လယ်ဝန်းတွင် pumpkin စျေးနည်းနေမှုကို တွေ့ရသည်။ သင်ဘာကြောင့်ထင်သလဲ?

🎃 သင်သည် အပိုင် pumpkin စျေးထုတ်ခန့်မှန်းနိုင်သော မော်ဒယ်တစ်ခု ဖန်တီးပြီးဖြစ်သည်။ Pumpkin အမျိုးအစားအားလုံးအတွက် ထပ်မံလုပ်ဆောင်နိုင်သော်လည်း ၎င်းသည် ဝေဒနာဖြစ်ပါမည်။ ယခု ကိစ္စအား model တွင် pumpkin အမျိုးအစားကို ထည့်သွင်းရန် သင်ယူကြမည်။

## Categorical Features

Ideal လူမှုပညာတွင်၊ မတူညီသော pumpkin မျိုးစုံအတွက် တူညီသော မော်ဒယ်ဖြင့် စျေးနှုန်း ခန့်မှန်းနိုင်ရမည်။ သို့သော် `Variety` ကော်လံသည် `Month` ကဲ့သို့ တိုက်ရိုက်ဂဏန်းမဟုတ်ပဲ၊ ကွဲလွဲသည့် တန်ဖိုးများပါဝင်သည်။ ထိုကော်လံများကို **categorical** ဟုခေါ်သည်။

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 အထက်ပါ ပုံကို နှိပ်၍ categorical feature များ အသုံးပြုခြင်း၏ ဗီဒီယိုအနှစ်ချုပ်ကို ကြည့်ရှုပါ။

Variety အလိုက် ပျမ်းမျှစျေးနှုန်းသည် အောက်ပါအတိုင်း ဖြစ်သည် -

<img alt="Average price by variety" src="../../../../translated_images/my/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Variety ကို တွက်ချက်ရန်၊ ပထမဆုံး ဂဏန်းပုံစံသို့ ပြောင်းလဲသို့မဟုတ် **encode** ရပါမည်။ ၎င်းကို ပြုလုပ်နိုင်သည့် နည်းလမ်းများ အချို့မှာ -

* ရိုးရှင်းသော **numeric encoding** သည် မတူညီသော variety များ စာရင်းဆွဲ၍ အဲ့သည့် စာရင်းအတွင်း အညွှန်းတစ်ခုဖြင့် variety အမည်ကို တပ်ဆင်သည်။ ဤနည်းပညာသည် linear regression အတွက် အကောင်းဆုံး မဟုတ်ပါ၊ တကယ်လျှင် linear regression သည် အညွှန်း၏ တန်ဖိုးကို သုံးပြီး coefficient တစ်ခုထပ်ဖွဲ့သည်။ ကျွန်ုပ်တို့ကိစ္စတွင် အညွှန်းနံပါတ်နှင့် စျေးနှုန်း ဆက်စပ်မှုသည် nonlinear ဖြစ်ပြီး၊ အညွှန်းများ စနစ်တကျ အစီအစဉ်လိုက်ဖြစ်စေသော်လည်း ဖြစ်သည်။
* **One-hot encoding** သည် `Variety` ကော်လံကို မျိုးစုံ ၄ ကော်လံသို့ အစားထိုးပြီး၊ မျိုးစုံတစ်ခုစီအတွက် ကော်လံတစ်ခုစီ ထားသည်။ ဝေငှားစုရင်းတစ်ကြောင်းသည် ထိုမျိုးစုံဖြစ်လျှင် `1` ဖြစ်မည်၊ မဟုတ်လျှင် `0` ဖြစ်မည်။ ၎င်းဖြင့် linear regression တွင် ကိုယ်စားလှယ် ၄ ယောက်ရှိမည်ဖြစ်ပြီး၊ pumpkin မျိုးစုံ တစ်ခုချင်းစီအတွက် စျေးအစ၊ ဒါမှမဟုတ် "အပေါင်းစုပေးငွေ" အတွက် တစ်ခုစီကို တာဝန်ယူမည်။

အောက်ပါ ကုဒ်သည် variety ကို one-hot encode ပြုလုပ်ပုံကို ပြသည် -

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

one-hot encode တဲ့ variety ကို input အဖြစ်သုံး၍ linear regression လေ့ကျင့်ရန်၊ `X` နှင့် `y` အချက်အလက်များကို မှန်ကန်စွာ စတင်ရပါမည်။

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```
  
အကြောင်းအရာများသည် အပေါ်က Linear Regression ခေတ်မီပုံစံနှင့် ထပ်တူတူဖြစ်သည်။ စမ်းကြည့်လျှင် mean squared error မျှ ၎င်းနှင့်အတူတူ နီးပါး ဖြစ်ပြီး coefficient of determination ပြင်းထန်စွာ မြင့်သည် ( ~77%) ဟု တွေ့ရမည်။ ပိုမိုတိကျသော ခန့်မှန်းချက်များရရန် categorical features အပြင် numeric features များ (ဥပမာ `Month`၊ `DayOfYear`) ကိုပါ ထည့်သွင်းနိုင်ပြီး၊ feature များ စုစည်းရန် `join` ကို အသုံးပြုနိုင်သည် -

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```
  
ဒီမှာ `City` နှင့် `Package` အမျိုးအစားများကိုပါ ထည့်သွင်းပြီး၊ မည်သည့် MSE 2.84 (10%) နှင့် determination 0.94 ရရှိသည်။

## အားလုံးကို ပေါင်းစပ်ခြင်း

အကောင်းဆုံး မော်ဒယ်ကို မိတ်ဆက်ရန်၊ အပေါ်ပါ ဥပမာ၏ combined (one-hot encoded categorical + numeric) data ကို Polynomial Regression နှင့် တွဲသုံးရပါမည်။ ကျွန်ုပ်တို့၏ တူညီမှုအတွက် အပြည့်အစုံကုဒ် -

```python
# သင်ကြားရေးဒေတာကို စီစဉ်ပါ
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# သင်ကြားမှုနှင့် စမ်းသပ်မှု ခွဲခြားပါ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ပိုင်လိုင်းအား စီစဉ်ပြီး သင်ကြားပါ
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# စမ်းသပ်ဒေတာအတွက် ခန့်မှန်းရလဒ်များ ထုတ်ပါ
pred = pipeline.predict(X_test)

# MSE နှင့် ဆုံးဖြတ်ချက်ကို တွက်ချက်ပါ
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```
  
ဤသည်မှာ determination coefficient အများဆုံး ၉၇% နီးပါး၊ MSE=2.23 (ခန့်မှန်းချက်အမှား ၈% လောက်) ပေးမည်။

| Model | MSE | Determination |  
|-------|-----|---------------|  
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |  
| `Variety` Linear | 5.24 (19.7%) | 0.77 |  
| All features Linear | 2.84 (10.5%) | 0.94 |  
| All features Polynomial | 2.23 (8.25%) | 0.97 |

🏆 ကောင်းမွန်ပါတယ်! သင်သည် Regression မော်ဒယ် ၄ ခုကို သင်ခန်းစာတစ်ခုတွင်တည်း ဖန်တီးပြီး မော်ဒယ်အရည်အသွေးကို ၉၇% ထိ မြှင့်တင်ခဲ့ပါသည်။ Regression နောက်ဆုံးအပိုင်းတွင် မည်သည့် category များဖြစ်သည်ကို သတ်မှတ်ပေးသည့် Logistic Regression ကို သင်ကြားမည် ဖြစ်သည်။

---  
## 🚀 စိန်ခေါ်မှု

ဒီ notebook တွင် မတူညီသော variables များကို စမ်းသပ်ပါ။ Correlation နှင့် model တိကျမှု ယှဉ်တွဲမှုကို ကြည့်ရှုပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာမှု

ဒီသင်ခန်းစာတွင် Linear Regression အကြောင်း သင်ယူခဲ့ပါသည်။ ထိုသို့ အရေးကြီးသော Regression အမျိုးအစားအခြားများလည်း ရှိပါသည်။ Stepwise, Ridge, Lasso နှင့် Elasticnet နည်းစနစ်များကိုလေ့လာပါ။ ပိုမိုလေ့လာရန် Stanford Statistical Learning သင်တန်းရုပ်သံသည် အကောင်းဆုံးဖြစ်သည်။ [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## တာဝန်လက်ခံခြင်း

[Build a Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ချက်ပြောချက်**  
ဤစာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) များဖြင့် ဘာသာပြန်ထားပါသည်။ တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း အလိုအလျောက်ဘာသာပြန်မှုများတွင် မှားယွင်းချက်များ သို့မဟုတ် တိကျမှုမရှိနိုင်မှုများ ရှိနိုင်သည်ကို သတိပြုပါရန် မေတ္တာရပ်ခံအပ်ပါသည်။ မူရင်းစာတမ်းကို ၎င်း၏ မူလဘာသာဖြင့်သာ ယုံကြည်စိတ်ချရသော အချက်အလက်အရင်းအမြစ်အဖြစ် ထင်မှတ်သင့်ပါသည်။ အရေးကြီးသောသတင်းအချက်အလက်များအတွက် စစ်မှန်သော လူကြီးမင်းတို့၏ ဘာသာပြန်သူများက ဘာသာပြန်ပေးခြင်းကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်ချက်ကို အသုံးပြုရာမှ ဖြစ်ပေါ်လာနိုင်သည့် ပဋိပက္ခ သို့မဟုတ် အထူးသဖြင့် နားလည်သဘောပေါက်မှု မမှန်ကန်မှုများအတွက် ကျွန်ုပ်တို့တာဝန်မရှိပါ။
<!-- CO-OP TRANSLATOR DISCLAIMER END -->