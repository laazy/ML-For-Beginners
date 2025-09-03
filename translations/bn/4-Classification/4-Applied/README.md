<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-08-29T21:45:01+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "bn"
}
-->
# একটি কুইজিন রিকমেন্ডার ওয়েব অ্যাপ তৈরি করুন

এই পাঠে, আপনি পূর্ববর্তী পাঠে শেখা কিছু কৌশল ব্যবহার করে একটি শ্রেণীবিন্যাস মডেল তৈরি করবেন এবং এই সিরিজে ব্যবহৃত সুস্বাদু কুইজিন ডেটাসেট ব্যবহার করবেন। এছাড়াও, আপনি একটি ছোট ওয়েব অ্যাপ তৈরি করবেন যা সংরক্ষিত মডেল ব্যবহার করবে, Onnx-এর ওয়েব রানটাইম ব্যবহার করে।

মেশিন লার্নিংয়ের অন্যতম কার্যকর ব্যবহার হল রিকমেন্ডেশন সিস্টেম তৈরি করা, এবং আপনি আজ সেই দিকে প্রথম পদক্ষেপ নিতে পারেন!

[![এই ওয়েব অ্যাপটি উপস্থাপন করা হচ্ছে](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 উপরের ছবিতে ক্লিক করুন একটি ভিডিওর জন্য: জেন লুপার শ্রেণীবিন্যাস করা কুইজিন ডেটা ব্যবহার করে একটি ওয়েব অ্যাপ তৈরি করছেন

## [পাঠ-পূর্ব কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

এই পাঠে আপনি শিখবেন:

- কীভাবে একটি মডেল তৈরি এবং এটি Onnx মডেল হিসেবে সংরক্ষণ করবেন
- কীভাবে Netron ব্যবহার করে মডেল পরিদর্শন করবেন
- কীভাবে আপনার মডেল একটি ওয়েব অ্যাপে ইনফারেন্সের জন্য ব্যবহার করবেন

## আপনার মডেল তৈরি করুন

প্রয়োগকৃত মেশিন লার্নিং সিস্টেম তৈরি করা আপনার ব্যবসায়িক সিস্টেমের জন্য এই প্রযুক্তিগুলি ব্যবহার করার একটি গুরুত্বপূর্ণ অংশ। আপনি আপনার ওয়েব অ্যাপ্লিকেশনের মধ্যে মডেল ব্যবহার করতে পারেন (এবং প্রয়োজনে অফলাইন প্রসঙ্গে ব্যবহার করতে পারেন) Onnx ব্যবহার করে।

একটি [পূর্ববর্তী পাঠে](../../3-Web-App/1-Web-App/README.md), আপনি UFO দর্শন সম্পর্কিত একটি রিগ্রেশন মডেল তৈরি করেছিলেন, এটি "পিকল" করেছিলেন এবং একটি Flask অ্যাপে ব্যবহার করেছিলেন। যদিও এই আর্কিটেকচারটি জানা খুবই উপকারী, এটি একটি ফুল-স্ট্যাক পাইথন অ্যাপ, এবং আপনার প্রয়োজনীয়তাগুলি একটি জাভাস্ক্রিপ্ট অ্যাপ্লিকেশন অন্তর্ভুক্ত করতে পারে।

এই পাঠে, আপনি ইনফারেন্সের জন্য একটি বেসিক জাভাস্ক্রিপ্ট-ভিত্তিক সিস্টেম তৈরি করতে পারেন। তবে প্রথমে, আপনাকে একটি মডেল প্রশিক্ষণ দিতে হবে এবং এটি Onnx-এ রূপান্তর করতে হবে।

## অনুশীলন - শ্রেণীবিন্যাস মডেল প্রশিক্ষণ দিন

প্রথমে, আমরা পূর্বে ব্যবহৃত পরিষ্কার করা কুইজিন ডেটাসেট ব্যবহার করে একটি শ্রেণীবিন্যাস মডেল প্রশিক্ষণ দেব।

1. প্রথমে দরকারী লাইব্রেরি ইমপোর্ট করুন:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    আপনাকে '[skl2onnx](https://onnx.ai/sklearn-onnx/)' প্রয়োজন যা আপনার Scikit-learn মডেলকে Onnx ফরম্যাটে রূপান্তর করতে সাহায্য করবে।

1. তারপর, পূর্ববর্তী পাঠের মতো আপনার ডেটা নিয়ে কাজ করুন, একটি CSV ফাইল `read_csv()` ব্যবহার করে পড়ুন:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. প্রথম দুটি অপ্রয়োজনীয় কলাম সরিয়ে ফেলুন এবং বাকি ডেটা 'X' হিসেবে সংরক্ষণ করুন:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. লেবেলগুলো 'y' হিসেবে সংরক্ষণ করুন:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### প্রশিক্ষণ প্রক্রিয়া শুরু করুন

আমরা 'SVC' লাইব্রেরি ব্যবহার করব যা ভালো নির্ভুলতা প্রদান করে।

1. Scikit-learn থেকে প্রাসঙ্গিক লাইব্রেরি ইমপোর্ট করুন:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. প্রশিক্ষণ এবং টেস্ট সেট আলাদা করুন:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. পূর্ববর্তী পাঠের মতো একটি SVC শ্রেণীবিন্যাস মডেল তৈরি করুন:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. এখন, আপনার মডেল পরীক্ষা করুন, `predict()` কল করে:

    ```python
    y_pred = model.predict(X_test)
    ```

1. মডেলের গুণমান পরীক্ষা করতে একটি শ্রেণীবিন্যাস রিপোর্ট প্রিন্ট করুন:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    যেমন আমরা আগে দেখেছি, নির্ভুলতা ভালো:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### আপনার মডেল Onnx-এ রূপান্তর করুন

সঠিক টেনসর সংখ্যা দিয়ে রূপান্তর নিশ্চিত করুন। এই ডেটাসেটে ৩৮০টি উপাদান তালিকাভুক্ত রয়েছে, তাই আপনাকে `FloatTensorType`-এ সেই সংখ্যা উল্লেখ করতে হবে:

1. ৩৮০ টেনসর সংখ্যা ব্যবহার করে রূপান্তর করুন।

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. একটি onx তৈরি করুন এবং **model.onnx** নামে একটি ফাইল হিসেবে সংরক্ষণ করুন:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > লক্ষ্য করুন, আপনি আপনার রূপান্তর স্ক্রিপ্টে [অপশন](https://onnx.ai/sklearn-onnx/parameterized.html) পাস করতে পারেন। এই ক্ষেত্রে, আমরা 'nocl' কে True এবং 'zipmap' কে False পাস করেছি। যেহেতু এটি একটি শ্রেণীবিন্যাস মডেল, আপনি ZipMap সরানোর অপশন পাবেন যা একটি ডিকশনারির তালিকা তৈরি করে (যা প্রয়োজনীয় নয়)। `nocl` মডেলে শ্রেণী তথ্য অন্তর্ভুক্ত হওয়ার বিষয়টি নির্দেশ করে। `nocl`-কে 'True' সেট করে আপনার মডেলের আকার কমান।

পুরো নোটবুক চালালে এখন একটি Onnx মডেল তৈরি হবে এবং এটি এই ফোল্ডারে সংরক্ষণ হবে।

## আপনার মডেল দেখুন

Onnx মডেলগুলো Visual Studio Code-এ খুব বেশি দৃশ্যমান নয়, তবে একটি খুব ভালো ফ্রি সফটওয়্যার রয়েছে যা অনেক গবেষক মডেল ভিজ্যুয়ালাইজ করতে ব্যবহার করেন। [Netron](https://github.com/lutzroeder/Netron) ডাউনলোড করুন এবং আপনার model.onnx ফাইলটি খুলুন। আপনি আপনার সহজ মডেলটি ভিজ্যুয়ালাইজ করতে পারবেন, যেখানে এর ৩৮০টি ইনপুট এবং শ্রেণীবিন্যাসকারী তালিকাভুক্ত রয়েছে:

![Netron ভিজ্যুয়াল](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.bn.png)

Netron আপনার মডেলগুলো দেখার জন্য একটি সহায়ক টুল।

এখন আপনি এই চমৎকার মডেলটি একটি ওয়েব অ্যাপে ব্যবহার করতে প্রস্তুত। চলুন একটি অ্যাপ তৈরি করি যা আপনার ফ্রিজে থাকা উপাদানগুলোর সংমিশ্রণ থেকে কোন কুইজিন রান্না করা যায় তা নির্ধারণ করতে সাহায্য করবে, আপনার মডেল অনুযায়ী।

## একটি রিকমেন্ডার ওয়েব অ্যাপ্লিকেশন তৈরি করুন

আপনার মডেল সরাসরি একটি ওয়েব অ্যাপে ব্যবহার করা যেতে পারে। এই আর্কিটেকচারটি আপনাকে এটি লোকাল এবং এমনকি প্রয়োজনে অফলাইনে চালানোর সুযোগ দেয়। যেখানে আপনি আপনার `model.onnx` ফাইল সংরক্ষণ করেছেন সেই একই ফোল্ডারে একটি `index.html` ফাইল তৈরি করে শুরু করুন।

1. এই ফাইল _index.html_-এ, নিম্নলিখিত মার্কআপ যোগ করুন:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. এখন, `body` ট্যাগের মধ্যে কাজ করে, কিছু উপাদান দেখানোর জন্য একটি চেকবক্সের তালিকা যোগ করুন:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    লক্ষ্য করুন যে প্রতিটি চেকবক্সকে একটি মান দেওয়া হয়েছে। এটি ডেটাসেট অনুযায়ী উপাদানটি যেখানে পাওয়া যায় সেই ইনডেক্সকে প্রতিফলিত করে। উদাহরণস্বরূপ, আপেল এই বর্ণানুক্রমিক তালিকায় পঞ্চম কলামে রয়েছে, তাই এর মান '4', কারণ আমরা 0 থেকে গণনা শুরু করি। একটি নির্দিষ্ট উপাদানের ইনডেক্স জানতে [উপাদান স্প্রেডশিট](../../../../4-Classification/data/ingredient_indexes.csv) পরামর্শ করুন।

    index.html ফাইলে আপনার কাজ চালিয়ে যান এবং একটি স্ক্রিপ্ট ব্লক যোগ করুন যেখানে মডেলটি চূড়ান্ত বন্ধ `</div>` এর পরে কল করা হয়।

1. প্রথমে, [Onnx Runtime](https://www.onnxruntime.ai/) ইমপোর্ট করুন:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime ব্যবহার করা হয় আপনার Onnx মডেলগুলোকে বিভিন্ন হার্ডওয়্যার প্ল্যাটফর্মে চালানোর জন্য, অপ্টিমাইজেশন এবং একটি API সহ।

1. একবার Runtime ইনপ্লেসে থাকলে, আপনি এটি কল করতে পারেন:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

এই কোডে, কয়েকটি বিষয় ঘটছে:

1. আপনি ৩৮০ সম্ভাব্য মানের (১ বা ০) একটি অ্যারে তৈরি করেছেন যা ইনফারেন্সের জন্য মডেলে পাঠানোর জন্য সেট করা হবে, নির্ভর করে একটি উপাদান চেকবক্স চেক করা হয়েছে কিনা।
2. আপনি চেকবক্সের একটি অ্যারে এবং একটি `init` ফাংশন তৈরি করেছেন যা অ্যাপ্লিকেশন শুরু হলে কল করা হয়। যখন একটি চেকবক্স চেক করা হয়, তখন `ingredients` অ্যারে সংশোধিত হয় নির্বাচিত উপাদানটি প্রতিফলিত করতে।
3. আপনি একটি `testCheckboxes` ফাংশন তৈরি করেছেন যা পরীক্ষা করে যে কোনো চেকবক্স চেক করা হয়েছে কিনা।
4. আপনি `startInference` ফাংশন ব্যবহার করেন যখন বোতামটি চাপা হয় এবং যদি কোনো চেকবক্স চেক করা হয়, আপনি ইনফারেন্স শুরু করেন।
5. ইনফারেন্স রুটিন অন্তর্ভুক্ত করে:
   1. মডেলটি অ্যাসিঙ্ক্রোনাস লোড সেটআপ করা
   2. মডেলে পাঠানোর জন্য একটি টেনসর স্ট্রাকচার তৈরি করা
   3. 'ফিডস' তৈরি করা যা আপনি মডেল প্রশিক্ষণ করার সময় তৈরি করা `float_input` ইনপুটকে প্রতিফলিত করে (আপনি Netron ব্যবহার করে সেই নামটি যাচাই করতে পারেন)
   4. এই 'ফিডস' মডেলে পাঠানো এবং একটি প্রতিক্রিয়ার জন্য অপেক্ষা করা

## আপনার অ্যাপ্লিকেশন পরীক্ষা করুন

Visual Studio Code-এ একটি টার্মিনাল সেশন খুলুন যেখানে আপনার index.html ফাইলটি রয়েছে। নিশ্চিত করুন যে আপনার [http-server](https://www.npmjs.com/package/http-server) গ্লোবালি ইনস্টল করা আছে এবং প্রম্পটে `http-server` টাইপ করুন। একটি লোকালহোস্ট খুলবে এবং আপনি আপনার ওয়েব অ্যাপ দেখতে পারবেন। বিভিন্ন উপাদানের উপর ভিত্তি করে কোন কুইজিন সুপারিশ করা হচ্ছে তা পরীক্ষা করুন:

![উপাদান ওয়েব অ্যাপ](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.bn.png)

অভিনন্দন, আপনি কয়েকটি ফিল্ড সহ একটি 'রিকমেন্ডেশন' ওয়েব অ্যাপ তৈরি করেছেন। এই সিস্টেমটি তৈরি করতে কিছু সময় নিন!

## 🚀চ্যালেঞ্জ

আপনার ওয়েব অ্যাপটি খুবই সাধারণ, তাই [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) ডেটা থেকে উপাদান এবং তাদের ইনডেক্স ব্যবহার করে এটি আরও তৈরি করুন। কোন স্বাদ সংমিশ্রণ একটি নির্দিষ্ট জাতীয় খাবার তৈরি করতে কাজ করে?

## [পাঠ-পরবর্তী কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## পর্যালোচনা ও স্ব-অধ্যয়ন

যদিও এই পাঠটি খাদ্য উপাদানের জন্য একটি রিকমেন্ডেশন সিস্টেম তৈরির উপযোগিতার উপর শুধু স্পর্শ করেছে, মেশিন লার্নিং অ্যাপ্লিকেশনের এই ক্ষেত্রটি উদাহরণে খুব সমৃদ্ধ। এই সিস্টেমগুলো কীভাবে তৈরি হয় সে সম্পর্কে আরও পড়ুন:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## অ্যাসাইনমেন্ট

[একটি নতুন রিকমেন্ডার তৈরি করুন](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসাধ্য সঠিকতা নিশ্চিত করার চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।