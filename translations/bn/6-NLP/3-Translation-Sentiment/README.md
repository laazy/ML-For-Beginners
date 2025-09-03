<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-08-29T22:34:04+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "bn"
}
-->
# মেশিন লার্নিং দিয়ে অনুবাদ এবং অনুভূতি বিশ্লেষণ

পূর্ববর্তী পাঠে আপনি `TextBlob` ব্যবহার করে একটি সাধারণ বট তৈরি করতে শিখেছেন, যা একটি লাইব্রেরি যা মৌলিক NLP কাজ যেমন noun phrase extraction সম্পাদন করতে পর্দার আড়ালে মেশিন লার্নিং ব্যবহার করে। কম্পিউটেশনাল ভাষাতত্ত্বের আরেকটি গুরুত্বপূর্ণ চ্যালেঞ্জ হলো একটি ভাষা থেকে অন্য ভাষায় সঠিকভাবে _অনুবাদ_ করা।

## [পূর্ব-পাঠ কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

অনুবাদ একটি অত্যন্ত কঠিন সমস্যা, কারণ পৃথিবীতে হাজার হাজার ভাষা রয়েছে এবং প্রতিটির ব্যাকরণ নিয়ম ভিন্ন হতে পারে। একটি পদ্ধতি হলো একটি ভাষার (যেমন ইংরেজি) আনুষ্ঠানিক ব্যাকরণ নিয়মকে একটি ভাষা-নিরপেক্ষ কাঠামোতে রূপান্তর করা এবং তারপর অন্য ভাষায় রূপান্তর করা। এই পদ্ধতিতে নিম্নলিখিত ধাপগুলো সম্পন্ন করতে হয়:

1. **সনাক্তকরণ**। ইনপুট ভাষার শব্দগুলোকে noun, verb ইত্যাদি হিসেবে চিহ্নিত বা ট্যাগ করা।
2. **অনুবাদ তৈরি করা**। লক্ষ্য ভাষার ফরম্যাটে প্রতিটি শব্দের সরাসরি অনুবাদ তৈরি করা।

### উদাহরণ বাক্য, ইংরেজি থেকে আইরিশ

ইংরেজি ভাষায়, বাক্য _I feel happy_ তিনটি শব্দ নিয়ে গঠিত এবং এর ক্রম হলো:

- **subject** (I)
- **verb** (feel)
- **adjective** (happy)

তবে, আইরিশ ভাষায় একই বাক্যের ব্যাকরণ কাঠামো সম্পূর্ণ ভিন্ন - "*happy*" বা "*sad*" এর মতো আবেগগুলোকে *upon* (উপর) হিসেবে প্রকাশ করা হয়।

ইংরেজি বাক্য `I feel happy` আইরিশ ভাষায় হবে `Tá athas orm`। একটি *শব্দগত* অনুবাদ হবে `Happy is upon me`।

একজন আইরিশ ভাষাভাষী ইংরেজিতে অনুবাদ করলে বলবে `I feel happy`, `Happy is upon me` নয়, কারণ তারা বাক্যের অর্থ বুঝতে পারে, যদিও শব্দ এবং বাক্য কাঠামো ভিন্ন।

আইরিশ ভাষায় বাক্যের আনুষ্ঠানিক ক্রম হলো:

- **verb** (Tá বা is)
- **adjective** (athas বা happy)
- **subject** (orm বা upon me)

## অনুবাদ

একটি সরল অনুবাদ প্রোগ্রাম শুধুমাত্র শব্দ অনুবাদ করতে পারে, বাক্য কাঠামো উপেক্ষা করে।

✅ যদি আপনি একজন প্রাপ্তবয়স্ক হিসেবে দ্বিতীয় (বা তৃতীয় বা আরও বেশি) ভাষা শিখে থাকেন, তাহলে আপনি হয়তো আপনার মাতৃভাষায় চিন্তা করে, শব্দ ধরে ধরে দ্বিতীয় ভাষায় অনুবাদ করে এবং তারপর আপনার অনুবাদটি উচ্চারণ করে শুরু করেছিলেন। এটি সরল অনুবাদ কম্পিউটার প্রোগ্রামগুলোর কাজের মতো। সাবলীলতা অর্জনের জন্য এই পর্যায়টি অতিক্রম করা গুরুত্বপূর্ণ!

সরল অনুবাদ খারাপ (এবং কখনও কখনও হাস্যকর) ভুল অনুবাদ তৈরি করে: `I feel happy` আইরিশ ভাষায় শব্দগতভাবে অনুবাদ করলে হয় `Mise bhraitheann athas`। এর অর্থ (শব্দগতভাবে) `me feel happy` এবং এটি একটি বৈধ আইরিশ বাক্য নয়। যদিও ইংরেজি এবং আইরিশ দুটি ভাষা দুটি কাছাকাছি দ্বীপে কথিত হয়, এগুলো খুবই ভিন্ন ভাষা এবং ব্যাকরণ কাঠামোও ভিন্ন।

> আপনি আইরিশ ভাষার ঐতিহ্য সম্পর্কে কিছু ভিডিও দেখতে পারেন যেমন [এই ভিডিওটি](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### মেশিন লার্নিং পদ্ধতি

এখন পর্যন্ত, আপনি প্রাকৃতিক ভাষা প্রক্রিয়াকরণের আনুষ্ঠানিক নিয়ম পদ্ধতি সম্পর্কে শিখেছেন। আরেকটি পদ্ধতি হলো শব্দের অর্থ উপেক্ষা করা এবং _মেশিন লার্নিং ব্যবহার করে প্যাটার্ন সনাক্ত করা_। যদি আপনার কাছে প্রচুর টেক্সট (একটি *corpus*) বা টেক্সটসমূহ (*corpora*) থাকে উত্স এবং লক্ষ্য ভাষায়, তাহলে এটি অনুবাদে কাজ করতে পারে।

উদাহরণস্বরূপ, *Pride and Prejudice*, একটি বিখ্যাত ইংরেজি উপন্যাস যা ১৮১৩ সালে জেন অস্টেন লিখেছেন। যদি আপনি বইটি ইংরেজিতে এবং এর একটি মানব অনুবাদ *ফরাসি* ভাষায় পরামর্শ করেন, তাহলে আপনি একটিতে এমন বাক্যাংশ সনাক্ত করতে পারেন যা অন্যটিতে _idiomatically_ অনুবাদ করা হয়েছে। আপনি এটি কিছুক্ষণের মধ্যে করবেন।

উদাহরণস্বরূপ, যখন একটি ইংরেজি বাক্যাংশ `I have no money` শব্দগতভাবে ফরাসি ভাষায় অনুবাদ করা হয়, এটি হতে পারে `Je n'ai pas de monnaie`। "Monnaie" একটি জটিল ফরাসি 'false cognate', কারণ 'money' এবং 'monnaie' সমার্থক নয়। একজন মানব অনুবাদক একটি ভালো অনুবাদ করতে পারে `Je n'ai pas d'argent`, কারণ এটি ভালোভাবে বোঝায় যে আপনার কোনো টাকা নেই (বরং 'loose change' যা 'monnaie' এর অর্থ)।

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.bn.png)

> ছবি: [Jen Looper](https://twitter.com/jenlooper)

যদি একটি ML মডেলের কাছে পর্যাপ্ত মানব অনুবাদ থাকে যার উপর ভিত্তি করে একটি মডেল তৈরি করা যায়, এটি পূর্বে উভয় ভাষার দক্ষ মানব বক্তাদের দ্বারা অনুবাদিত টেক্সটগুলোর সাধারণ প্যাটার্ন সনাক্ত করে অনুবাদের সঠিকতা উন্নত করতে পারে।

### অনুশীলন - অনুবাদ

আপনি `TextBlob` ব্যবহার করে বাক্য অনুবাদ করতে পারেন। **Pride and Prejudice** এর বিখ্যাত প্রথম লাইনটি চেষ্টা করুন:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` অনুবাদটি বেশ ভালোভাবে করে: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

এটি বলা যেতে পারে যে TextBlob এর অনুবাদটি ১৯৩২ সালে V. Leconte এবং Ch. Pressoir এর ফরাসি অনুবাদের চেয়ে অনেক বেশি সঠিক:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

এই ক্ষেত্রে, ML দ্বারা পরিচালিত অনুবাদটি মানব অনুবাদকের চেয়ে ভালো কাজ করেছে, যিনি অপ্রয়োজনীয়ভাবে মূল লেখকের বক্তব্যে অতিরিক্ত শব্দ যোগ করেছেন 'স্পষ্টতার' জন্য।

> এখানে কী ঘটছে? এবং কেন TextBlob অনুবাদে এত ভালো? আসলে, এটি Google Translate ব্যবহার করছে, একটি উন্নত AI যা লক্ষ লক্ষ বাক্যাংশ বিশ্লেষণ করতে পারে এবং নির্দিষ্ট কাজের জন্য সেরা স্ট্রিংগুলি পূর্বাভাস দিতে পারে। এখানে কিছুই ম্যানুয়াল নয় এবং `blob.translate` ব্যবহার করতে আপনার ইন্টারনেট সংযোগ প্রয়োজন।

✅ আরও কিছু বাক্য চেষ্টা করুন। কোনটি ভালো, ML না মানব অনুবাদ? কোন ক্ষেত্রে?

## অনুভূতি বিশ্লেষণ

মেশিন লার্নিং খুব ভালোভাবে কাজ করতে পারে এমন আরেকটি ক্ষেত্র হলো অনুভূতি বিশ্লেষণ। একটি non-ML পদ্ধতি হলো 'positive' এবং 'negative' শব্দ এবং বাক্যাংশ সনাক্ত করা। তারপর, একটি নতুন টেক্সট দেওয়া হলে, positive, negative এবং neutral শব্দগুলোর মোট মান গণনা করে সামগ্রিক অনুভূতি সনাক্ত করা। 

এই পদ্ধতি সহজেই বিভ্রান্ত হতে পারে যেমন আপনি Marvin টাস্কে দেখেছেন - বাক্যটি `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` একটি ব্যঙ্গাত্মক, negative অনুভূতির বাক্য, কিন্তু সহজ অ্যালগরিদমটি 'great', 'wonderful', 'glad' কে positive এবং 'waste', 'lost' এবং 'dark' কে negative হিসেবে সনাক্ত করে। সামগ্রিক অনুভূতি এই বিরোধপূর্ণ শব্দগুলোর দ্বারা প্রভাবিত হয়।

✅ এক মুহূর্ত থামুন এবং ভাবুন আমরা মানুষ হিসেবে কীভাবে ব্যঙ্গ প্রকাশ করি। স্বরের ওঠানামা একটি বড় ভূমিকা পালন করে। "Well, that film was awesome" বাক্যটি বিভিন্নভাবে বলার চেষ্টা করুন এবং দেখুন কীভাবে আপনার কণ্ঠ অর্থ প্রকাশ করে।

### ML পদ্ধতি

ML পদ্ধতি হলো negative এবং positive টেক্সটের সংগ্রহ - টুইট, বা মুভি রিভিউ, বা যেকোনো কিছু যেখানে মানুষ একটি স্কোর *এবং* একটি লিখিত মতামত দিয়েছে। তারপর NLP কৌশলগুলো মতামত এবং স্কোরের উপর প্রয়োগ করা যায়, যাতে প্যাটার্নগুলো প্রকাশ পায় (যেমন, positive মুভি রিভিউতে 'Oscar worthy' বাক্যাংশটি negative মুভি রিভিউয়ের চেয়ে বেশি থাকে, বা positive রেস্টুরেন্ট রিভিউতে 'gourmet' শব্দটি 'disgusting' এর চেয়ে বেশি থাকে)।

> ⚖️ **উদাহরণ**: ধরুন আপনি একজন রাজনীতিবিদের অফিসে কাজ করছেন এবং একটি নতুন আইন নিয়ে বিতর্ক চলছে, জনগণ অফিসে ইমেইল লিখে আইনটির পক্ষে বা বিপক্ষে মতামত জানাচ্ছে। ধরুন আপনাকে ইমেইলগুলো পড়তে এবং দুটি ভাগে সাজাতে বলা হলো, *পক্ষে* এবং *বিপক্ষে*। যদি প্রচুর ইমেইল থাকে, তাহলে আপনি সবগুলো পড়তে গিয়ে অভিভূত হতে পারেন। যদি একটি বট সবগুলো ইমেইল পড়ে, বুঝতে পারে এবং কোন ভাগে ইমেইলটি রাখা উচিত তা বলে দেয়, তাহলে কি ভালো হতো না? 
> 
> এটি অর্জনের একটি উপায় হলো মেশিন লার্নিং ব্যবহার করা। আপনি মডেলটি কিছু *বিপক্ষে* ইমেইল এবং কিছু *পক্ষে* ইমেইল দিয়ে প্রশিক্ষণ দেবেন। মডেলটি শব্দ এবং প্যাটার্নগুলোকে বিপক্ষে বা পক্ষে ইমেইলের সাথে যুক্ত করতে পারে, *কিন্তু এটি কোনো বিষয়বস্তু বুঝবে না*, শুধুমাত্র নির্দিষ্ট শব্দ এবং প্যাটার্নগুলো বিপক্ষে বা পক্ষে ইমেইলে বেশি দেখা যায়। আপনি এটি এমন কিছু ইমেইল দিয়ে পরীক্ষা করতে পারেন যা আপনি মডেল প্রশিক্ষণে ব্যবহার করেননি এবং দেখুন এটি আপনার সিদ্ধান্তের সাথে একমত কিনা। তারপর, যখন আপনি মডেলের সঠিকতায় সন্তুষ্ট হবেন, তখন ভবিষ্যতের ইমেইলগুলো প্রক্রিয়া করতে পারবেন প্রতিটি ইমেইল না পড়ে।

✅ এই প্রক্রিয়াটি কি আপনি পূর্ববর্তী পাঠে ব্যবহৃত কোনো প্রক্রিয়ার মতো মনে হয়?

## অনুশীলন - অনুভূতিমূলক বাক্য

অনুভূতি *polarity* -১ থেকে ১ এর মধ্যে পরিমাপ করা হয়, যেখানে -১ সবচেয়ে negative অনুভূতি এবং ১ সবচেয়ে positive। অনুভূতি ০ - ১ স্কোর দিয়ে objectivity (০) এবং subjectivity (১) হিসেবেও পরিমাপ করা হয়।

জেন অস্টেনের *Pride and Prejudice* এর দিকে আবার তাকান। টেক্সটটি এখানে পাওয়া যাবে [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)। নিচের নমুনাটি একটি ছোট প্রোগ্রাম দেখায় যা বইয়ের প্রথম এবং শেষ বাক্যের অনুভূতি বিশ্লেষণ করে এবং এর sentiment polarity এবং subjectivity/objectivity স্কোর প্রদর্শন করে।

আপনাকে `TextBlob` লাইব্রেরি (উপরে বর্ণিত) ব্যবহার করে `sentiment` নির্ধারণ করতে হবে (আপনার নিজস্ব sentiment calculator লিখতে হবে না) নিম্নলিখিত কাজে।

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

আপনি নিম্নলিখিত আউটপুট দেখতে পাবেন:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## চ্যালেঞ্জ - sentiment polarity পরীক্ষা করুন

আপনার কাজ হলো sentiment polarity ব্যবহার করে নির্ধারণ করা যে *Pride and Prejudice* এ absolutely positive বাক্যগুলো absolutely negative বাক্যগুলোর চেয়ে বেশি কিনা। এই কাজের জন্য, আপনি ধরে নিতে পারেন যে polarity স্কোর ১ বা -১ absolutely positive বা negative যথাক্রমে।

**ধাপসমূহ:**

1. [Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) এর একটি কপি Project Gutenberg থেকে .txt ফাইল হিসেবে ডাউনলোড করুন। ফাইলের শুরু এবং শেষের মেটাডেটা সরিয়ে শুধুমাত্র মূল টেক্সট রেখে দিন।
2. ফাইলটি Python এ খুলুন এবং এর বিষয়বস্তু একটি string হিসেবে বের করুন।
3. বই string ব্যবহার করে একটি TextBlob তৈরি করুন।
4. একটি লুপে বইয়ের প্রতিটি বাক্য বিশ্লেষণ করুন।
   1. যদি polarity ১ বা -১ হয়, তাহলে বাক্যটি positive বা negative বার্তাগুলোর একটি array বা list এ সংরক্ষণ করুন।
5. শেষে, সমস্ত positive বাক্য এবং negative বাক্য (আলাদাভাবে) এবং প্রতিটির সংখ্যা প্রিন্ট করুন।

এখানে একটি নমুনা [সমাধান](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)।

✅ জ্ঞান যাচাই

1. অনুভূতি বাক্যে ব্যবহৃত শব্দের উপর ভিত্তি করে, কিন্তু কোড কি শব্দগুলো *বুঝতে* পারে?
2. আপনি কি sentiment polarity এর সঠিকতা নিয়ে একমত, বা অন্যভাবে বললে, আপনি কি স্কোরগুলোর সাথে *একমত*?
   1. বিশেষ করে, আপনি কি নিম্নলিখিত বাক্যগুলোর absolute **positive** polarity এর সাথে একমত বা দ্বিমত?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. পরবর্তী ৩টি বাক্য absolute positive sentiment দিয়ে স্কোর করা হয়েছে, কিন্তু গভীরভাবে পড়লে, এগুলো positive বাক্য নয়। কেন sentiment analysis মনে করেছে এগুলো positive বাক্য?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. আপনি কি নিম্নলিখিত বাক্যগুলোর absolute **negative** polarity এর সাথে একমত বা দ্বিমত?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ জেন অস্টেনের কোনো অনুরাগী বুঝতে পারবেন যে তিনি প্রায়ই তার বইগুলোতে ইংরেজি রিজেন্সি সমাজের আরও হাস্যকর দিকগুলো সমালোচনা করেন। *Pride and Prejudice* এর প্রধান চরিত্র এলিজাবেথ বেনেট একজন তীক্ষ্ণ সামাজিক পর্যবেক্ষক (লেখকের মতো) এবং তার ভাষা প্রায়ই গভীরভাবে সূক্ষ্ম। এমনকি মিস্টার ডারসি (গল্পের প্রেমের চরিত্র) এলিজাবেথের খেলাধুলাপূর্ণ এবং ঠাট্টামূলক ভাষার ব্যবহার লক্ষ্য করেন: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀চ্যালেঞ্জ

আপনি কি Marvin কে আরও ভালো করতে পারেন ব্যবহারকারীর ইনপুট থেকে অন্যান্য বৈশিষ্ট্য বের করে?

## [পাঠ-পরবর্তী কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## পর্যালোচনা এবং স্ব-অধ্যয়ন
টেক্সট থেকে অনুভূতি নির্ণয়ের অনেক পদ্ধতি রয়েছে। এই কৌশলটি ব্যবহার করে ব্যবসায়িক প্রয়োগগুলোর কথা ভাবুন। এটি কীভাবে ভুল পথে যেতে পারে, সেটাও চিন্তা করুন। উন্নত এবং এন্টারপ্রাইজ-প্রস্তুত সিস্টেম সম্পর্কে আরও জানুন, যা অনুভূতি বিশ্লেষণ করে, যেমন [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)। উপরের "Pride and Prejudice" এর কিছু বাক্য পরীক্ষা করুন এবং দেখুন এটি সূক্ষ্মতা শনাক্ত করতে পারে কিনা।

## অ্যাসাইনমেন্ট

[Poetic license](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা তার জন্য দায়ী থাকব না।