<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-08-29T22:21:47+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "bn"
}
-->
# হোটেল রিভিউ দিয়ে সেন্টিমেন্ট অ্যানালাইসিস - ডেটা প্রক্রিয়াকরণ

এই সেকশনে আপনি আগের পাঠে শেখা কৌশলগুলো ব্যবহার করে একটি বড় ডেটাসেটের উপর কিছু অনুসন্ধানমূলক ডেটা বিশ্লেষণ করবেন। বিভিন্ন কলামের কার্যকারিতা সম্পর্কে ভালো ধারণা পাওয়ার পর, আপনি শিখবেন:

- কীভাবে অপ্রয়োজনীয় কলামগুলো সরানো যায়
- কীভাবে বিদ্যমান কলাম থেকে নতুন ডেটা গণনা করা যায়
- কীভাবে চূড়ান্ত চ্যালেঞ্জের জন্য ডেটাসেট সংরক্ষণ করা যায়

## [পূর্ব-পাঠ কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### ভূমিকা

এখন পর্যন্ত আপনি শিখেছেন যে টেক্সট ডেটা সংখ্যাসূচক ডেটার মতো নয়। যদি এটি মানুষের লেখা বা বলা টেক্সট হয়, তবে এটি বিশ্লেষণ করে প্যাটার্ন, ফ্রিকোয়েন্সি, সেন্টিমেন্ট এবং অর্থ খুঁজে বের করা যায়। এই পাঠে আপনাকে একটি বাস্তব ডেটাসেট এবং একটি বাস্তব চ্যালেঞ্জের সাথে পরিচয় করানো হবে: **[ইউরোপের ৫১৫ হাজার হোটেল রিভিউ ডেটা](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, যা [CC0: পাবলিক ডোমেইন লাইসেন্স](https://creativecommons.org/publicdomain/zero/1.0/) এর অধীনে রয়েছে। এটি Booking.com থেকে পাবলিক সোর্স থেকে সংগ্রহ করা হয়েছে। ডেটাসেটটি তৈরি করেছেন জিয়াশেন লিউ।

### প্রস্তুতি

আপনার প্রয়োজন হবে:

* Python 3 ব্যবহার করে .ipynb নোটবুক চালানোর সক্ষমতা
* pandas
* NLTK, [যা আপনাকে লোকালভাবে ইন্সটল করতে হবে](https://www.nltk.org/install.html)
* ডেটাসেট, যা Kaggle-এ পাওয়া যাবে [ইউরোপের ৫১৫ হাজার হোটেল রিভিউ ডেটা](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)। আনজিপ করার পর এটি প্রায় ২৩০ এমবি। এটি এই NLP পাঠগুলোর সাথে সম্পর্কিত `/data` ফোল্ডারে ডাউনলোড করুন।

## অনুসন্ধানমূলক ডেটা বিশ্লেষণ

এই চ্যালেঞ্জটি ধরে নেয় যে আপনি সেন্টিমেন্ট অ্যানালাইসিস এবং অতিথিদের রিভিউ স্কোর ব্যবহার করে একটি হোটেল রিকমেন্ডেশন বট তৈরি করছেন। আপনি যে ডেটাসেটটি ব্যবহার করবেন, তাতে ৬টি শহরের ১৪৯৩টি ভিন্ন হোটেলের রিভিউ অন্তর্ভুক্ত রয়েছে।

Python, হোটেল রিভিউ ডেটাসেট এবং NLTK-এর সেন্টিমেন্ট অ্যানালাইসিস ব্যবহার করে আপনি জানতে পারবেন:

* রিভিউতে সবচেয়ে বেশি ব্যবহৃত শব্দ এবং বাক্যাংশ কী কী?
* হোটেল বর্ণনা করার জন্য ব্যবহৃত *ট্যাগ* গুলো কি রিভিউ স্কোরের সাথে সম্পর্কিত (যেমন, *Family with young children* এর জন্য কি কোনো নির্দিষ্ট হোটেলে বেশি নেতিবাচক রিভিউ রয়েছে, যা হয়তো ইঙ্গিত করে যে এটি *Solo traveller* এর জন্য বেশি উপযুক্ত)?
* NLTK সেন্টিমেন্ট স্কোর কি হোটেল রিভিউয়ারের সংখ্যাসূচক স্কোরের সাথে মিলে যায়?

#### ডেটাসেট

চলুন ডেটাসেটটি এক্সপ্লোর করি, যা আপনি লোকালভাবে ডাউনলোড এবং সংরক্ষণ করেছেন। ফাইলটি VS Code বা এমনকি Excel-এর মতো কোনো এডিটরে খুলুন।

ডেটাসেটের হেডারগুলো নিম্নরূপ:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

এগুলোকে এমনভাবে গ্রুপ করা হয়েছে, যা পরীক্ষা করা সহজ হতে পারে:
##### হোটেল কলাম

* `Hotel_Name`, `Hotel_Address`, `lat` (অক্ষাংশ), `lng` (দ্রাঘিমাংশ)
  * *lat* এবং *lng* ব্যবহার করে আপনি Python দিয়ে একটি মানচিত্র তৈরি করতে পারেন, যেখানে হোটেলের অবস্থানগুলো দেখানো হবে (সম্ভবত নেতিবাচক এবং ইতিবাচক রিভিউয়ের জন্য ভিন্ন রঙে চিহ্নিত)
  * Hotel_Address আমাদের জন্য সরাসরি খুব একটা উপকারী নয়, এবং আমরা সম্ভবত এটি একটি দেশের সাথে প্রতিস্থাপন করব, যাতে সহজে সাজানো এবং অনুসন্ধান করা যায়।

**হোটেল মেটা-রিভিউ কলাম**

* `Average_Score`
  * ডেটাসেটের নির্মাতা অনুযায়ী, এই কলামটি হলো *হোটেলের গড় স্কোর, যা গত বছরের সর্বশেষ মন্তব্যের উপর ভিত্তি করে গণনা করা হয়েছে*। এটি একটি অস্বাভাবিক পদ্ধতি মনে হতে পারে, তবে এটি স্ক্র্যাপ করা ডেটা, তাই আপাতত আমরা এটিকে গ্রহণ করতে পারি।
  
  ✅ এই ডেটার অন্যান্য কলামের উপর ভিত্তি করে গড় স্কোর গণনা করার আরেকটি উপায় কি আপনি ভাবতে পারেন?

* `Total_Number_of_Reviews`
  * এই হোটেলটি কতগুলো রিভিউ পেয়েছে তার মোট সংখ্যা - এটি স্পষ্ট নয় (কোড না লিখে) যে এটি ডেটাসেটে থাকা রিভিউগুলোকে বোঝায় কিনা।
* `Additional_Number_of_Scoring`
  * এর মানে হলো রিভিউ স্কোর দেওয়া হয়েছে, কিন্তু রিভিউয়ার কোনো ইতিবাচক বা নেতিবাচক রিভিউ লেখেননি।

**রিভিউ কলাম**

- `Reviewer_Score`
  - এটি একটি সংখ্যাসূচক মান, যা সর্বাধিক ১ দশমিক স্থান পর্যন্ত হতে পারে এবং এর সর্বনিম্ন এবং সর্বোচ্চ মান যথাক্রমে ২.৫ এবং ১০।
  - কেন ২.৫ সর্বনিম্ন স্কোর তা ব্যাখ্যা করা হয়নি।
- `Negative_Review`
  - যদি কোনো রিভিউয়ার কিছু না লেখেন, তবে এই ফিল্ডে "**No Negative**" থাকবে।
  - লক্ষ্য করুন যে, রিভিউয়ার নেতিবাচক রিভিউ কলামে ইতিবাচক কিছু লিখতে পারেন (যেমন, "এই হোটেলে খারাপ কিছু নেই")।
- `Review_Total_Negative_Word_Counts`
  - বেশি নেতিবাচক শব্দের সংখ্যা কম স্কোর নির্দেশ করে (সেন্টিমেন্টালিটি যাচাই না করেই)।
- `Positive_Review`
  - যদি কোনো রিভিউয়ার কিছু না লেখেন, তবে এই ফিল্ডে "**No Positive**" থাকবে।
  - লক্ষ্য করুন যে, রিভিউয়ার ইতিবাচক রিভিউ কলামে নেতিবাচক কিছু লিখতে পারেন (যেমন, "এই হোটেলে মোটেও ভালো কিছু নেই")।
- `Review_Total_Positive_Word_Counts`
  - বেশি ইতিবাচক শব্দের সংখ্যা বেশি স্কোর নির্দেশ করে (সেন্টিমেন্টালিটি যাচাই না করেই)।
- `Review_Date` এবং `days_since_review`
  - একটি রিভিউয়ের সতেজতা বা পুরনো হওয়ার পরিমাপ প্রয়োগ করা যেতে পারে (পুরনো রিভিউগুলো নতুনগুলোর মতো সঠিক নাও হতে পারে, কারণ হোটেল ম্যানেজমেন্ট পরিবর্তিত হয়েছে, সংস্কার করা হয়েছে, বা একটি সুইমিং পুল যোগ করা হয়েছে ইত্যাদি)।
- `Tags`
  - এগুলো ছোট বর্ণনা, যা রিভিউয়ার তাদের অতিথি হিসেবে ধরণ (যেমন, একা বা পরিবার), রুমের ধরণ, থাকার সময়কাল এবং কীভাবে রিভিউ জমা দেওয়া হয়েছে তা বর্ণনা করতে নির্বাচন করতে পারেন।
  - দুর্ভাগ্যবশত, এই ট্যাগগুলো ব্যবহার করা সমস্যাজনক, নিচের সেকশনে এদের কার্যকারিতা নিয়ে আলোচনা করা হয়েছে।

**রিভিউয়ার কলাম**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - এটি একটি রিকমেন্ডেশন মডেলে একটি ফ্যাক্টর হতে পারে, উদাহরণস্বরূপ, আপনি যদি নির্ধারণ করতে পারেন যে শতাধিক রিভিউ দেওয়া বেশি সক্রিয় রিভিউয়াররা নেতিবাচক হওয়ার চেয়ে ইতিবাচক হওয়ার সম্ভাবনা বেশি। তবে, কোনো নির্দিষ্ট রিভিউয়ের রিভিউয়ারকে একটি ইউনিক কোড দিয়ে চিহ্নিত করা হয়নি, এবং তাই এটি একটি সেট রিভিউয়ের সাথে সংযুক্ত করা সম্ভব নয়। ১০০ বা তার বেশি রিভিউ দেওয়া ৩০ জন রিভিউয়ার রয়েছে, তবে এটি রিকমেন্ডেশন মডেলে কীভাবে সহায়তা করতে পারে তা বোঝা কঠিন।
- `Reviewer_Nationality`
  - কিছু মানুষ মনে করতে পারেন যে নির্দিষ্ট জাতীয়তার লোকেরা ইতিবাচক বা নেতিবাচক রিভিউ দেওয়ার প্রবণতা বেশি। তবে, এমন অনুমানমূলক দৃষ্টিভঙ্গি আপনার মডেলে অন্তর্ভুক্ত করার সময় সতর্ক থাকুন। এগুলো জাতীয় (এবং কখনও কখনও জাতিগত) স্টেরিওটাইপ, এবং প্রতিটি রিভিউয়ার তাদের অভিজ্ঞতার ভিত্তিতে একটি রিভিউ লিখেছেন। এটি তাদের পূর্ববর্তী হোটেল থাকার অভিজ্ঞতা, ভ্রমণের দূরত্ব এবং তাদের ব্যক্তিগত মেজাজের মতো অনেক দৃষ্টিকোণ থেকে ফিল্টার হতে পারে। তাদের জাতীয়তাই রিভিউ স্কোরের কারণ ছিল এমনটি ভাবা কঠিন।

##### উদাহরণ

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

যেমনটি দেখা যাচ্ছে, এই অতিথি হোটেলে খুবই খারাপ অভিজ্ঞতা পেয়েছেন। হোটেলের গড় স্কোর ৭.৮ এবং ১৯৪৫টি রিভিউ রয়েছে, কিন্তু এই রিভিউয়ার ২.৫ স্কোর দিয়েছেন এবং তাদের নেতিবাচক অভিজ্ঞতা নিয়ে ১১৫টি শব্দ লিখেছেন। যদি তারা Positive_Review কলামে কিছুই না লিখতেন, তবে আপনি অনুমান করতে পারতেন যে সেখানে কিছুই ইতিবাচক ছিল না, কিন্তু তারা সতর্কবার্তা হিসেবে ৭টি শব্দ লিখেছেন। যদি আমরা শব্দ গণনা করি অর্থ বা শব্দের সেন্টিমেন্টের পরিবর্তে, তবে আমরা রিভিউয়ারের উদ্দেশ্য সম্পর্কে একটি বিকৃত ধারণা পেতে পারি। অদ্ভুতভাবে, তাদের ২.৫ স্কোর বিভ্রান্তিকর, কারণ যদি হোটেলে থাকার অভিজ্ঞতা এতটাই খারাপ হয়, তবে কেন কোনো পয়েন্ট দেওয়া হলো? ডেটাসেটটি ঘনিষ্ঠভাবে পরীক্ষা করলে, আপনি দেখবেন যে সর্বনিম্ন স্কোর সম্ভবত ২.৫, ০ নয়। সর্বোচ্চ স্কোর সম্ভবত ১০।

##### ট্যাগ

উপরোক্ত আলোচনায় উল্লেখ করা হয়েছে যে, প্রথম নজরে `Tags` ব্যবহার করে ডেটা শ্রেণীবদ্ধ করার ধারণাটি যৌক্তিক মনে হয়। দুর্ভাগ্যবশত, এই ট্যাগগুলো স্ট্যান্ডার্ডাইজড নয়, যার মানে একটি নির্দিষ্ট হোটেলে অপশনগুলো হতে পারে *Single room*, *Twin room*, এবং *Double room*, কিন্তু অন্য একটি হোটেলে এগুলো হতে পারে *Deluxe Single Room*, *Classic Queen Room*, এবং *Executive King Room*। এগুলো একই জিনিস হতে পারে, কিন্তু এত বেশি বৈচিত্র্য রয়েছে যে পছন্দগুলো হয়ে যায়:

1. সমস্ত টার্মকে একটি একক স্ট্যান্ডার্ডে পরিবর্তন করার চেষ্টা করা, যা খুবই কঠিন, কারণ প্রতিটি ক্ষেত্রে রূপান্তরের পথ কী হবে তা স্পষ্ট নয় (যেমন, *Classic single room* কে *Single room* এ ম্যাপ করা যায়, কিন্তু *Superior Queen Room with Courtyard Garden or City View* কে ম্যাপ করা অনেক কঠিন)।

1. আমরা একটি NLP পদ্ধতি নিতে পারি এবং প্রতিটি হোটেলের ক্ষেত্রে *Solo*, *Business Traveller*, বা *Family with young kids* এর মতো নির্দিষ্ট টার্মগুলোর ফ্রিকোয়েন্সি পরিমাপ করতে পারি এবং এটি রিকমেন্ডেশনে অন্তর্ভুক্ত করতে পারি।

ট্যাগগুলো সাধারণত (কিন্তু সবসময় নয়) একটি একক ফিল্ড, যা ৫ থেকে ৬টি কমা দ্বারা পৃথক মানের তালিকা ধারণ করে, যা *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, এবং *Type of device review was submitted on* এর সাথে সামঞ্জস্যপূর্ণ। তবে, কিছু রিভিউয়ার প্রতিটি ফিল্ড পূরণ করেন না (তারা একটি ফিল্ড ফাঁকা রাখতে পারেন), তাই মানগুলো সবসময় একই ক্রমে থাকে না।

উদাহরণস্বরূপ, *Type of group* নিন। এই ফিল্ডে `Tags` কলামে ১০২৫টি ইউনিক সম্ভাবনা রয়েছে, এবং দুর্ভাগ্যবশত এর মধ্যে কেবল কিছু গ্রুপকে বোঝায় (কিছু রুমের ধরণ ইত্যাদি)। যদি আপনি শুধুমাত্র সেই মানগুলো ফিল্টার করেন যা পরিবার উল্লেখ করে, তবে ফলাফলে অনেক *Family room* টাইপের মান অন্তর্ভুক্ত থাকে। যদি আপনি *with* টার্মটি অন্তর্ভুক্ত করেন, অর্থাৎ *Family with* মানগুলো গণনা করেন, তবে ফলাফলগুলো আরও ভালো হয়, যেখানে ৫১৫,০০০ ফলাফলের মধ্যে ৮০,০০০ এর বেশি "Family with young children" বা "Family with older children" বাক্যাংশ অন্তর্ভুক্ত থাকে।

এর মানে হলো ট্যাগ কলামটি আমাদের জন্য পুরোপুরি অপ্রয়োজনীয় নয়, তবে এটি কার্যকর করতে কিছু কাজ করতে হবে।

##### গড় হোটেল স্কোর

ডেটাসেটের সাথে কিছু অদ্ভুততা বা অসঙ্গতি রয়েছে, যা আমি বুঝতে পারিনি, তবে এখানে সেগুলো চিত্রিত করা হয়েছে যাতে আপনি মডেল তৈরি করার সময় সেগুলো সম্পর্কে সচেতন থাকেন। আপনি যদি এটি বের করতে পারেন, তবে আমাদের আলোচনা সেকশনে জানান!

ডেটাসেটে গড় স্কোর এবং রিভিউয়ের সংখ্যার সাথে সম্পর্কিত নিম্নলিখিত কলামগুলো রয়েছে:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

এই ডেটাসেটে সবচেয়ে বেশি রিভিউ থাকা একক হোটেল হলো *Britannia International Hotel Canary Wharf*, যার ৪৭৮৯টি রিভিউ রয়েছে ৫১৫,০০০ এর মধ্যে। কিন্তু যদি আমরা এই হোটেলের জন্য `Total_Number_of_Reviews` মানটি দেখি, এটি ৯০৮৬। আপনি অনুমান করতে পারেন যে আরও অনেক স্কোর রিভিউ ছাড়াই রয়েছে, তাই সম্ভবত আমাদের `Additional_Number_of_Scoring` কলামের মান যোগ করা উচিত। সেই মানটি ২৬৮২, এবং এটি ৪৭৮৯ এর সাথে যোগ করলে ৭৪৭১ হয়, যা এখনও `Total_Number_of_Reviews` এর চেয়ে ১৬১৫ কম।

যদি আপনি `Average_Score` কলামটি নেন, আপনি অনুমান করতে পারেন এটি ডেটাসেটে থাকা রিভিউয়ের গড়, কিন্তু Kaggle-এর বর্ণনা অনুযায়ী এটি "*গত বছরের সর্বশেষ মন্তব্যের উপর ভিত্তি করে হোটেলের গড় স্কোর*"। এটি খুব একটা কার্যকর মনে হয় না, তবে আমরা ডেটাসেটে থাকা রিভিউ স্কোরের উপর ভিত্তি করে আমাদের নিজস্ব গড় স্কোর গণনা করতে পারি। একই হোটেলকে উদাহরণ হিসেবে ব্যবহার করে, গড় হোটেল স্কোর দেওয়া হয়েছে ৭.১, কিন্তু ডেটাসেটে থাকা রিভিউ স্কোরের গড় (গণনা করা) হলো ৬.৮। এটি কাছাকাছি, কিন্তু একই মান নয়, এবং আমরা কেবল অনুমান করতে পারি যে `Additional_Number_of_Scoring` রিভিউগুলো গড়কে ৭.১-এ উন্নীত করেছে। দুর্ভাগ্যবশত, সেই অনুমান পরীক্ষা বা প্রমাণ করার কোনো উপায় না থাকায়, `Average_Score`, `Additional_Number_of_Scoring` এবং `Total_Number_of_Reviews` ব্যবহার বা বিশ্বাস করা কঠিন, কারণ সেগুলো এমন ডেটার উপর ভিত্তি করে তৈরি, যা আমাদের কাছে নেই।

বিষয়টিকে আরও জটিল করে তুলতে, ডেটাসেটে দ্বিতীয় সর্বাধিক রিভিউ থাকা হোটেলের গণিত অনুযায়ী গড় স্কোর ৮.১২ এবং ডেটাসেটের `Average_Score` হলো ৮.১। এটি কি সঠিক স্কোরের একটি কাকতালীয় ঘটনা, নাকি প্রথম হোটেলটি একটি অসঙ্গতি?
এই হোটেলটি যদি একটি ব্যতিক্রম হয় এবং হয়তো বেশিরভাগ মানগুলি সঠিকভাবে মিলে যায় (কিন্তু কিছু কারণে কিছু মান মিলে না), তাহলে আমরা একটি ছোট প্রোগ্রাম লিখব যা ডেটাসেটে মানগুলি পরীক্ষা করবে এবং সঠিক ব্যবহার (বা অ-ব্যবহার) নির্ধারণ করবে।

> 🚨 সতর্কতার একটি নোট
>
> এই ডেটাসেট নিয়ে কাজ করার সময় আপনি এমন কোড লিখবেন যা টেক্সট থেকে কিছু গণনা করে, টেক্সট নিজে পড়া বা বিশ্লেষণ করার প্রয়োজন ছাড়াই। এটি NLP-এর মূল বিষয়, অর্থ বা অনুভূতি ব্যাখ্যা করা যা একজন মানুষের দ্বারা না করেও সম্ভব। তবে, এটি সম্ভব যে আপনি কিছু নেতিবাচক রিভিউ পড়বেন। আমি আপনাকে এটি না করার পরামর্শ দেব, কারণ এটি করার প্রয়োজন নেই। কিছু রিভিউ হাস্যকর বা অপ্রাসঙ্গিক নেতিবাচক হোটেল রিভিউ হতে পারে, যেমন "আবহাওয়া ভালো ছিল না", যা হোটেলের বা কারো নিয়ন্ত্রণের বাইরে। তবে কিছু রিভিউয়ের একটি অন্ধকার দিকও রয়েছে। কখনও কখনও নেতিবাচক রিভিউগুলো বর্ণবাদী, লিঙ্গবাদী বা বয়সবাদী হতে পারে। এটি দুর্ভাগ্যজনক, তবে একটি পাবলিক ওয়েবসাইট থেকে স্ক্র্যাপ করা ডেটাসেটে এটি প্রত্যাশিত। কিছু রিভিউয়ার এমন রিভিউ রেখে যান যা আপনি অপ্রীতিকর, অস্বস্তিকর বা বিরক্তিকর মনে করতে পারেন। কোডকে অনুভূতি পরিমাপ করতে দিন, নিজে পড়ে বিরক্ত হওয়ার চেয়ে। তবে, এটি একটি সংখ্যালঘু যারা এমন জিনিস লিখে, কিন্তু তারা সবসময়ই থাকে।

## অনুশীলন - ডেটা অনুসন্ধান
### ডেটা লোড করুন

ডেটা ভিজ্যুয়ালি পরীক্ষা করা যথেষ্ট হয়েছে, এখন আপনি কিছু কোড লিখবেন এবং কিছু উত্তর পাবেন! এই অংশে pandas লাইব্রেরি ব্যবহার করা হয়েছে। আপনার প্রথম কাজ হল নিশ্চিত করা যে আপনি CSV ডেটা লোড এবং পড়তে পারেন। pandas লাইব্রেরির একটি দ্রুত CSV লোডার রয়েছে, এবং ফলাফলটি একটি ডেটাফ্রেমে রাখা হয়, যেমন আগের পাঠে। আমরা যে CSV লোড করছি তাতে অর্ধ মিলিয়নেরও বেশি সারি রয়েছে, কিন্তু মাত্র ১৭টি কলাম। pandas আপনাকে ডেটাফ্রেমের সাথে ইন্টারঅ্যাক্ট করার জন্য অনেক শক্তিশালী উপায় দেয়, যার মধ্যে প্রতিটি সারিতে অপারেশন করার ক্ষমতা রয়েছে।

এই পাঠের পর থেকে কোড স্নিপেট এবং কোডের কিছু ব্যাখ্যা এবং ফলাফল সম্পর্কে কিছু আলোচনা থাকবে। আপনার কোডের জন্য অন্তর্ভুক্ত _notebook.ipynb_ ব্যবহার করুন।

চলুন শুরু করি আপনি যে ডেটা ফাইলটি ব্যবহার করবেন তা লোড করার মাধ্যমে:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

ডেটা লোড হয়ে গেলে, আমরা এর উপর কিছু অপারেশন করতে পারি। আপনার প্রোগ্রামের পরবর্তী অংশের জন্য এই কোডটি উপরে রাখুন।

## ডেটা অনুসন্ধান

এই ক্ষেত্রে, ডেটা ইতিমধ্যেই *পরিষ্কার*, অর্থাৎ এটি কাজ করার জন্য প্রস্তুত এবং এতে অন্য ভাষার অক্ষর নেই যা শুধুমাত্র ইংরেজি অক্ষর প্রত্যাশা করা অ্যালগরিদমকে বিভ্রান্ত করতে পারে।

✅ আপনাকে এমন ডেটার সাথে কাজ করতে হতে পারে যা NLP কৌশল প্রয়োগ করার আগে কিছু প্রাথমিক প্রক্রিয়াকরণ প্রয়োজন, তবে এই সময় নয়। যদি প্রয়োজন হয়, আপনি কীভাবে অ-ইংরেজি অক্ষর পরিচালনা করবেন?

এক মুহূর্ত সময় নিন নিশ্চিত করতে যে ডেটা লোড হওয়ার পরে আপনি কোড দিয়ে এটি অনুসন্ধান করতে পারেন। `Negative_Review` এবং `Positive_Review` কলামগুলিতে ফোকাস করতে চাওয়া খুব সহজ। এগুলো আপনার NLP অ্যালগরিদমের জন্য প্রাকৃতিক টেক্সট দিয়ে পূর্ণ। কিন্তু অপেক্ষা করুন! NLP এবং অনুভূতির দিকে ঝাঁপ দেওয়ার আগে, আপনাকে নিচের কোডটি অনুসরণ করতে হবে যাতে ডেটাসেটে দেওয়া মানগুলি pandas দিয়ে গণনা করা মানগুলির সাথে মিলে কিনা তা নিশ্চিত করা যায়।

## ডেটাফ্রেম অপারেশন

এই পাঠের প্রথম কাজ হল কিছু কোড লিখে ডেটাফ্রেম পরীক্ষা করা (এটি পরিবর্তন না করে) এবং নিচের দাবিগুলি সঠিক কিনা তা যাচাই করা।

> অনেক প্রোগ্রামিং কাজের মতো, এটি সম্পন্ন করার জন্য বেশ কয়েকটি উপায় রয়েছে, তবে ভালো পরামর্শ হল এটি যতটা সম্ভব সহজ এবং সহজ উপায়ে করা, বিশেষত যদি এটি ভবিষ্যতে এই কোডে ফিরে আসার সময় বুঝতে সহজ হয়। ডেটাফ্রেমের সাথে, একটি বিস্তৃত API রয়েছে যা প্রায়শই আপনি যা চান তা দক্ষতার সাথে করার একটি উপায় থাকবে।

নিচের প্রশ্নগুলোকে কোডিং টাস্ক হিসেবে বিবেচনা করুন এবং সমাধান না দেখে উত্তর দেওয়ার চেষ্টা করুন।

1. আপনি যে ডেটাফ্রেমটি লোড করেছেন তার *shape* প্রিন্ট করুন (shape হল সারি এবং কলামের সংখ্যা)
2. রিভিউয়ার জাতীয়তার ফ্রিকোয়েন্সি গণনা করুন:
   1. `Reviewer_Nationality` কলামের জন্য কতটি স্বতন্ত্র মান রয়েছে এবং সেগুলো কী কী?
   2. ডেটাসেটে সবচেয়ে সাধারণ রিভিউয়ার জাতীয়তা কোনটি (দেশ এবং রিভিউ সংখ্যা প্রিন্ট করুন)?
   3. পরবর্তী শীর্ষ ১০টি সবচেয়ে বেশি পাওয়া জাতীয়তা এবং তাদের ফ্রিকোয়েন্সি গণনা কী কী?
3. শীর্ষ ১০টি রিভিউয়ার জাতীয়তার জন্য সবচেয়ে বেশি রিভিউ করা হোটেল কোনটি?
4. ডেটাসেটে প্রতি হোটেলের জন্য কতটি রিভিউ রয়েছে (হোটেলের ফ্রিকোয়েন্সি গণনা)?
5. যদিও ডেটাসেটে প্রতিটি হোটেলের জন্য একটি `Average_Score` কলাম রয়েছে, আপনি একটি গড় স্কোরও গণনা করতে পারেন (ডেটাসেটে প্রতিটি হোটেলের জন্য সমস্ত রিভিউয়ার স্কোরের গড় পাওয়া)। আপনার ডেটাফ্রেমে `Calc_Average_Score` শিরোনামের একটি নতুন কলাম যোগ করুন যা সেই গণিত গড় ধারণ করে।
6. কোনও হোটেলের `Average_Score` এবং `Calc_Average_Score` (১ দশমিক স্থানে রাউন্ড করা) একই আছে কিনা?
   1. একটি Python ফাংশন লিখে দেখুন যা একটি Series (সারি) আর্গুমেন্ট হিসেবে নেয় এবং মানগুলো তুলনা করে, যখন মানগুলো সমান নয় তখন একটি বার্তা প্রিন্ট করে। তারপর `.apply()` পদ্ধতি ব্যবহার করে প্রতিটি সারি ফাংশন দিয়ে প্রক্রিয়া করুন।
7. `Negative_Review` কলামের "No Negative" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।
8. `Positive_Review` কলামের "No Positive" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।
9. `Positive_Review` কলামের "No Positive" **এবং** `Negative_Review` কলামের "No Negative" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

### কোড উত্তর

1. আপনি যে ডেটাফ্রেমটি লোড করেছেন তার *shape* প্রিন্ট করুন (shape হল সারি এবং কলামের সংখ্যা)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. রিভিউয়ার জাতীয়তার ফ্রিকোয়েন্সি গণনা করুন:

   1. `Reviewer_Nationality` কলামের জন্য কতটি স্বতন্ত্র মান রয়েছে এবং সেগুলো কী কী?
   2. ডেটাসেটে সবচেয়ে সাধারণ রিভিউয়ার জাতীয়তা কোনটি (দেশ এবং রিভিউ সংখ্যা প্রিন্ট করুন)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. পরবর্তী শীর্ষ ১০টি সবচেয়ে বেশি পাওয়া জাতীয়তা এবং তাদের ফ্রিকোয়েন্সি গণনা কী কী?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. শীর্ষ ১০টি রিভিউয়ার জাতীয়তার জন্য সবচেয়ে বেশি রিভিউ করা হোটেল কোনটি?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. ডেটাসেটে প্রতি হোটেলের জন্য কতটি রিভিউ রয়েছে (হোটেলের ফ্রিকোয়েন্সি গণনা)?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   আপনি লক্ষ্য করতে পারেন যে *ডেটাসেটে গণনা করা* ফলাফল `Total_Number_of_Reviews` মানের সাথে মেলে না। এটি স্পষ্ট নয় যে এই মানটি ডেটাসেটে হোটেলের মোট রিভিউ সংখ্যা প্রতিনিধিত্ব করেছিল, কিন্তু সবগুলো স্ক্র্যাপ করা হয়নি, বা অন্য কোনও গণনা। `Total_Number_of_Reviews` মডেলে ব্যবহার করা হয় না কারণ এই অস্পষ্টতার কারণে।

5. যদিও ডেটাসেটে প্রতিটি হোটেলের জন্য একটি `Average_Score` কলাম রয়েছে, আপনি একটি গড় স্কোরও গণনা করতে পারেন (ডেটাসেটে প্রতিটি হোটেলের জন্য সমস্ত রিভিউয়ার স্কোরের গড় পাওয়া)। আপনার ডেটাফ্রেমে `Calc_Average_Score` শিরোনামের একটি নতুন কলাম যোগ করুন যা সেই গণিত গড় ধারণ করে। `Hotel_Name`, `Average_Score`, এবং `Calc_Average_Score` কলামগুলো প্রিন্ট করুন।

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   আপনি `Average_Score` মান এবং কেন এটি কখনও কখনও গণিত গড় স্কোর থেকে আলাদা তা নিয়ে ভাবতে পারেন। যেহেতু আমরা জানি না কেন কিছু মান মিলে যায়, কিন্তু অন্যগুলোর মধ্যে পার্থক্য রয়েছে, এই ক্ষেত্রে আমাদের কাছে থাকা রিভিউ স্কোর ব্যবহার করে গড় নিজে গণনা করা নিরাপদ। তবে, পার্থক্যগুলো সাধারণত খুব ছোট হয়, এখানে ডেটাসেট গড় এবং গণিত গড়ের মধ্যে সবচেয়ে বেশি বিচ্যুতি থাকা হোটেলগুলো:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   মাত্র ১টি হোটেলের স্কোরের পার্থক্য ১-এর বেশি, এর মানে আমরা সম্ভবত পার্থক্যটি উপেক্ষা করতে পারি এবং গণিত গড় স্কোর ব্যবহার করতে পারি।

6. `Negative_Review` কলামের "No Negative" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

7. `Positive_Review` কলামের "No Positive" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

8. `Positive_Review` কলামের "No Positive" **এবং** `Negative_Review` কলামের "No Negative" মানের কতটি সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## অন্য একটি উপায়

Lambda ছাড়া আইটেম গণনা করার আরেকটি উপায় এবং সারি গণনা করতে sum ব্যবহার করুন:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   আপনি লক্ষ্য করতে পারেন যে `Negative_Review` এবং `Positive_Review` কলামের "No Negative" এবং "No Positive" মানের ১২৭টি সারি রয়েছে। এর মানে রিভিউয়ার হোটেলকে একটি সংখ্যাসূচক স্কোর দিয়েছেন, কিন্তু কোনও ইতিবাচক বা নেতিবাচক রিভিউ লিখতে অস্বীকার করেছেন। সৌভাগ্যক্রমে এটি একটি ছোট পরিমাণ সারি (৫১৫৭৩৮-এর মধ্যে ১২৭, বা ০.০২%), তাই এটি সম্ভবত আমাদের মডেল বা ফলাফলকে কোনও নির্দিষ্ট দিকে প্রভাবিত করবে না, তবে আপনি একটি রিভিউ ডেটাসেটে কোনও রিভিউ ছাড়া সারি থাকতে পারে তা আশা করেননি, তাই এটি অনুসন্ধান করা মূল্যবান।

এখন আপনি ডেটাসেটটি অনুসন্ধান করেছেন, পরবর্তী পাঠে আপনি ডেটা ফিল্টার করবেন এবং কিছু অনুভূতি বিশ্লেষণ যোগ করবেন।

---
## 🚀চ্যালেঞ্জ

এই পাঠটি দেখায়, যেমন আমরা আগের পাঠে দেখেছি, যে আপনার ডেটা এবং এর ত্রুটি বুঝতে কতটা গুরুত্বপূর্ণ। বিশেষত টেক্সট-ভিত্তিক ডেটা সাবধানে পরীক্ষা করা প্রয়োজন। বিভিন্ন টেক্সট-ভারী ডেটাসেট খনন করুন এবং দেখুন আপনি এমন ক্ষেত্রগুলি আবিষ্কার করতে পারেন কিনা যা একটি মডেলে পক্ষপাত বা বিকৃত অনুভূতি প্রবর্তন করতে পারে।

## [পোস্ট-লেকচার কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## পর্যালোচনা এবং স্ব-অধ্যয়ন

[NLP নিয়ে এই লার্নিং পাথটি](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) গ্রহণ করুন এবং টেক্সট-ভারী মডেল তৈরি করার সময় চেষ্টা করার জন্য টুলগুলি আবিষ্কার করুন।

## অ্যাসাইনমেন্ট 

[NLTK](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা তার জন্য দায়ী থাকব না।