[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 多語言支援

#### 透過 GitHub Action 支援（自動且持續更新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[阿拉伯語](../ar/README.md) | [孟加拉語](../bn/README.md) | [保加利亞語](../bg/README.md) | [緬甸語](../my/README.md) | [中文（簡體）](../zh-CN/README.md) | [中文（繁體，香港）](../zh-HK/README.md) | [中文（繁體，澳門）](../zh-MO/README.md) | [中文（繁體，臺灣）](./README.md) | [克羅地亞語](../hr/README.md) | [捷克語](../cs/README.md) | [丹麥語](../da/README.md) | [荷蘭語](../nl/README.md) | [愛沙尼亞語](../et/README.md) | [芬蘭語](../fi/README.md) | [法語](../fr/README.md) | [德語](../de/README.md) | [希臘語](../el/README.md) | [希伯來語](../he/README.md) | [印地語](../hi/README.md) | [匈牙利語](../hu/README.md) | [印度尼西亞語](../id/README.md) | [義大利語](../it/README.md) | [日語](../ja/README.md) | [卡納達語](../kn/README.md) | [韓語](../ko/README.md) | [立陶宛語](../lt/README.md) | [馬來語](../ms/README.md) | [馬拉雅拉姆語](../ml/README.md) | [馬拉地語](../mr/README.md) | [尼泊爾語](../ne/README.md) | [奈及利亞洋泾浜語](../pcm/README.md) | [挪威語](../no/README.md) | [波斯語（法爾西語）](../fa/README.md) | [波蘭語](../pl/README.md) | [葡萄牙語（巴西）](../pt-BR/README.md) | [葡萄牙語（葡萄牙）](../pt-PT/README.md) | [旁遮普語（古魯穆奇）](../pa/README.md) | [羅馬尼亞語](../ro/README.md) | [俄語](../ru/README.md) | [塞爾維亞語（西里爾字母）](../sr/README.md) | [斯洛伐克語](../sk/README.md) | [斯洛文尼亞語](../sl/README.md) | [西班牙語](../es/README.md) | [斯瓦希里語](../sw/README.md) | [瑞典語](../sv/README.md) | [他加祿語（菲律賓語）](../tl/README.md) | [泰米爾語](../ta/README.md) | [泰盧固語](../te/README.md) | [泰語](../th/README.md) | [土耳其語](../tr/README.md) | [烏克蘭語](../uk/README.md) | [烏爾都語](../ur/README.md) | [越南語](../vi/README.md)

> **偏好本地 Clone？**
>
> 本倉庫包含 50 多種語言的翻譯，會大幅增加下載大小。若要不含翻譯檔案的 Clone，請使用 sparse checkout：
>
> **Bash / macOS / Linux:**
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
>
> **CMD (Windows):**
> ```cmd
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone "/*" "!translations" "!translated_images"
> ```
>
> 這讓你可以更快下載，且擁有所需完成課程的所有內容。
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### 加入我們的社群

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們有一系列 Discord AI 學習活動，詳情及參與請訪問 [Learn with AI Series](https://aka.ms/learnwithai/discord)，時間為 2025 年 9 月 18 日至 30 日。您將學到使用 GitHub Copilot 於資料科學的技巧與秘訣。

![Learn with AI series](../../translated_images/zh-TW/3.9b58fd8d6c373c20.webp)

# 機器學習給初學者 - 課程大綱

> 🌍 透過世界多元文化，一同探索機器學習 🌍

微軟雲端推廣團隊很高興提供一個為期 12 週、共 26 課的 **機器學習** 課程。在這套課程中，您將學習有時被稱為 **經典機器學習** 的內容，主要使用 Scikit-learn 這個函式庫，避免使用深度學習（深度學習部分包含在我們的 [AI for Beginners 課程](https://aka.ms/ai4beginners) 中）。此外，也建議搭配我們的 [初學者資料科學課程](https://aka.ms/ds4beginners) 一同學習。

跟著我們環遊世界，應用這些經典技術於全球各地的資料。每堂課都包含課前和課後測驗、書面教學指示、解答、作業等。我們採取以專案為基礎的教學法，讓你透過實作學習新技能，成效更佳。

**✍️ 由衷感謝我們的作者**：Jen Looper、Stephen Howell、Francesca Lazzeri、Tomomi Imura、Cassie Breviu、Dmitry Soshnikov、Chris Noring、Anirban Mukherjee、Ornella Altunyan、Ruth Yakubu 及 Amy Boyd

**🎨 也感謝我們的插畫師**：Tomomi Imura、Dasani Madipalli 與 Jen Looper

**🙏 特別感謝 🙏 微軟學生大使團隊作者、審查者與內容貢獻者**，特別是 Rishit Dagli、Muhammad Sakib Khan Inan、Rohan Raj、Alexandru Petrescu、Abhishek Jaiswal、Nawrin Tabassum、Ioan Samuila 和 Snigdha Agarwal

**🤩 額外感謝微軟學生大使 Eric Wanjau、Jasleen Sondhi 和 Vidushi Gupta 貢獻 R 課程內容！**

# 快速開始

請依序操作：
1. **Fork 倉庫**：點擊本頁右上方的「Fork」按鈕。
2. **Clone 倉庫**：`git clone https://github.com/microsoft/ML-For-Beginners.git`

> [在我們的 Microsoft Learn 集合中找到本課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要幫助？** 請參考我們的 [故障排除指南](TROUBLESHOOTING.md)，解決安裝、設定與課程執行常見問題。

**[學生專屬頁面](https://aka.ms/student-page)**，使用本課程時，請將整個倉庫 Fork 至您自己的 GitHub 帳號，自行或與團隊完成練習：

- 從課前測驗開始。
- 閱讀課程並完成活動，每個知識點停下來思考。
- 嘗試自行理解並完成專案，不要直接執行解答程式碼；不過解答程式碼位於各專案課程的 `/solution` 資料夾中供參考。
- 參加課後測驗。
- 完成挑戰題。
- 完成作業。
- 完成一組課程後，請造訪 [討論區](https://github.com/microsoft/ML-For-Beginners/discussions)，並透過填寫適當的 PAT 評量表來「大聲學習」。PAT（進度評量工具）是您填寫的學習工具，還可以回應其他人的 PAT，一起互相學習。

> 想要深入學習，我們推薦關注這些 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 模組與學習路徑。

**教師專用**，我們提供了[使用本課程的建議](for-teachers.md)。

---

## 影片導覽

部分課程有短版影片，您可在課程內嵌連結觀看，或前往 [Microsoft Developer YouTube 頻道上機器學習入門影片清單](https://aka.ms/ml-beginners-videos) 按下面圖片播放。

[![ML for beginners banner](../../translated_images/zh-TW/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 認識團隊

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**Gif 製作：** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 點擊上方圖片觀看關於專案和開發團隊的影片！

---

## 教學法

我們在打造本課程時堅持兩大教學原則：確保課程是動手做的 **專案導向**，以及包含 **頻繁測驗**。另外，本課程有一致的 **主題** 以保持連貫。

透過確保內容與專案對應，學習過程更吸引學生，概念吸收也更牢固。課前的低壓測驗幫助學生設定學習目標，而課後測驗則加強記憶。本課程設計靈活有趣，您可全部修完或擇取部分。專案由淺入深，隨著 12 週結束而逐漸複雜。此外，本課程包含機器學習實際應用的後記，可作為額外學分或討論基礎。

> 找到我們的[行為守則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯指南](TRANSLATIONS.md)與[故障排除指南](TROUBLESHOOTING.md)。歡迎您提供建設性回饋！

## 每堂課包含

- 可選的手繪筆記
- 可選的補充影片
- 影片導覽（部分課程）
- [課前暖身測驗](https://ff-quizzes.netlify.app/en/ml/)
- 書面課程內容
- 專案導向課程有詳細逐步指引，教您完成專案
- 知識檢查
- 挑戰題
- 補充閱讀
- 作業
- [課後測驗](https://ff-quizzes.netlify.app/en/ml/)
> **關於語言的說明**：這些課程主要以 Python 撰寫，但也有許多課程提供 R 語言版本。若要完成 R 課程，請前往 `/solution` 資料夾尋找 R 課程檔案。它們附有 .rmd 副檔名，代表 **R Markdown** 文件，簡單來說，是將 `程式碼區塊`（可能是 R 或其他語言）與 `YAML 標頭`（引導如何格式化輸出，例如 PDF）內嵌於 `Markdown 文件` 中的格式。 因此，它是資料科學的優秀撰寫框架，允許您結合程式碼、輸出與您的想法，並以 Markdown 撰寫。此外，R Markdown 文件可匯出成 PDF、HTML 或 Word 等格式。

> **關於小測驗的說明**：所有小測驗皆包含於[Quiz App 資料夾](../../quiz-app)中，總共 52 個小測驗，每個包含三個問題。這些小測驗會在課程中以連結形式呈現，但您也可以在本機執行 Quiz App；請參考 `quiz-app` 資料夾中的說明在本機部署或部署至 Azure。

| 課程編號 |                              主題                               |                   課程分類                   | 學習目標                                                                                                                     |                                                             相關課程                                                              |                         作者                         |
| :-------: | :-------------------------------------------------------------: | :------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
|    01     |                    機器學習導論                    |      [Introduction](1-Introduction/README.md)       | 了解機器學習的基本概念                                                                                                       |                                          [課程](1-Introduction/1-intro-to-ML/README.md)                                          |                      Muhammad                        |
|    02     |                    機器學習的歷史                    |      [Introduction](1-Introduction/README.md)       | 了解此領域的歷史背景                                                                                                         |                                         [課程](1-Introduction/2-history-of-ML/README.md)                                          |                    Jen 和 Amy                       |
|    03     |                    公平性與機器學習                    |      [Introduction](1-Introduction/README.md)       | 學生應考慮建構和應用機器學習模型時關於公平性的重要哲學議題                                                                   |                                            [課程](1-Introduction/3-fairness/README.md)                                            |                       Tomomi                        |
|    04     |                    機器學習技術                    |      [Introduction](1-Introduction/README.md)       | 機器學習研究者使用何種技術來建立機器學習模型？                                                                               |                                         [課程](1-Introduction/4-techniques-of-ML/README.md)                                         |                  Chris 和 Jen                        |
|    05     |                    迴歸導論                    |        [Regression](2-Regression/README.md)         | 透過 Python 和 Scikit-learn 開始建立迴歸模型                                                                                 |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |         Jen • Eric Wanjau         |
|    06     |                    北美南瓜價格 🎃                    |        [Regression](2-Regression/README.md)         | 對資料進行視覺化與清理，為 ML 準備資料                                                                                       |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |         Jen • Eric Wanjau         |
|    07     |                    北美南瓜價格 🎃                    |        [Regression](2-Regression/README.md)         | 建立線性與多項式迴歸模型                                                                                                     |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |         Jen 和 Dmitry • Eric Wanjau         |
|    08     |                    北美南瓜價格 🎃                    |        [Regression](2-Regression/README.md)         | 建立邏輯斯迴歸模型                                                                                                           |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |         Jen • Eric Wanjau         |
|    09     |                          網頁應用 🔌                          |           [Web App](3-Web-App/README.md)            | 建立一個網頁應用以使用您訓練好的模型                                                                                         |                                              [Python](3-Web-App/1-Web-App/README.md)                                               |                        Jen                          |
|    10     |                    分類導論                    |    [Classification](4-Classification/README.md)     | 清理、準備及視覺化您的資料；分類導論                                                                                        | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  |          Jen 和 Cassie • Eric Wanjau          |
|    11     |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 分類器入門                                                                                                                  | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) |          Jen 和 Cassie • Eric Wanjau          |
|    12     |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                                 | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) |          Jen 和 Cassie • Eric Wanjau          |
|    13     |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 使用您的模型建立推薦網站應用                                                                                               |                                              [Python](4-Classification/4-Applied/README.md)                                               |                        Jen                          |
|    14     |                    聚類導論                    |        [Clustering](5-Clustering/README.md)         | 清理、準備及視覺化資料；聚類導論                                                                                            |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |         Jen • Eric Wanjau         |
|    15     |              探索奈及利亞音樂品味 🎧              |        [Clustering](5-Clustering/README.md)         | 探索 K-均值聚類法                                                                                                          |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |         Jen • Eric Wanjau         |
|    16     |              自然語言處理導論 ☕️              |   [Natural language processing](6-NLP/README.md)    | 透過建立簡單機器人學習 NLP 基本知識                                                                                          |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|    17     |                    常見 NLP 任務 ☕️                    |   [Natural language processing](6-NLP/README.md)    | 透徹了解處理語言結構時常見任務                                                                                              |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|    18     |               翻譯與情感分析 ♥️               |   [Natural language processing](6-NLP/README.md)    | 以珍·奧斯汀作品進行情感與翻譯分析                                                                                          |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|    19     |                歐洲浪漫旅館 ♥️                |   [Natural language processing](6-NLP/README.md)    | 旅館評論情感分析（一）                                                                                                      |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|    20     |                歐洲浪漫旅館 ♥️                |   [Natural language processing](6-NLP/README.md)    | 旅館評論情感分析（二）                                                                                                      |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|    21     |                時間序列預測導論                |        [Time series](7-TimeSeries/README.md)        | 時間序列預測導論                                                                                                            |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                     Francesca                       |
|    22     | ⚡️ 世界用電 ⚡️ - 使用 ARIMA 進行時間序列預測 |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 實作時間序列預測                                                                                                 |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                     Francesca                       |
|    23     |  ⚡️ 世界用電 ⚡️ - 使用 SVR 進行時間序列預測  |        [Time series](7-TimeSeries/README.md)        | 使用支持向量回歸進行時間序列預測                                                                                            |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                      Anirban                        |
|    24     |                強化學習導論                | [Reinforcement learning](8-Reinforcement/README.md) | 使用 Q-Learning 認識強化學習                                                                                               |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|    25     |                幫助 Peter 避免狼群！🐺                | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                               |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  附錄  |            真實世界的機器學習場景與應用            |      [ML in the Wild](9-Real-World/README.md)       | 經典機器學習在真實世界中的有趣且啟發性的應用                                                                                 |                                             [課程](9-Real-World/1-Applications/README.md)                                              |                        團隊                          |
|  附錄  |             使用 RAI 儀表板來偵錯機器學習模型              |      [ML in the Wild](9-Real-World/README.md)       | 使用負責任的人工智慧儀表板元件來進行機器學習模型偵錯                                                                         |                                             [課程](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                      Ruth Yakubu                       |

> [在我們的 Microsoft Learn 收藏中找到此課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線瀏覽

您可以使用 [Docsify](https://docsify.js.org/#/) 離線瀏覽本文件。請 fork 此 repo，並在本地機器上[安裝 Docsify](https://docsify.js.org/#/quickstart)，然後於本 repo 根目錄輸入 `docsify serve`，網站將架設在本機的 3000 埠：`localhost:3000`。

## PDFs

課程綱要的 PDF（含連結）請見[此處](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)。

## 🎒 其他課程

我們團隊還製作其他課程！請參考：

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j 初學者](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js 初學者](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)
[![LangChain 初學者](https://img.shields.io/badge/LangChain%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://github.com/microsoft/langchain-for-beginners?WT.mc_id=m365-94501-dwahlin)
---

### Azure / Edge / MCP / Agents
[![AZD 初學者](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI 初學者](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 MCP](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 AI 代理人](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### 生成式 AI 系列
[![初學者生成式 AI](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### 核心學習
[![初學者機器學習](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者資料科學](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 AI](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者資安](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![初學者網頁開發](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者物聯網](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 XR 開發](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot 系列
[![AI 配對程式設計 Copilot](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![C#/.NET Copilot](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot 冒險](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## 尋求協助

如果您在構建 AI 應用程式時遇到困難或有任何問題。加入其他學習者和經驗豐富的開發者一起討論 MCP。這是一個支援性的社群，歡迎提出問題並自由分享知識。

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

如果您在構建過程中有產品反饋或錯誤，請訪問：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)
## 額外學習提示

- 每堂課後複習筆記本以增進理解。
- 練習自行實作演算法。
- 運用所學概念探索實際資料集。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意自動翻譯可能包含錯誤或不準確之處。原文文件的母語版本應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤譯負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->