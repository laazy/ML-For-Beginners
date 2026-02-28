[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 多語言支援

#### 通過 GitHub Action 支援（自動化及一直保持最新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[阿拉伯語](../ar/README.md) | [孟加拉語](../bn/README.md) | [保加利亞語](../bg/README.md) | [緬甸語](../my/README.md) | [中文（簡體）](../zh-CN/README.md) | [中文（繁體，香港）](./README.md) | [中文（繁體，澳門）](../zh-MO/README.md) | [中文（繁體，台灣）](../zh-TW/README.md) | [克羅地亞語](../hr/README.md) | [捷克語](../cs/README.md) | [丹麥語](../da/README.md) | [荷蘭語](../nl/README.md) | [愛沙尼亞語](../et/README.md) | [芬蘭語](../fi/README.md) | [法語](../fr/README.md) | [德語](../de/README.md) | [希臘語](../el/README.md) | [希伯來語](../he/README.md) | [印地語](../hi/README.md) | [匈牙利語](../hu/README.md) | [印尼語](../id/README.md) | [意大利語](../it/README.md) | [日語](../ja/README.md) | [卡納達語](../kn/README.md) | [韓語](../ko/README.md) | [立陶宛語](../lt/README.md) | [馬來語](../ms/README.md) | [馬拉雅拉姆語](../ml/README.md) | [馬拉地語](../mr/README.md) | [尼泊爾語](../ne/README.md) | [奈及利亞洋芋拼音](../pcm/README.md) | [挪威語](../no/README.md) | [波斯語（法爾斯語）](../fa/README.md) | [波蘭語](../pl/README.md) | [巴西葡萄牙語](../pt-BR/README.md) | [葡萄牙語（葡萄牙）](../pt-PT/README.md) | [旁遮普語（Gurmukhi）](../pa/README.md) | [羅馬尼亞語](../ro/README.md) | [俄語](../ru/README.md) | [塞爾維亞語（西里爾字母）](../sr/README.md) | [斯洛伐克語](../sk/README.md) | [斯洛維尼亞語](../sl/README.md) | [西班牙語](../es/README.md) | [斯瓦希里語](../sw/README.md) | [瑞典語](../sv/README.md) | [他加祿語（菲律賓語）](../tl/README.md) | [泰米爾語](../ta/README.md) | [泰盧固語](../te/README.md) | [泰語](../th/README.md) | [土耳其語](../tr/README.md) | [烏克蘭語](../uk/README.md) | [烏爾都語](../ur/README.md) | [越南語](../vi/README.md)

> **偏好本地複製？**
>
> 此存儲庫包含 50 多種語言翻譯，這會大幅增加下載容量。要不含翻譯地複製，請使用稀疏檢出：
>
> **Bash / macOS / Linux:**
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
>
> **CMD（Windows）:**
> ```cmd
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone "/*" "!translations" "!translated_images"
> ```
>
> 這樣可以讓你用更快的速度下載所有完成課程所需的內容。
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### 加入我們的社區

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們正進行 Discord AI 學習系列，2025 年 9 月 18 日至 30 日，了解更多並加入我們的 [Learn with AI Series](https://aka.ms/learnwithai/discord)。屆時你將獲得使用 GitHub Copilot 進行資料科學的提示和技巧。

![AI 學習系列](../../translated_images/zh-HK/3.9b58fd8d6c373c20.webp)

# 初學者機器學習課程

> 🌍 一起環遊世界並透過世界文化探索機器學習 🌍

微軟的雲端倡導者們很高興提供一套為期 12 週、包含 26 課的課程，專注於**機器學習**。本課程將介紹有時稱為**經典機器學習**的技術，主要使用 Scikit-learn 函式庫，避免深度學習內容，後者在我們的 [初學者 AI 課程](https://aka.ms/ai4beginners) 中涵蓋。同時可搭配我們的 [初學者資料科學課程](https://aka.ms/ds4beginners) 一起學習！

跟我們一起環遊世界，將這些經典技術應用於來自全球多個地區的資料。每課皆包含課前和課後小測驗、書面指導、解答、作業等。我們採用以專案為導向的教學法，讓你邊建構邊學習，是新技能長期掌握的經驗證方式。

**✍️ 誠摯感謝作者** Jen Looper、Stephen Howell、Francesca Lazzeri、Tomomi Imura、Cassie Breviu、Dmitry Soshnikov、Chris Noring、Anirban Mukherjee、Ornella Altunyan、Ruth Yakubu 以及 Amy Boyd

**🎨 同時感謝插畫師** Tomomi Imura、Dasani Madipalli 與 Jen Looper

**🙏 特別感謝🙏 微軟學生大使團隊的作者、審閱者與內容貢獻者**，包括 Rishit Dagli、Muhammad Sakib Khan Inan、Rohan Raj、Alexandru Petrescu、Abhishek Jaiswal、Nawrin Tabassum、Ioan Samuila 與 Snigdha Agarwal

**🤩 額外感謝微軟學生大使 Eric Wanjau、Jasleen Sondhi 與 Vidushi Gupta 參與 R 課程製作！**

# 開始學習

請遵循以下步驟：
1. **分叉此存儲庫**：點擊本頁右上角的「Fork」按鈕。
2. **複製存儲庫**： `git clone https://github.com/microsoft/ML-For-Beginners.git`

> [在我們的 Microsoft Learn 合集中找到本課程的所有附加資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要協助？** 請查看我們的 [疑難排解指南](TROUBLESHOOTING.md)，了解安裝、設置及運行課程常見問題的解決方案。


**[學生](https://aka.ms/student-page)** 適用方法：將整個存儲庫分叉到你的 GitHub 帳戶，在個人或小組內完成練習：

- 從課前小測開始。
- 閱讀課程內容並完成各項活動，遇到檢核點時暫停並反思。
- 嘗試透過理解課程自行創建專案，而非直接運行解答程式碼；不過解答程式碼會放在各面向專案課程的 `/solution` 資料夾內。
- 進行課後小測驗。
- 完成挑戰任務。
- 完成作業。
- 完成一組課程後，請訪問 [討論區](https://github.com/microsoft/ML-For-Beginners/discussions)，透過填寫 PAT 標準並「大聲學習」。'PAT' 是進度評估工具，一種用來促進學習的標準表格，你也可以對其他人的 PAT 回應，讓大家一起進步。

> 進階學習推薦追蹤這些 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 模組和學習路徑。

**老師們**，我們提供了一些關於如何使用本課程的[建議](for-teachers.md)。

---

## 影片導覽

部分課程有短影片導覽。你可以在課程中內嵌觀看，或點擊下方圖片至 [Microsoft Developer YouTube 頻道的「初學者機器學習」播放清單](https://aka.ms/ml-beginners-videos)。

[![ML for beginners banner](../../translated_images/zh-HK/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 認識團隊

[![推廣影片](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**Gif 製作：** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 點擊上方圖片觀看有關此專案及團隊成員的影片！

---

## 教學法

我們建構此課程時，採取了兩大教學原則：確保課程是以 **專案為基礎的實作**，以及包含 **頻繁的小測驗**。此外，也設計了共通的 **主題** 以保持內容一致性。

透過讓內容與專案對齊，增加學生的參與度並加強概念記憶。課前的小測驗有助於設定學生學習目標，課後小測則促進知識鞏固。此課程彈性且有趣，可整套完成或分段學習。專案從簡單開始，逐漸在 12 週循環結束時達到複雜度。課程末端亦包含一段關於機器學習在現實世界中的應用說明，適合作為額外學分或討論主題。

> 請參閱我們的 [行為守則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯說明](TRANSLATIONS.md) 及 [疑難排解](TROUBLESHOOTING.md) 指南。我們歡迎您的建設性意見反饋！

## 每節課內容包含

- 可選擇的素描筆記
- 可選擇的補充影片
- 影片導覽（部分課程）
- [課前暖身小測驗](https://ff-quizzes.netlify.app/en/ml/)
- 書面課程
- 專案課程的逐步建構指導
- 知識檢核
- 挑戰任務
- 補充閱讀資料
- 作業
- [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)
> **關於語言的說明**：這些課程主要以 Python 編寫，但也有許多課程提供 R 版本。若要完成 R 課程，請前往 `/solution` 資料夾並尋找 R 課程。這些課程包含 .rmd 副檔名，代表一個 **R Markdown** 檔案，簡單來說，它是將 `程式碼區塊`（R 或其他語言）與 `YAML 標頭`（用以指示如何格式化輸出，例如 PDF）嵌入於 `Markdown 文件` 中。基於此，R Markdown 作為資料科學的典範編輯架構，可讓你於 Markdown 中同時撰寫程式碼、其輸出與你的想法。此外，R Markdown 文件可以渲染為 PDF、HTML 或 Word 等輸出格式。

> **關於測驗的說明**：所有測驗皆收錄在 [Quiz App folder](../../quiz-app) 中，總計 52 組，每組包含三個問題。這些測驗會在課程內連結，但你也可以在本地執行測驗應用程式；請依照 `quiz-app` 資料夾中的說明，在本地端架設或部署至 Azure。

| 課程編號 |                                 主題                                  |                   課程群組                    | 學習目標                                                                                                                        |                                                               相關課程                                                                |                       作者                       |
| :-------: | :------------------------------------------------------------------: | :--------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------: |
|    01     |                      機器學習入門介紹                                |      [Introduction](1-Introduction/README.md)       | 學習機器學習的基本概念                                                                                                         |                                              [Lesson](1-Introduction/1-intro-to-ML/README.md)                                              |                      Muhammad                    |
|    02     |                      機器學習的歷史                                  |      [Introduction](1-Introduction/README.md)       | 了解此領域的歷史背景                                                                                                           |                                             [Lesson](1-Introduction/2-history-of-ML/README.md)                                             |                    Jen 與 Amy                     |
|    03     |                      公平性與機器學習                                |      [Introduction](1-Introduction/README.md)       | 建立與應用機器學習模型時，應考慮的重要哲學公平性議題                                                                         |                                               [Lesson](1-Introduction/3-fairness/README.md)                                               |                       Tomomi                      |
|    04     |                      機器學習技術                                    |      [Introduction](1-Introduction/README.md)       | 機器學習研究者用於建立機器學習模型的技術                                                                                       |                                           [Lesson](1-Introduction/4-techniques-of-ML/README.md)                                            |                    Chris 與 Jen                    |
|    05     |                      迴歸入門                                        |        [Regression](2-Regression/README.md)         | 開始使用 Python 與 Scikit-learn 建立迴歸模型                                                                                   |          [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)           |                   Jen • Eric Wanjau                |
|    06     |                      北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 資料視覺化與清理，為機器學習做準備                                                                                            |           [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |                   Jen • Eric Wanjau                |
|    07     |                      北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 建立線性及多項式迴歸模型                                                                                                      |         [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)         |             Jen 與 Dmitry • Eric Wanjau          |
|    08     |                      北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 建立邏輯迴歸模型                                                                                                              |       [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)       |                   Jen • Eric Wanjau                |
|    09     |                          網頁應用 🔌                                 |           [Web App](3-Web-App/README.md)            | 建立一個網頁應用以使用你訓練好的模型                                                                                        |                                               [Python](3-Web-App/1-Web-App/README.md)                                                |                        Jen                         |
|    10     |                      分類入門                                        |    [Classification](4-Classification/README.md)     | 數據清理、準備與視覺化；分類介紹                                                                                            |   [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)   |              Jen 與 Cassie • Eric Wanjau          |
|    11     |                      美味亞洲與印度料理 🍜                            |    [Classification](4-Classification/README.md)     | 分類器介紹                                                                                                                  |   [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html)  |              Jen 與 Cassie • Eric Wanjau          |
|    12     |                      美味亞洲與印度料理 🍜                            |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                                 |   [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html)  |              Jen 與 Cassie • Eric Wanjau          |
|    13     |                      美味亞洲與印度料理 🍜                            |    [Classification](4-Classification/README.md)     | 使用你的模型建立推薦網頁應用                                                                                               |                                               [Python](4-Classification/4-Applied/README.md)                                               |                        Jen                         |
|    14     |                      分群入門                                        |        [Clustering](5-Clustering/README.md)         | 數據清理、準備與視覺化；分群介紹                                                                                            |        [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)        |                   Jen • Eric Wanjau                |
|    15     |                      探索奈及利亞音樂品味 🎧                          |        [Clustering](5-Clustering/README.md)         | 探索 K 平均 (K-Means) 分群方法                                                                                            |          [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)          |                   Jen • Eric Wanjau                |
|    16     |                      自然語言處理入門 ☕️                             |   [Natural language processing](6-NLP/README.md)    | 透過建置簡易聊天機器人學習自然語言處理基礎                                                                               |                                            [Python](6-NLP/1-Introduction-to-NLP/README.md)                                             |                      Stephen                       |
|    17     |                      常見的自然語言處理任務 ☕️                       |   [Natural language processing](6-NLP/README.md)    | 透析常見 NLP 任務以深化對語言結構的理解                                                                                 |                                                      [Python](6-NLP/2-Tasks/README.md)                                                    |                      Stephen                       |
|    18     |                      翻譯與情感分析 ♥️                              |   [Natural language processing](6-NLP/README.md)    | 利用 Jane Austen 文本進行情感分析與翻譯                                                                                   |                                           [Python](6-NLP/3-Translation-Sentiment/README.md)                                            |                      Stephen                       |
|    19     |                      歐洲浪漫飯店 ♥️                                |   [Natural language processing](6-NLP/README.md)    | 用飯店評論進行情感分析 1                                                                                                  |                                                  [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                               |                      Stephen                       |
|    20     |                      歐洲浪漫飯店 ♥️                                |   [Natural language processing](6-NLP/README.md)    | 用飯店評論進行情感分析 2                                                                                                  |                                                  [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                               |                      Stephen                       |
|    21     |                      時間序列預測入門                                |        [Time series](7-TimeSeries/README.md)        | 時間序列預測介紹                                                                                                         |                                            [Python](7-TimeSeries/1-Introduction/README.md)                                             |                   Francesca                        |
|    22     | ⚡️ 世界電力使用 ⚡️ - 使用 ARIMA 進行時間序列預測                     |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 進行時間序列預測                                                                                              |                                                [Python](7-TimeSeries/2-ARIMA/README.md)                                                |                   Francesca                        |
|    23     | ⚡️ 世界電力使用 ⚡️ - 使用 SVR 進行時間序列預測                       |        [Time series](7-TimeSeries/README.md)        | 使用支持向量回歸器 (SVR) 進行時間序列預測                                                                                  |                                                [Python](7-TimeSeries/3-SVR/README.md)                                                 |                    Anirban                         |
|    24     |                      強化學習入門                                    | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習基礎介紹：Q-Learning                                                                                                |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                             |                      Dmitry                         |
|    25     |                      幫彼得避開狼！ 🐺                               | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                              |                                               [Python](8-Reinforcement/2-Gym/README.md)                                                 |                      Dmitry                         |
|  後記     |                      現實世界中的機器學習場景與應用                   |      [ML in the Wild](9-Real-World/README.md)       | 傳統機器學習在真實世界的有趣且啟發性的應用                                                                                 |                                            [Lesson](9-Real-World/1-Applications/README.md)                                             |                        團隊                         |
|  後記     |                      使用 RAI 儀表板進行機器學習模型偵錯             |      [ML in the Wild](9-Real-World/README.md)       | 使用 Responsible AI 儀表板元件進行機器學習模型偵錯                                                                        |                                            [Lesson](9-Real-World/2-Debugging-ML-Models/README.md)                                             |                   Ruth Yakubu                      |

> [在我們的 Microsoft Learn 集合中找到本課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線存取

你可以使用 [Docsify](https://docsify.js.org/#/) 離線運行本文件。將此資料庫 fork 到本地機器，並安裝 [Docsify](https://docsify.js.org/#/quickstart)，接著於此資料庫根目錄下執行 `docsify serve`。網站將會在本機端 3000 埠口提供服務：`localhost:3000`。

## PDF

這裡可找到課程大綱的 PDF 並附帶連結 [here](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)。

## 🎒 其他課程

我們團隊還有其他課程！請參考：

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)
[![LangChain for Beginners](https://img.shields.io/badge/LangChain%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://github.com/microsoft/langchain-for-beginners?WT.mc_id=m365-94501-dwahlin)
---

### Azure / Edge / MCP / Agents
[![AZD for Beginners](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI for Beginners](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 MCP](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 AI 代理](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### 生成式 AI 系列
[![初學者生成式 AI](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![生成式 AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### 核心學習
[![初學者機器學習](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者數據科學](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 AI](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者網絡安全](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![初學者網頁開發](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者物聯網](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![初學者 XR 開發](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot 系列
[![為 AI 配對編程設計的 Copilot](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![為 C#/.NET 設計的 Copilot](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot 冒險](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## 尋求協助

如果你在建立 AI 應用程式時遇到困難或有任何疑問，歡迎加入學習者和資深開發者的討論，一同參與 MCP 社群。這是一個支持性的社群，歡迎提出問題並自由分享知識。

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

如果你在開發過程中有產品反饋或遇到錯誤，請造訪：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)
## 額外學習貼士

- 每堂課後復習筆記本，加深理解。
- 練習自行實作算法。
- 利用所學概念探索真實世界數據集。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件由人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於確保準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議採用專業人工翻譯。本公司對因使用此翻譯而引致的任何誤解或誤讀概不負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->