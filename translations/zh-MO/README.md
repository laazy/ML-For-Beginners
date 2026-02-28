[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 多語言支援

#### 透過 GitHub Action 支援（自動化且始終保持最新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[阿拉伯語](../ar/README.md) | [孟加拉語](../bn/README.md) | [保加利亞語](../bg/README.md) | [緬甸語 (Myanmar)](../my/README.md) | [中文 (簡體)](../zh-CN/README.md) | [中文 (繁體，香港)](../zh-HK/README.md) | [中文 (繁體，澳門)](./README.md) | [中文 (繁體，臺灣)](../zh-TW/README.md) | [克羅地亞語](../hr/README.md) | [捷克語](../cs/README.md) | [丹麥語](../da/README.md) | [荷蘭語](../nl/README.md) | [愛沙尼亞語](../et/README.md) | [芬蘭語](../fi/README.md) | [法語](../fr/README.md) | [德語](../de/README.md) | [希臘語](../el/README.md) | [希伯來語](../he/README.md) | [印地語](../hi/README.md) | [匈牙利語](../hu/README.md) | [印尼語](../id/README.md) | [義大利語](../it/README.md) | [日語](../ja/README.md) | [坎納達語](../kn/README.md) | [韓語](../ko/README.md) | [立陶宛語](../lt/README.md) | [馬來語](../ms/README.md) | [馬拉雅拉姆語](../ml/README.md) | [馬拉地語](../mr/README.md) | [尼泊爾語](../ne/README.md) | [奈及利亞皮欽語](../pcm/README.md) | [挪威語](../no/README.md) | [波斯語 (法爾西語)](../fa/README.md) | [波蘭語](../pl/README.md) | [葡萄牙語 (巴西)](../pt-BR/README.md) | [葡萄牙語 (葡萄牙)](../pt-PT/README.md) | [旁遮普語 (Gurmukhi)](../pa/README.md) | [羅馬尼亞語](../ro/README.md) | [俄羅斯語](../ru/README.md) | [塞爾維亞語 (西里爾字母)](../sr/README.md) | [斯洛伐克語](../sk/README.md) | [斯洛維尼亞語](../sl/README.md) | [西班牙語](../es/README.md) | [斯瓦希里語](../sw/README.md) | [瑞典語](../sv/README.md) | [他加祿語 (菲律賓語)](../tl/README.md) | [泰米爾語](../ta/README.md) | [泰盧固語](../te/README.md) | [泰語](../th/README.md) | [土耳其語](../tr/README.md) | [烏克蘭語](../uk/README.md) | [烏爾都語](../ur/README.md) | [越南語](../vi/README.md)

> **想要本地複製？**
>
> 此儲存庫包含 50 多種語言翻譯，會大幅增加下載大小。若想不帶翻譯內容複製，請使用稀疏檢出：
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
> 這樣可以更快速下載，且包含完成課程所需的一切。
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### 加入我們的社群

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們正在舉辦 Discord 的 AI 學習系列活動，請於 2025 年 9 月 18 日至 30 日前往 [Learn with AI Series](https://aka.ms/learnwithai/discord) 瞭解更多並加入。我們將分享使用 GitHub Copilot 進行資料科學的技巧與方法。

![Learn with AI series](../../translated_images/zh-MO/3.9b58fd8d6c373c20.webp)

# 機器學習初學者課程綱要

> 🌍 透過世界文化的視角，帶你環遊世界探索機器學習 🌍

微軟的雲端擁護者很高興提供一套為期 12 週、共 26 課的完整課程，主題為 **機器學習**。本課程聚焦於所謂的 **經典機器學習**，主要使用 Scikit-learn 函式庫，避開深度學習部分（相關內容可見於我們的[初學者 AI 課程](https://aka.ms/ai4beginners)）。也建議和我們的[初學者資料科學課程](https://aka.ms/ds4beginners) 搭配學習！

跟我們一起周遊世界，將這些經典技術應用於各地的資料。每課包含課前與課後測驗、書面教學指引、解答、作業與更多。採用專案導向教學法，讓你邊學邊做，是幫助新技能扎根的有效方法。

**✍️ 衷心感謝我們的作者** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Anirban Mukherjee, Ornella Altunyan, Ruth Yakubu 與 Amy Boyd

**🎨 以及感謝我們的插畫師** Tomomi Imura, Dasani Madipalli 與 Jen Looper

**🙏 特別感謝 🙏 微軟學生大使的作者、審閱者與內容貢獻者**，尤其是 Rishit Dagli, Muhammad Sakib Khan Inan, Rohan Raj, Alexandru Petrescu, Abhishek Jaiswal, Nawrin Tabassum, Ioan Samuila 與 Snigdha Agarwal

**🤩 額外感謝微軟學生大使 Eric Wanjau, Jasleen Sondhi 與 Vidushi Gupta 對我們的 R 課程貢獻！**

# 開始使用

請依序操作：
1. **分叉此儲存庫（Fork）**：點擊本頁右上角的「Fork」按鈕。
2. **複製儲存庫（Clone）**：`git clone https://github.com/microsoft/ML-For-Beginners.git`

> [本課程的所有附加資源請見我們的 Microsoft Learn 集合](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要幫助？** 請查閱我們的[Troubleshooting Guide](TROUBLESHOOTING.md)，解決常見的安裝、設定及執行問題。

**[學生](https://aka.ms/student-page)**，請將此課程的整個儲存庫 Fork 到你的 GitHub 帳號，並自行或團體完成練習：

- 先做課前熱身測驗。
- 閱讀課程教材並完成練習，並在每個知識點停下來思考。
- 盡量透過理解課程內容來建立專案，而非直接運行解答程式碼；不過解答可在每個專案課的 `/solution` 資料夾找到。
- 做完課後測驗。
- 完成挑戰。
- 完成指定作業。
- 完成每組課程後，歡迎前往[討論區](https://github.com/microsoft/ML-For-Beginners/discussions)藉由填寫對應的 PAT 評分表「大聲學習」。PAT（進度評估工具）是用來提升學習的評分表。你也可以對其他人的 PAT 做出回應，共同進步。

> 若要進一步學習，我們建議跟隨這些[Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott)模組及學習路徑。

**教師**，我們已[提供一些建議](for-teachers.md) 關於如何使用此課程。

---

## 影片導覽

部分課程附有短片教學。你可在課程中內嵌觀看，或者前往[微軟開發者 YouTube 頻道的 ML 入門播放清單](https://aka.ms/ml-beginners-videos)點擊下面的圖片瀏覽。

[![ML for beginners banner](../../translated_images/zh-MO/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 團隊介紹

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**GIF 製作:** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 點擊上方圖片觀看有關本專案及其創作者的影片！

---

## 教學理念

在設計本課程時，我們選擇了兩項教學原則：確保全程是動手操作的 **專案導向**，並且包含 **頻繁測驗**。另外，課程設計有共同的 **主題** 以增加連貫性。

透過內容與專案的緊密結合，學習過程更能吸引學生注意力，且有助於鞏固概念。課前的低壓力測驗讓學生訂立學習目標，課後的再度測驗則支持加深記憶。課程有彈性且趣味十足，可完整學習或擷取部分主題。專案自簡入深，於 12 週循環末達到複雜度高峰。最後也有針對機器學習在現實世界應用的補充內容，可用作加分題或討論基礎。

> 請參考我們的[行為守則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯指南](TRANSLATIONS.md) 以及 [故障排除](TROUBLESHOOTING.md) 文件。我們十分歡迎您建設性的反饋！

## 每堂課包含

- 選擇性手繪筆記
- 選擇性補充影片
- 影片導覽（部分課程）
- [課前熱身測驗](https://ff-quizzes.netlify.app/en/ml/)
- 書面課程內容
- 專案課程附分步驟指引
- 知識檢核
- 挑戰題
- 補充閱讀資料
- 作業
- [課後測驗](https://ff-quizzes.netlify.app/en/ml/)
> **關於語言的說明**：這些課程主要以 Python 撰寫，但也有許多課程提供 R 版本。要完成 R 課程，請前往 `/solution` 資料夾，尋找 R 課程。它們包含 .rmd 副檔名，代表一個 **R Markdown** 檔案，可簡單定義為在 `Markdown 文件` 中嵌入 `程式碼區塊`（R 或其他語言）及 `YAML 標頭` （用於指導如何格式化輸出，如 PDF）。因此，它成為資料科學的典範編寫框架，允許你結合程式碼、其輸出及你的想法，並用 Markdown 記錄。此外，R Markdown 文件可渲染輸出格式如 PDF、HTML 或 Word。

> **關於測驗的說明**：所有測驗都包含在 [Quiz App 資料夾](../../quiz-app) 中，共有 52 個測驗，每個包含三個問題。測驗由課程內連結，可本地執行測驗應用程式；請依照 `quiz-app` 資料夾中的說明，在本地架設或部署至 Azure。

| Lesson Number |                             Topic                              |                   Lesson Grouping                   | Learning Objectives                                                                                                             |                                                              Linked Lesson                                                               |                        Author                        |
| :-----------: | :------------------------------------------------------------: | :-------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
|      01       |                機器學習簡介                                   |      [Introduction](1-Introduction/README.md)       | 學習機器學習的基本概念                                                                                                            |                                             [Lesson](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                       |
|      02       |                機器學習的歷史                                   |      [Introduction](1-Introduction/README.md)       | 了解該領域的歷史背景                                                                                                              |                                            [Lesson](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen and Amy                      |
|      03       |                 公平性與機器學習                                |      [Introduction](1-Introduction/README.md)       | 探討學生在建立及應用 ML 模型時應考慮的重要哲學性公平議題                                                                           |                                              [Lesson](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|      04       |                機器學習技術                                   |      [Introduction](1-Introduction/README.md)       | 機器學習研究人員用來建立 ML 模型的技術                                                                                           |                                          [Lesson](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris and Jen                     |
|      05       |                   迴歸簡介                                     |        [Regression](2-Regression/README.md)         | 使用 Python 和 Scikit-learn 入門迴歸模型                                                                                          |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|      06       |                北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 可視化與清理資料，以備 ML 使用                                                                                                   |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|      07       |                北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 建立線性與多項式迴歸模型                                                                                                         |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen and Dmitry • Eric Wanjau       |
|      08       |                北美南瓜價格 🎃                                  |        [Regression](2-Regression/README.md)         | 建立邏輯斯迴歸模型                                                                                                               |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|      09       |                          網頁應用 🔌                            |           [Web App](3-Web-App/README.md)            | 建置可使用你的訓練模型的網頁應用                                                                                                |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|      10       |                 分類簡介                                      |    [Classification](4-Classification/README.md)     | 清理、準備及視覺化資料；分類介紹                                                                                                  | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen and Cassie • Eric Wanjau |
|      11       |             美味亞洲與印度料理 🍜                              |    [Classification](4-Classification/README.md)     | 分類器介紹                                                                                                                     | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen and Cassie • Eric Wanjau |
|      12       |             美味亞洲與印度料理 🍜                              |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                                    | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen and Cassie • Eric Wanjau |
|      13       |             美味亞洲與印度料理 🍜                              |    [Classification](4-Classification/README.md)     | 使用你的模型建立推薦網頁應用                                                                                                |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|      14       |                   分群簡介                                    |        [Clustering](5-Clustering/README.md)         | 清理、準備及視覺化資料；分群介紹                                                                                                |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|      15       |              探索尼日利亞音樂喜好 🎧                          |        [Clustering](5-Clustering/README.md)         | 探索 K-Means 分群方法                                                                                                          |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|      16       |        自然語言處理簡介 ☕️                                   |   [Natural language processing](6-NLP/README.md)    | 透過建立簡單機器人學習 NLP 基礎                                                                                               |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|      17       |                      常見的 NLP 任務 ☕️                      |   [Natural language processing](6-NLP/README.md)    | 深化 NLP 知識，理解處理語言結構時需執行的常見任務                                                                              |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|      18       |             翻譯與情感分析 ♥️                                 |   [Natural language processing](6-NLP/README.md)    | 使用 Jane Austen 進行情感分析與翻譯                                                                                           |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|      19       |                  歐洲浪漫旅館 ♥️                               |   [Natural language processing](6-NLP/README.md)    | 利用旅館評論進行情感分析 1                                                                                                    |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|      20       |                  歐洲浪漫旅館 ♥️                               |   [Natural language processing](6-NLP/README.md)    | 利用旅館評論進行情感分析 2                                                                                                    |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|      21       |            時間序列預測簡介                                   |        [Time series](7-TimeSeries/README.md)        | 時間序列預測介紹                                                                                                               |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|      22       | ⚡️ 世界電力使用 ⚡️ - 使用 ARIMA 進行時間序列預測              |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 進行時間序列預測                                                                                                   |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|      23       | ⚡️ 世界電力使用 ⚡️ - 使用 SVR 進行時間序列預測                |        [Time series](7-TimeSeries/README.md)        | 使用支援向量回歸器進行時間序列預測                                                                                           |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|      24       |             強化學習簡介                                      | [Reinforcement learning](8-Reinforcement/README.md) | 使用 Q-Learning 進行強化學習簡介                                                                                              |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|      25       |                 幫助彼得躲避狼！🐺                             | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                                 |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  Postscript   |            實際機器學習場景與應用                             |      [ML in the Wild](9-Real-World/README.md)       | 經典機器學習有趣且富啟發性的實際應用                                                                                           |                                             [Lesson](9-Real-World/1-Applications/README.md)                                              |                         Team                         |
|  Postscript   |            使用 RAI 儀表板進行機器學習模型除錯               |      [ML in the Wild](9-Real-World/README.md)       | 使用 Responsible AI 儀表板組件進行機器學習模型除錯                                                                             |                                             [Lesson](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [在我們的 Microsoft Learn 集合中找到此課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線存取

你可以使用 [Docsify](https://docsify.js.org/#/) 離線執行此文件。分支此儲存庫，在你的本機安裝 [Docsify](https://docsify.js.org/#/quickstart)，然後在此儲存庫根目錄輸入 `docsify serve`。網站將會在本地主機的 3000 埠執行：`localhost:3000`。

## PDF 檔案

在[此處](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)找到課程大綱的帶連結 PDF。

## 🎒 其他課程

我們團隊還製作其他課程！快來看看：

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)
[![LangChain for Beginners](https://img.shields.io/badge/LangChain%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://github.com/microsoft/langchain-for-beginners?WT.mc_id=m365-94501-dwahlin)
---

### Azure / Edge / MCP / Agents
[![AZD for Beginners](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI for Beginners](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![MCP for Beginners](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![AI Agents for Beginners](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### 生成式 AI 系列
[![Generative AI for Beginners](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generative AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generative AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generative AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### 核心學習
[![ML for Beginners](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Data Science for Beginners](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![AI for Beginners](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Cybersecurity for Beginners](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Web Dev for Beginners](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT for Beginners](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR Development for Beginners](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot 系列
[![Copilot for AI Paired Programming](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot for C#/.NET](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot Adventure](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## 尋求幫助

如果你遇到困難或對建構 AI 應用程式有任何疑問，歡迎加入一同學習的夥伴與經驗豐富的開發者討論 MCP。這是一個支持性的社群，歡迎提問並自由分享知識。

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

如果你有產品反饋或在建構過程中遇到錯誤，請造訪：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)
## 進階學習提示

- 每堂課後複習筆記本，以加深理解。
- 練習自行實作演算法。
- 利用學到的概念探索實際數據集。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
此文件由人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 所翻譯。儘管我們致力於準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原文件的母語版本應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用此翻譯而引起的任何誤解或錯誤解讀承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->