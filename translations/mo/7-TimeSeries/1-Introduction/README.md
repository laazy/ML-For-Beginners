<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3150d40f36a77857316ecaed5f31e856",
  "translation_date": "2025-08-29T20:45:14+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "mo"
}
-->
# 時間序列預測簡介

![時間序列的手繪筆記摘要](../../../../translated_images/ml-timeseries.fb98d25f1013fc0c59090030080b5d1911ff336427bec31dbaf1ad08193812e9.mo.png)

> 手繪筆記由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 提供

在本課程及接下來的課程中，你將學習一些關於時間序列預測的知識。這是一個有趣且有價值的機器學習科學家技能，雖然它不像其他主題那麼廣為人知。時間序列預測就像一個“水晶球”：根據某個變量（例如價格）的過去表現，你可以預測其未來的潛在價值。

[![時間序列預測簡介](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "時間序列預測簡介")

> 🎥 點擊上方圖片觀看關於時間序列預測的影片

## [課前測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41/)

這是一個實用且有趣的領域，對於商業來說具有實際價值，因為它可以直接應用於定價、庫存和供應鏈問題。雖然深度學習技術已經開始被用來獲得更多洞察以更好地預測未來表現，但時間序列預測仍然是一個主要由經典機器學習技術驅動的領域。

> 賓州州立大學的實用時間序列課程可以在[這裡](https://online.stat.psu.edu/stat510/lesson/1)找到

## 簡介

假設你管理一組智慧停車計時器，這些計時器提供有關它們使用頻率和使用時長的數據。

> 如果你能根據計時器的過去表現，根據供需法則預測其未來價值，會怎麼樣？

準確預測何時採取行動以實現目標是一個挑戰，而這可以通過時間序列預測來解決。雖然在繁忙時段收取更高的停車費可能不會讓人們高興，但這無疑是一種為清潔街道創造收入的好方法！

讓我們來探索一些時間序列算法的類型，並開始一個筆記本來清理和準備一些數據。你將分析的數據來自 GEFCom2014 預測競賽。它包含了 2012 年到 2014 年之間 3 年的每小時電力負載和溫度值。根據電力負載和溫度的歷史模式，你可以預測電力負載的未來值。

在這個例子中，你將學習如何僅使用歷史負載數據來預測下一個時間步的值。然而，在開始之前，了解背後的原理是很有幫助的。

## 一些定義

當遇到“時間序列”這個術語時，你需要了解它在不同上下文中的使用。

🎓 **時間序列**

在數學中，“時間序列是一系列按時間順序索引（或列出或繪製）的數據點。最常見的是，時間序列是在時間上以相等間隔採集的序列。”時間序列的一個例子是[道瓊斯工業平均指數](https://wikipedia.org/wiki/Time_series)的每日收盤價。時間序列圖和統計建模的使用經常出現在信號處理、天氣預測、地震預測以及其他事件發生並可以隨時間繪製數據點的領域。

🎓 **時間序列分析**

時間序列分析是對上述時間序列數據的分析。時間序列數據可以採取不同的形式，包括“中斷時間序列”，它檢測時間序列在中斷事件前後的演變模式。所需的分析類型取決於數據的性質。時間序列數據本身可以是數字或字符的序列。

分析使用了多種方法，包括頻域和時域、線性和非線性等。[了解更多](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm)關於分析這類數據的方法。

🎓 **時間序列預測**

時間序列預測是使用模型根據過去收集的數據所顯示的模式來預測未來值。雖然可以使用回歸模型來探索時間序列數據，並將時間索引作為圖上的 x 變量，但這類數據最好使用特殊類型的模型進行分析。

時間序列數據是一個有序觀測值的列表，不同於可以通過線性回歸分析的數據。最常見的模型是 ARIMA，這是“自回歸整合移動平均”的縮寫。

[ARIMA 模型](https://online.stat.psu.edu/stat510/lesson/1/1.1)“將序列的當前值與過去的值和過去的預測誤差相關聯。”它們最適合分析按時間排序的數據。

> ARIMA 模型有多種類型，你可以在[這裡](https://people.duke.edu/~rnau/411arim.htm)了解更多，並在下一課中進一步探討。

在下一課中，你將使用[單變量時間序列](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm)構建一個 ARIMA 模型，該模型專注於一個隨時間變化的變量。這類數據的一個例子是[這個數據集](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm)，它記錄了 Mauna Loa 天文台的每月二氧化碳濃度：

|   CO2   | YearMonth | Year  | Month |
| :-----: | :-------: | :---: | :---: |
| 330.62  |  1975.04  | 1975  |   1   |
| 331.40  |  1975.13  | 1975  |   2   |
| 331.87  |  1975.21  | 1975  |   3   |
| 333.18  |  1975.29  | 1975  |   4   |
| 333.92  |  1975.38  | 1975  |   5   |
| 333.43  |  1975.46  | 1975  |   6   |
| 331.85  |  1975.54  | 1975  |   7   |
| 330.01  |  1975.63  | 1975  |   8   |
| 328.51  |  1975.71  | 1975  |   9   |
| 328.41  |  1975.79  | 1975  |  10   |
| 329.25  |  1975.88  | 1975  |  11   |
| 330.97  |  1975.96  | 1975  |  12   |

✅ 識別此數據集中隨時間變化的變量

## 時間序列數據的特徵考量

當查看時間序列數據時，你可能會注意到它具有[某些特徵](https://online.stat.psu.edu/stat510/lesson/1/1.1)，需要考慮並減少這些特徵以更好地理解其模式。如果你將時間序列數據視為可能提供“信號”的數據，這些特徵可以被視為“噪聲”。你通常需要通過一些統計技術來減少這些“噪聲”。

以下是一些你應該了解的概念，以便能夠處理時間序列數據：

🎓 **趨勢**

趨勢被定義為隨時間可測量的增長或減少。[了解更多](https://machinelearningmastery.com/time-series-trends-in-python)。在時間序列的上下文中，這是關於如何使用以及在必要時移除趨勢。

🎓 **[季節性](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

季節性被定義為週期性波動，例如假期高峰可能影響銷售。[了解更多](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm)關於不同類型的圖如何顯示數據中的季節性。

🎓 **異常值**

異常值遠離標準數據變異範圍。

🎓 **長期週期**

獨立於季節性，數據可能顯示長期週期，例如持續超過一年的經濟衰退。

🎓 **恆定變異**

隨時間推移，一些數據顯示恆定的波動，例如每日和夜間的能源使用量。

🎓 **突變**

數據可能顯示需要進一步分析的突變。例如，COVID 導致的企業突然關閉引起了數據的變化。

✅ 這裡有一個[時間序列圖示例](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python)，顯示了幾年內每日遊戲內貨幣的消費。你能在這些數據中識別出上述特徵嗎？

![遊戲內貨幣消費](../../../../translated_images/currency.e7429812bfc8c6087b2d4c410faaa4aaa11b2fcaabf6f09549b8249c9fbdb641.mo.png)

## 練習 - 開始使用電力使用數據

讓我們開始創建一個時間序列模型，根據過去的使用情況來預測未來的電力使用。

> 本例中的數據來自 GEFCom2014 預測競賽。它包含了 2012 年到 2014 年之間 3 年的每小時電力負載和溫度值。
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli 和 Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. 在本課程的 `working` 資料夾中，打開 _notebook.ipynb_ 文件。首先添加幫助你加載和可視化數據的庫：

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    注意，你正在使用來自 `common` 資料夾的文件，這些文件設置了你的環境並處理數據下載。

2. 接下來，通過調用 `load_data()` 和 `head()` 查看數據作為數據框：

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    你可以看到有兩列分別表示日期和負載：

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. 現在，通過調用 `plot()` 繪製數據：

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![能源圖](../../../../translated_images/energy-plot.5fdac3f397a910bc6070602e9e45bea8860d4c239354813fa8fc3c9d556f5bad.mo.png)

4. 現在，繪製 2014 年 7 月第一週的數據，通過將其作為輸入提供給 `energy`，格式為 `[from date]: [to date]`：

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![七月](../../../../translated_images/july-2014.9e1f7c318ec6d5b30b0d7e1e20be3643501f64a53f3d426d7c7d7b62addb335e.mo.png)

    一個漂亮的圖表！看看這些圖表，看看你是否能確定上述列出的任何特徵。通過可視化數據，我們可以推測出什麼？

在下一課中，你將創建一個 ARIMA 模型來進行一些預測。

---

## 🚀挑戰

列出你能想到的所有可能受益於時間序列預測的行業和研究領域。你能想到這些技術在藝術中的應用嗎？在計量經濟學中？生態學？零售業？工業？金融？還有其他地方嗎？

## [課後測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/)

## 回顧與自學

雖然我們不會在這裡討論，但有時會使用神經網絡來增強經典的時間序列預測方法。閱讀更多相關內容[在這篇文章中](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## 作業

[可視化更多時間序列](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解讀概不負責。