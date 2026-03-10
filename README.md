# Quant Research Playground

This repository contains personal quantitative research experiments focused on **market microstructure signals, machine learning trading strategies, and systematic strategy prototyping**.

本 repo 用於整理個人量化研究實驗，主要探討市場微結構訊號、機器學習交易策略與短週期 alpha 訊號的可能性。

---

📄 Research Presentation  

[OFI Short Horizon Machine Learning Trading Strategy](./OFI_short_horizon_ml_strategy.pdf)

---


# Research Project  
研究專案

## Trade-based Order Flow Imbalance: A Short-Horizon Machine Learning Trading Strategy  
主動買賣力不平衡（OFI）之短週期機器學習交易策略研究

This study investigates whether **trade-based Order Flow Imbalance (OFI)** contains predictive information for short-horizon price movements in Taiwan index futures.

Using **TAIFEX Mini Taiwan Index Futures (MTX) transaction data (2017–2023)**, OFI-based microstructure features are constructed and used in an **XGBoost multi-class model** to predict the next 1-minute price direction.

Model predictions are transformed into a **ranking-based trading strategy**, followed by backtesting and out-of-sample validation.

本研究探討 **成交資料所計算的 Order Flow Imbalance（OFI）** 是否能預測期貨市場短期價格方向。

研究使用 **台灣期交所小型台指期貨（MTX）2017–2023 逐筆成交資料**，建構 OFI 微結構特徵，並透過 **XGBoost 多分類模型** 預測未來 1 分鐘價格方向，將預測訊號轉換為 **排序式交易策略** 並進行回測與樣本外驗證。




# Repository Structure
```
Quant-research-playground/
│
├── OFI_short_horizon_ml_strategy.pdf   # research presentation / 研究簡報
├── README.md
│
├── data/          # datasets used in research / 研究資料
├── src/           # research code (data processing, modeling, backtesting)
├── results/       # backtest results and figures / 回測結果與圖表
├── research_raw/  # raw research notes and materials / 研究過程筆記
```