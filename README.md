# Quant Research Playground

This repository contains personal quantitative research focused on **market microstructure signals, machine learning-based trading strategies, and short-horizon alpha exploration**.

---

## 📄 Research Presentation  

[OFI Short Horizon Machine Learning Trading Strategy](./OFI_short_horizon_ml_strategy.pdf)  
*(Full presentation is written in Chinese; key ideas are summarized below in English.)*

---

## Trade-based Order Flow Imbalance (OFI):  
### A Short-Horizon Machine Learning Trading Strategy

### Overview

This project investigates whether **trade-based Order Flow Imbalance (OFI)** contains predictive power for short-horizon price movements in Taiwan index futures.

Using **TAIFEX Mini Taiwan Index Futures (MTX) transaction data (2017–2023)**, I construct microstructure features and apply an **XGBoost model** to predict next 1-minute price direction.

Model outputs are transformed into a **ranking-based trading strategy**, followed by backtesting and out-of-sample validation.

---

### Key Components

- OFI-based microstructure feature construction  
- XGBoost model for short-horizon prediction  
- Ranking-based signal selection (Top-K)  
- Backtesting framework with transaction cost  
- Time-based split for out-of-sample evaluation  

---

### Key Insights

- Alpha signals are concentrated at very short horizons and decay rapidly  
- Strategy performance is highly sensitive to ranking thresholds (Top-K selection)  
- Transaction cost is an important consideration in real-world strategy performance  
- Out-of-sample results suggest alpha exists but lacks stability  

---

## 中文說明（簡要）

本研究探討 **成交資料所計算的 Order Flow Imbalance（OFI）** 是否能預測期貨市場短期價格方向。

研究使用 **台灣期交所小型台指期貨（MTX）2017–2023 逐筆成交資料**，建構 OFI 微結構特徵，並透過 **XGBoost 模型** 預測未來 1 分鐘價格方向，將預測訊號轉換為 **排序式交易策略** 並進行回測與樣本外驗證。

---

## Repository Structure

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