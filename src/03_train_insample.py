"""
Train an in-sample prediction model and search strategy parameters.

Train XGBoost on minute-level features and output probability forecasts.
# Corresponds to research_raw/step02_data_feature_ml_model.py
"""
#%%
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
#%%

mm_pd = pd.read_parquet("minute_model.parquet") 
print(mm_pd.head())
# mm_pd["bar_minute"].is_unique
#%%
mm_pd = mm_pd.sort_values(
    ["交易日","商品代號","到期月份(週別)","bar_minute"]
).reset_index(drop=True)

# ===== 新增比例報酬 =====
mm_pd["future_ret_pct"] = mm_pd["future_ret"] / mm_pd["price_close"]


#%%



#%% ================== Train XGBoost ==================

feature_cols = [
    "ofi_1m","vol_1m","ret_1m",
    "ofi_1m_lag1","ofi_1m_lag2","ofi_1m_lag3","ofi_1m_lag4",
    "vol_1m_lag1","vol_1m_lag2","vol_1m_lag3","vol_1m_lag4",
    "ret_1m_lag1","ret_1m_lag2","ret_1m_lag3","ret_1m_lag4",
    "range_1m","range_1m_lag1","range_1m_lag2","range_1m_lag3","range_1m_lag4",
    "ret_3m",
    "vol_3m_avg",
    "ofi_ratio"
]


mm_pd["交易日"] = pd.to_datetime(mm_pd["交易日"])

train_end = pd.to_datetime("2022-01-01")
test_end  = pd.to_datetime("2022-12-31")

# ===== Train / Test mask =====
train_mask = mm_pd["交易日"] < train_end
test_mask  = (mm_pd["交易日"] >= train_end) & (mm_pd["交易日"] <= test_end)

# ===== X, y =====
X = mm_pd[feature_cols]
y = mm_pd["y_updown"]

label_map={-1:0,0:1,1:2}
y_m = y.map(label_map)

# ===== XGBoost DMatrix =====
dtrain = xgb.DMatrix(X.loc[train_mask], label=y_m.loc[train_mask])
dtest  = xgb.DMatrix(X.loc[test_mask],  label=y_m.loc[test_mask])


#%%

params={
 "objective":"multi:softprob",
 "num_class":3,
 "eta":0.1,
 "max_depth":5,
 "subsample":0.8,
 "colsample_bytree":0.8,
 "eval_metric":"mlogloss"
}

model=xgb.train(params,dtrain,num_boost_round=200)

proba=model.predict(dtest)
y_pred=np.argmax(proba,axis=1)
y_true=y_m.loc[test_mask].values


#%% ================== 評估指標 ==================
from sklearn.metrics import classification_report
y_prob = proba

# 取最大機率類別
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(
    y_true,
    y_pred,
    target_names=["Down","Flat","Up"]
))

proba_df = pd.DataFrame(
    proba,
    columns=["p_down","p_flat","p_up"],
    index=mm_pd.loc[test_mask].index   # 🔥 與測試集對齊
)


#%%
# 取出 importance (gain)
importance_dict = model.get_score(importance_type='gain')

# 轉成 dataframe
fi_df = (
    pd.DataFrame({
        "feature": list(importance_dict.keys()),
        "importance_gain": list(importance_dict.values())
    })
    .sort_values("importance_gain", ascending=False)
    .reset_index(drop=True)
)

print(fi_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(fi_df["feature"], fi_df["importance_gain"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()




#%%

def backtest_full_points_long_short(
    mm_pd: pd.DataFrame,
    proba_df: pd.DataFrame,
    K: float,
    HOLD: int,
    BASE_FEE: float,
    mode: str = "totalK",  # "symmetric"：多K+空K（總≈2K）；"totalK"：多K/2+空K/2（總≈K）
    allow_flip: bool = True,  # True：訊號每根都可翻多翻空；False：出場前不重複進場/翻向（較像持倉式）
    return_thresholds: bool = True
):
    """
    多空 Top-K（點數）回測：在你的現有框架上最小改動版
    - score = p_up - p_down
    - Long : score > th_long
    - Short: score < th_short
    - Signal 後移 1 根，用下一根 open 進場
    - 出場：固定 HOLD 分鐘後的 close（t+HOLD-1）
    - pnl_point = signal * (exit - entry)
    - cost：沿用你原本的點數成本邏輯（turnover=2/筆）
    
    參數：
    - mode="symmetric": 多空各佔 K，總交易密度約 2K
    - mode="totalK"   : 多空各佔 K/2，總交易密度約 K
    - allow_flip=True : 允許每根獨立交易（和你原本一樣，連續訊號會一直算交易）
    - allow_flip=False: 簡化版「不翻向」：有部位期間忽略新訊號（較接近持倉式，但仍用 entry/exit 方式計）
    
    回傳：
      sharpe_daily, df_bt, summary  (以及 thresholds 可選)
    """

    if proba_df is None or len(proba_df) == 0:
        empty = pd.DataFrame()
        summary = {
            "TotalPnL_point": 0.0,
            "AvgDailyPnL_point": 0.0,
            "StdDailyPnL_point": 0.0,
            "Sharpe_daily_point": 0.0,
            "MaxDD_point": 0.0,
            "WinRate": 0.0,
            "TotalTurnover": 0.0,
            "AvgDailyTurnover": 0.0,
        }
        if return_thresholds:
            summary.update({"th_long": np.nan, "th_short": np.nan})
        return 0.0, empty, summary

    FEE = float(BASE_FEE)

    #============ 1) score ============
    score = proba_df["p_up"] - proba_df["p_down"]

    #============ 2) thresholds ============
    if mode not in ("symmetric", "totalK"):
        raise ValueError("mode must be 'symmetric' or 'totalK'")

    k_side = K if mode == "symmetric" else (K / 2.0)

    # guard
    k_side = max(min(float(k_side), 0.499999), 0.0)

    th_long  = score.quantile(1 - k_side) if k_side > 0 else np.inf
    th_short = score.quantile(k_side)     if k_side > 0 else -np.inf

    #============ 3) raw signal (-1/0/+1) ============
    sig_raw = np.where(
        score > th_long,  1,
        np.where(score < th_short, -1, 0)
    )
    sig_raw = pd.Series(sig_raw, index=proba_df.index)

    #============ 4) align df ============
    df = mm_pd.loc[proba_df.index].copy()

    #============ 5) signal shift (avoid lookahead) ============
    df["signal_shift"] = sig_raw.shift(1).fillna(0).astype(int)

    #============ 6) (optional) block flips / re-entries during HOLD ============
    # 這段是「最小可用」的簡化版：一旦進場，就鎖住接下來 HOLD-1 根不再開新倉/翻向
    # 注意：它會降低交易次數與成本，也更接近持倉式。
    if not allow_flip:
        s = df["signal_shift"].values.copy()
        lock = 0
        for i in range(len(s)):
            if lock > 0:
                s[i] = 0
                lock -= 1
            else:
                if s[i] != 0:
                    lock = max(HOLD - 1, 0)
        df["signal_shift"] = s.astype(int)

    #============ 7) entry / exit ============
    df["entry_price"] = df["price_open"]

    df["exit_price"] = (
        df.groupby(["商品代號", "到期月份(週別)"])["price_close"]
          .shift(-(HOLD - 1))
    )

    #============ 8) pnl (points) ============
    df["pnl_point"] = df["signal_shift"] * (df["exit_price"] - df["entry_price"])

    #============ 9) turnover ============
    # 沿用你的簡化：每次有交易訊號就算一筆完整 round-trip（2 次）
    df["turnover"] = np.where(df["signal_shift"] != 0, 2, 0)

    #============ 10) cost ============
    df["price"] = df["entry_price"] + df["exit_price"]

    df["fee_cost_pt"] = df["turnover"] * FEE

    df["tax_cost_pt"] = np.where(
        df["signal_shift"] != 0,
        df["price"] * 2 / 100000,
        0.0
    )

    df["cost_pt"] = df["fee_cost_pt"] + df["tax_cost_pt"]
    df["net_pnl_pt"] = df["pnl_point"] - df["cost_pt"]

    #============ 11) drop tail NaN ============
    df_bt = df.dropna(subset=["entry_price", "exit_price"]).copy()

    ret = df_bt["net_pnl_pt"]

    #============ 12) Daily Sharpe ============
    daily_pnl = ret.groupby(df_bt["交易日"]).sum()
    sharpe_daily = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() != 0 else 0.0

    #============ 13) Equity / MaxDD ============
    eq = ret.cumsum()
    dd_point = eq.cummax() - eq
    maxdd_point = dd_point.max() if len(dd_point) > 0 else 0.0

    #============ 14) Win Rate (in position) ============
    pnl_pos = ret[df_bt["signal_shift"] != 0]
    win_rate = (pnl_pos > 0).mean() if len(pnl_pos) > 0 else 0.0

    #============ 15) Turnover stats ============
    total_turnover = df_bt["turnover"].sum()
    avg_daily_turnover = df_bt.groupby(df_bt["交易日"].dt.date)["turnover"].sum().mean()

    #============ 16) Total PnL ============
    total_pnl_pt = ret.sum()

    summary = {
        "TotalPnL_point": float(total_pnl_pt),
        "AvgDailyPnL_point": float(daily_pnl.mean()),
        "StdDailyPnL_point": float(daily_pnl.std()),
        "Sharpe_daily_point": float(sharpe_daily),
        "MaxDD_point": float(maxdd_point),
        "WinRate": float(win_rate),
        "TotalTurnover": float(total_turnover),
        "AvgDailyTurnover": float(avg_daily_turnover),
    }

    if return_thresholds:
        summary.update({
            "th_long": float(th_long) if np.isfinite(th_long) else th_long,
            "th_short": float(th_short) if np.isfinite(th_short) else th_short
        })

    return sharpe_daily, df_bt, summary




#%%


BASE_FEE = 0.4
K_list = [0.005,0.01,0.02,0.05,0.10,0.20]
HOLD_list = [10,20,30,45,60,90]

results = []
for K in K_list:
    for HOLD in HOLD_list:
        sharpe_daily, df_bt, summary = backtest_full_points_long_short(
            mm_pd=mm_pd,
            proba_df=proba_df,
            K=K,
            HOLD=HOLD,
            BASE_FEE=BASE_FEE,
            mode="totalK",     # ✅ 多空各佔K（總≈2K）totalK/symmetric
            allow_flip=True       # ✅ 和你原本一樣：每根都可觸發新交易
        )

        results.append({
            "K": K,
            "HOLD": HOLD,
            "Sharpe_daily_point": summary["Sharpe_daily_point"],
            "TotalPnL_point": summary["TotalPnL_point"],
            "MaxDD_point": summary["MaxDD_point"],
            "WinRate": summary["WinRate"],
            "TotalTurnover": summary["TotalTurnover"],
            "AvgDailyTurnover": summary["AvgDailyTurnover"],
            "th_long": summary.get("th_long", np.nan),
            "th_short": summary.get("th_short", np.nan),
        })

results_df = (
    pd.DataFrame(results)
      .sort_values(["TotalPnL_point"], ascending=[False])
      .reset_index(drop=True)
)

print(results_df.head(10))



#%%
best_row = results_df.iloc[0]
print("\n================ 最佳回測參數 ================\n")
print(f"Mode                  : totalK")
print(f"K                     : {best_row['K']}")
print(f"HOLD (minutes)        : {int(best_row['HOLD'])}")
print(f"Long Threshold        : {best_row['th_long']:.6f}")
print(f"Short Threshold       : {best_row['th_short']:.6f}")
print("\n--------------- Performance ------------------")
print(f"Sharpe (Daily)        : {best_row['Sharpe_daily_point']:.4f}")
print(f"Total PnL (point)     : {best_row['TotalPnL_point']:.2f}")
print(f"Avg PnL (point)       : {(best_row['TotalPnL_point']/best_row['TotalTurnover'])*2:.2f}")
print(f"Max Drawdown (point)  : {best_row['MaxDD_point']:.2f}")
print(f"Win Rate              : {best_row['WinRate']:.2%}")
print(f"Total Turnover        : {best_row['TotalTurnover']:.0f}")
print(f"Avg Daily Turnover    : {best_row['AvgDailyTurnover']:.2f}")
print("\n==============================================\n")

best_params_df = pd.DataFrame([{
    "mode": "totalK",
    "K": best_row["K"],
    "HOLD": int(best_row["HOLD"]),
    "th_long": best_row["th_long"],
    "th_short": best_row["th_short"],
    "Sharpe_daily_point": best_row["Sharpe_daily_point"],
    "TotalPnL_point": best_row["TotalPnL_point"],
    "AvgPnL_point": (best_row['TotalPnL_point']/best_row['TotalTurnover'])*2,
    "MaxDD_point": best_row["MaxDD_point"],
    "WinRate": best_row["WinRate"],
    "TotalTurnover": best_row["TotalTurnover"],
    "AvgDailyTurnover": best_row["AvgDailyTurnover"],
}])

best_params_df.to_csv("best_params_long_short.csv", index=False)
print("✅ 已儲存最佳參數到 best_params_long_short.csv")



#%%


_, df_bt_best, summary_best = backtest_full_points_long_short(
    mm_pd=mm_pd,
    proba_df=proba_df,
    K=best_row["K"],
    HOLD=int(best_row["HOLD"]),
    BASE_FEE=BASE_FEE,
    mode="totalK",
    allow_flip=True
)

#%%
num_long  = (df_bt_best["signal_shift"] == 1).sum()
num_short = (df_bt_best["signal_shift"] == -1).sum()
num_trade = num_long + num_short

print("\n=========== Trade Count ===========")
print(f"Long trades  : {num_long}")
print(f"Short trades : {num_short}")
print(f"Total trades : {num_trade}")
print("==================================")