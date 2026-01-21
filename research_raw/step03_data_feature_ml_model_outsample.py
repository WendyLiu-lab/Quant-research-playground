#%%
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb


#%% ===================================================================================================

# mm_pd.to_parquet("minute_model.parquet", index=False)

mm_pd = pd.read_parquet("minute_model.parquet") 
print(mm_pd.head())

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

train_end = pd.to_datetime("2023-01-01")

train_mask = mm_pd["交易日"] < train_end
test_mask  = mm_pd["交易日"] >= train_end


# ===== X, y =====
X = mm_pd[feature_cols]
y = mm_pd["y_updown"]

label_map = {-1:0, 0:1, 1:2}
y_m = y.map(label_map)

dtrain = xgb.DMatrix(X.loc[train_mask], label=y_m.loc[train_mask])
dtest  = xgb.DMatrix(X.loc[test_mask],  label=y_m.loc[test_mask])



#%%

# mm_pd["交易日"]=pd.to_datetime(mm_pd["交易日"])
# split_date=pd.to_datetime("2023-01-01")

# X=mm_pd[feature_cols]
# y=mm_pd["y_updown"] 
# # y = mm_pd["y_updown_5m"] #改5分鐘

# label_map={-1:0,0:1,1:2}
# y_m=y.map(label_map)

# train_mask=mm_pd["交易日"]<split_date
# test_mask=~train_mask

# dtrain=xgb.DMatrix(X.loc[train_mask],label=y_m.loc[train_mask])
# dtest=xgb.DMatrix(X.loc[test_mask],label=y_m.loc[test_mask])

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
    index=mm_pd.loc[test_mask].index
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
def backtest_oos_with_fixed_threshold(
    mm_pd: pd.DataFrame,
    proba_df: pd.DataFrame,
    th_long: float,
    th_short: float,
    HOLD: int,
    BASE_FEE: float,
    allow_flip: bool = True
):
    FEE = float(BASE_FEE)

    #============ 1) score ============
    score = proba_df["p_up"] - proba_df["p_down"]

    #============ 2) fixed signal (-1/0/+1) ============
    sig_raw = np.where(
        score > th_long,  1,
        np.where(score < th_short, -1, 0)
    )
    sig_raw = pd.Series(sig_raw, index=proba_df.index)

    #============ 3) align df ============
    df = mm_pd.loc[proba_df.index].copy()

    #============ 4) signal shift ============
    df["signal_shift"] = sig_raw.shift(1).fillna(0).astype(int)

    #============ 5) optional hold lock ============
    if not allow_flip:
        s = df["signal_shift"].values.copy()
        lock = 0
        for i in range(len(s)):
            if lock > 0:
                s[i] = 0
                lock -= 1
            elif s[i] != 0:
                lock = max(HOLD - 1, 0)
        df["signal_shift"] = s.astype(int)

    #============ 6) entry / exit ============
    df["entry_price"] = df["price_open"]
    df["exit_price"] = (
        df.groupby(["商品代號", "到期月份(週別)"])["price_close"]
          .shift(-(HOLD - 1))
    )

    #============ 7) pnl ============
    df["pnl_point"] = df["signal_shift"] * (
        df["exit_price"] - df["entry_price"]
    )

    df["turnover"] = np.where(df["signal_shift"] != 0, 2, 0)

    df["price"] = df["entry_price"] + df["exit_price"]
    df["fee_cost_pt"] = df["turnover"] * FEE
    df["tax_cost_pt"] = np.where(
        df["signal_shift"] != 0,
        df["price"] * 2 / 100000,
        0.0
    )

    df["net_pnl_pt"] = df["pnl_point"] - (df["fee_cost_pt"] + df["tax_cost_pt"])

    df_bt = df.dropna(subset=["exit_price"]).copy()

    #============ 8) metrics ============
    ret = df_bt["net_pnl_pt"]
    daily_pnl = ret.groupby(df_bt["交易日"]).sum()

    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() != 0 else 0.0
    maxdd = (ret.cumsum().cummax() - ret.cumsum()).max()

    summary = {
        "Sharpe_daily_point": sharpe,
        "TotalPnL_point": ret.sum(),
        "AvgDailyPnL_point": daily_pnl.mean(),
        "MaxDD_point": maxdd,
        "TotalTurnover": df_bt["turnover"].sum(),
        "LongTrades": (df_bt["signal_shift"] == 1).sum(),
        "ShortTrades": (df_bt["signal_shift"] == -1).sum(),
    }

    return df_bt, summary


#%%
params_df = pd.read_csv("best_params_long_short.csv").iloc[0]

BEST_MODE = params_df["mode"]
BEST_K = params_df["K"]
BEST_HOLD = int(params_df["HOLD"])
BEST_TH_LONG = params_df["th_long"]
BEST_TH_SHORT = params_df["th_short"]

print("📌 載入最佳交易參數（from 2022）：")
print(params_df)


#%%
BASE_FEE = 0.4
df_bt_2023, summary_2023 = backtest_oos_with_fixed_threshold(
    mm_pd=mm_pd,
    proba_df=proba_df,   # ← 這是 2023 的 model 預測
    th_long=BEST_TH_LONG,
    th_short=BEST_TH_SHORT,
    HOLD=BEST_HOLD,
    BASE_FEE=BASE_FEE,
    allow_flip=True
)

print("\n===== 📊 2023 OOS Backtest Result =====")
for k, v in summary_2023.items():
    print(f"{k}: {v}")




























#%%

def backtest_full_points(
    mm_pd: pd.DataFrame,
    proba_df: pd.DataFrame,
    # K: float,
    HOLD: int,
    BASE_FEE: float,
    THRESHOLD,
):
    """
    固定 HOLD、Top-K 策略的完整回測（點數）
    回傳：
        sharpe_daily
        df_bt（含 pnl / cost）
        summary（dict）
    """

    FEE = BASE_FEE

    #============ 1) score ============
    score = proba_df["p_up"] - proba_df["p_down"]

    #============ 2) Top-K ============
    # th = score.quantile(1 - K)

    sig_raw = (score > THRESHOLD).astype(int)
    # sig_raw = (score > th).astype(int)

    #============ 3) 對齊測試資料 ============
    df = mm_pd.loc[proba_df.index].copy()

    #============ 4) 訊號後移 ============
    df["signal_shift"] = sig_raw.shift(1).fillna(0).astype(int)

    #============ 5) entry / exit 價格 ============
    df["entry_price"] = df["price_open"]

    df["exit_price"] = (
        df.groupby(["商品代號", "到期月份(週別)"])["price_close"]
          .shift(-(HOLD - 1))
    )

    #============ 6) 損益（點） ============
    df["pnl_point"] = df["signal_shift"] * (
        df["exit_price"] - df["entry_price"]
    )

    #============ 7) turnover ============
    df["turnover"] = np.where(
        df["signal_shift"] == 1,
        2,
        0
    )

    #============ 8) 成本 ============
    df["price"] = df["entry_price"] + df["exit_price"]

    df["fee_cost_pt"] = df["turnover"] * FEE

    df["tax_cost_pt"] = np.where(
        df["signal_shift"] == 1,
        df["price"] * 2 / 100000,
        0.0
    )

    df["cost_pt"] = df["fee_cost_pt"] + df["tax_cost_pt"]

    df["net_pnl_pt"] = df["pnl_point"] - df["cost_pt"]

    #============ 9) 清掉尾端 NaN ============
    df_bt = df.dropna(subset=["entry_price", "exit_price"]).copy()

    ret = df_bt["net_pnl_pt"]

    #============ 10) Daily Sharpe ============
    daily_pnl = ret.groupby(df_bt["交易日"]).sum()
    sharpe_daily = (
        daily_pnl.mean() / daily_pnl.std()
    ) * np.sqrt(252) if daily_pnl.std() != 0 else 0.0

    #============ 11) Equity / MaxDD ============
    eq = ret.cumsum()
    dd_point = eq.cummax() - eq
    maxdd_point = dd_point.max()

    #============ 12) Win Rate ============
    pnl_pos = ret[df_bt["signal_shift"] == 1]
    win_rate = (pnl_pos > 0).mean() if len(pnl_pos) > 0 else 0.0

    #============ 13) Turnover ============
    total_turnover = df_bt["turnover"].sum()
    avg_daily_turnover = (
        df_bt.groupby(df_bt["交易日"].dt.date)["turnover"].sum().mean()
    )

    #============ 14) Total PnL ============
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

    return sharpe_daily, df_bt, summary



#%%
BASE_FEE = 0.4 


sharpe_23, df_23, summary_23 = backtest_full_points(
    mm_pd,
    proba_df,    
    HOLD = BEST_HOLD,
    BASE_FEE = 0.4 ,
    THRESHOLD = th_best
)

print("\n====== 2023 OOS Result ======")
print(summary_23)

#%%
summary_23







#%%
BASE_FEE = 0.4   # 單邊費用 (點)

def backtest_full_points(mm_pd, proba_df, HOLD, FEE, THRESHOLD):

    score = proba_df["p_up"] - proba_df["p_down"]

    sig_raw = (score > THRESHOLD).astype(int)

    sig = (
        sig_raw.replace(0, np.nan)
        .ffill(limit=int(HOLD))
        .fillna(0)
        .astype(int)
    )

    df = mm_pd.loc[proba_df.index].copy()

    df["signal_shift"] = sig.shift(1).fillna(0).astype(int)

    df["pnl_point"] = df["signal_shift"] * (
        df["future_price"] - df["price_close"]
    )

    df["turnover"] = df["signal_shift"].diff().abs().fillna(0)

    df["price"] = df["price_close"]

    df["fee_cost_pt"] = df["turnover"] * FEE
    df["tax_cost_pt"] = df["turnover"] * (df["price"] * 2 / 100000)

    df["cost_pt"] = df["fee_cost_pt"] + df["tax_cost_pt"]

    df["net_pnl_pt"] = df["pnl_point"] - df["cost_pt"]

    ret = df["net_pnl_pt"]

    daily_pnl = ret.groupby(df["交易日"]).sum()
    sharpe_daily = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

    eq = ret.cumsum()
    dd_point = eq.cummax() - eq
    maxdd_point = dd_point.max()

    pnl_pos = ret[df["signal_shift"] == 1]
    win_rate = (pnl_pos > 0).mean()

    exposure = (df["signal_shift"] == 1).mean()

    total_turnover = df["turnover"].sum()
    avg_daily_turnover = df.groupby(df["交易日"].dt.date)["turnover"].sum().mean()

    total_pnl_pt = ret.sum()

    summary = {
        "TotalPnL_point": float(total_pnl_pt),
        "AvgDailyPnL_point": float(daily_pnl.mean()),
        "Sharpe_daily_point": float(sharpe_daily),
        "MaxDD_point": float(maxdd_point),
        "Exposure": float(exposure),
        "WinRate": float(win_rate),
        "TotalTurnover": float(total_turnover),
        "AvgDailyTurnover": float(avg_daily_turnover),
    }

    return sharpe_daily, df, summary





















































































































# %%
