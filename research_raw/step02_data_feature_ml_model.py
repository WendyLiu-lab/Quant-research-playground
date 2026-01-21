#%%
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb





#%%   1) 讀檔（Polars）

csv_path = "C:/Wendy/TXF/MTX_near_month_20171225_20231231.csv"
settle_path = "C:/Wendy/Project/TX_MTX_TMF_結算日_整理後.txt"

# === 讀原始成交資料 ===
df = (
    pl.read_csv(csv_path)
      .with_columns([
          pl.col("成交日期").str.to_datetime(strict=False),
          pl.col("成交時間").cast(pl.Utf8).str.zfill(6),   # 補到 HHMMSS
      ])
      .with_columns([
          pl.concat_str([
              pl.col("成交日期").dt.strftime("%Y-%m-%d"),
              pl.lit(" "),
              pl.col("成交時間")
          ]).str.to_datetime("%Y-%m-%d %H%M%S").alias("成交_dt")
      ])
      .with_columns([
          pl.col("到期月份(週別)").cast(pl.Utf8)
      ])
)

# === 讀結算日資料 ===
settle_pd = pd.read_csv(settle_path, sep=r"\s+", engine="python")
settle_df = pl.from_pandas(settle_pd)

# 清理欄位格式
settle_df = settle_df.with_columns([
    pl.col("契約月份").cast(pl.Utf8),
    pl.col("最後結算日").str.to_datetime()
])


#%%
earliest = df.select(pl.col("成交日期").min()).item()
latest   = df.select(pl.col("成交日期").max()).item()

print("最早成交日期：", earliest)
print("最晚成交日期：", latest)
#%%  2) JOIN + 建「交易日」

df = (
    df.join(
        settle_df.select(["契約月份", "最後結算日"]),
        left_on="到期月份(週別)",
        right_on="契約月份",
        how="left"
    )
)

# === 交易日：日盤要往後推一天 ===
df = (
    df.with_columns([
        pl.col("成交_dt").dt.date().alias("交易日"),
        pl.col("成交_dt").dt.hour().alias("hour")
    ])
      .with_columns([
          pl.when(pl.col("hour") >= 15)
            .then(pl.col("交易日") + pl.duration(days=1))
            .otherwise(pl.col("交易日"))
            .alias("交易日")
      ])
      .drop("hour")
)

# is_last_trade_day
df = df.with_columns([
    (pl.col("交易日") == pl.col("最後結算日").dt.date()).alias("is_last_trade_day")
])


#%%

#%%
def make_second_bars_full_session(group: pd.DataFrame) -> pd.DataFrame:
    """
    對單一 (交易日, 到期月份(週別)) 做：
    - 若有夜盤：前一日 15:00:00 ~ 當日 04:59:59
    - 若有日盤：當日 08:45:00 ~ 13:44:59（最後交易日到 13:29:59）
    以「一秒一筆」輸出，不管該秒有沒有成交，沒成交的秒價量為 NaN。
    """
    trade_date = group["交易日"].iloc[0].normalize()
    is_last_trade_day = bool(group["is_last_trade_day"].iloc[0])

    g = group.set_index("成交_dt").sort_index()

    parts = []

    # ===== 夜盤 session =====
    night_start = trade_date - pd.Timedelta(days=1) + pd.Timedelta(hours=15)  # 前一日 15:00:00
    night_end_excl = trade_date + pd.Timedelta(hours=5)                       # 當日 05:00:00 (排除端點)

    # 先抓出這段內真的有成交的資料，來判斷這天到底有沒有夜盤
    night_trades = g.loc[(g.index >= night_start) & (g.index < night_end_excl)]

    if not night_trades.empty:
        # 先用 resample 把同一秒多筆壓成一筆（價取 last，量取 sum）
        night_agg = (
            night_trades
            .resample("1S")
            .agg({
                "成交價格": "last",
                "成交數量(B+S)": "sum",
                "近月價格": "last",
                "遠月價格": "last",
            })
        )

        # 然後依「交易規則」建出完整秒級 index：15:00:00 ~ 04:59:59
        full_night_idx = pd.date_range(
            start=night_start,
            end=night_end_excl - pd.Timedelta(seconds=1),  # 05:00:00 前一秒 = 04:59:59
            freq="1S"
        )

        # 強制對齊到完整夜盤秒軸，沒成交的秒會是 NaN
        night_full = night_agg.reindex(full_night_idx)

        parts.append(night_full)

    # ===== 日盤 session =====
    day_start = trade_date + pd.Timedelta(hours=8, minutes=45)

    if is_last_trade_day:
        day_end_excl = trade_date + pd.Timedelta(hours=13, minutes=30)  # 不含 13:30:00 → 最後一秒 13:29:59
    else:
        day_end_excl = trade_date + pd.Timedelta(hours=13, minutes=45)  # 不含 13:45:00 → 最後一秒 13:44:59

    day_trades = g.loc[(g.index >= day_start) & (g.index < day_end_excl)]

    if not day_trades.empty:
        day_agg = (
            day_trades
            .resample("1S")
            .agg({
                "成交價格": "last",
                "成交數量(B+S)": "sum",
                "近月價格": "last",
                "遠月價格": "last",
            })
        )

        full_day_idx = pd.date_range(
            start=day_start,
            end=day_end_excl - pd.Timedelta(seconds=1),
            freq="1S"
        )

        day_full = day_agg.reindex(full_day_idx)

        parts.append(day_full)

    if not parts:
        return pd.DataFrame()

    # 串起夜盤 + 日盤，index 是完整的秒級時間軸
    per_sec = pd.concat(parts)
    per_sec.index.name = "成交_dt"

    # 把 key 資訊補回來
    per_sec["交易日"] = trade_date
    per_sec["成交日期"] = trade_date
    per_sec["到期月份(週別)"] = group["到期月份(週別)"].iloc[0]
    per_sec["商品代號"] = group["商品代號"].iloc[0]
    per_sec["近月yyyymm"] = group["近月yyyymm"].iloc[0]

    # 還原 index，並重建 HHMMSS 的成交時間欄位
    per_sec = per_sec.reset_index()
    per_sec["成交時間"] = per_sec["成交_dt"].dt.strftime("%H%M%S").astype(int)

    return per_sec



#%%

#%%
def build_minute_model_for_group(group_pl: pl.DataFrame) -> pl.DataFrame:
    if group_pl.height == 0:
        return pl.DataFrame()

    # ---------- 1) pandas 秒級補值 ----------
    sec_pd = make_second_bars_full_session(group_pl.to_pandas())
    if sec_pd is None or len(sec_pd)==0:
        return pl.DataFrame()

    sec = pl.from_pandas(sec_pd).sort(
        ["交易日","商品代號","到期月份(週別)","成交_dt"]
    )

    sec = sec.with_columns([
        pl.col("成交價格")
          .forward_fill()
          .over(["交易日","商品代號","到期月份(週別)"])
          .alias("成交價格_filled"),

        pl.col("成交數量(B+S)")
          .fill_null(0)
          .alias("成交數量_filled"),

        pl.col("成交_dt").dt.truncate("1m").alias("bar_minute")
    ])

    # ---------- 2) 1分鐘 bar ----------
    minute = (
        sec.group_by(["交易日","商品代號","到期月份(週別)","bar_minute"])
           .agg([
               pl.col("成交價格_filled").first().alias("price_open"),
               pl.col("成交價格_filled").last().alias("price_close"),
               pl.col("成交價格_filled").max().alias("price_high"),
               pl.col("成交價格_filled").min().alias("price_low"),
               pl.col("成交數量_filled").sum().alias("vol_1m"),
           ])
    )

    # ---------- 3) 基本報酬/波動 ----------
    minute = minute.with_columns([
        (pl.col("price_close") - pl.col("price_open")).alias("ret_1m"),
        (pl.col("price_high") - pl.col("price_low")).alias("range_1m"),
    ])

    # # ---------- 4) Label：下一分鐘回報 ----------
    minute = minute.sort([ "bar_minute"] )

    minute = minute.with_columns([
        pl.col("price_close")
          .shift(-1)
          .over(["商品代號","到期月份(週別)"])
          .alias("future_price")
    ])

    minute = minute.with_columns([
        (pl.col("future_price") - pl.col("price_close")).alias("future_ret")
    ])

    minute = minute.with_columns([
        pl.when(pl.col("future_ret")>0).then(1)
         .when(pl.col("future_ret")<0).then(-1)
         .otherwise(0)
         .alias("y_updown")
    ])

    minute = minute.drop_nulls(["future_price"])


    # ---------- 5) OFI ----------
    minute = minute.with_columns([
        (pl.col("ret_1m").sign() * pl.col("vol_1m")).alias("ofi_1m")
    ])

    # ---------- 6) Lag1~3 ----------
    minute = minute.sort(["商品代號","到期月份(週別)","bar_minute"])

    for col in ["ofi_1m","vol_1m","ret_1m","range_1m"]:
        for k in [1,2,3,4]:
            minute = minute.with_columns([
                pl.col(col)
                  .shift(k)
                  .over(["商品代號","到期月份(週別)"])
                  .alias(f"{col}_lag{k}")
            ])

    # ---------- 7) Rolling ----------
    minute = minute.with_columns([
        pl.col("ret_1m")
          .rolling_sum(3)
          .over(["商品代號","到期月份(週別)"])
          .alias("ret_3m"),

        pl.col("vol_1m")
          .rolling_mean(3)
          .over(["商品代號","到期月份(週別)"])
          .alias("vol_3m_avg"),

        (pl.col("ofi_1m")/(pl.col("vol_1m")+1e-9)).alias("ofi_ratio")
    ])

    # ---------- 8) 移除 NaN ----------
    minute = minute.drop_nulls()

    return minute



#%% ================== Build 全部資料 ==================
keys = df.select(["交易日","到期月份(週別)"]).unique()

minute_model_list = []
for tdate,contract in keys.iter_rows():
    g = df.filter((pl.col("交易日")==tdate)&(pl.col("到期月份(週別)")==contract))
    m = build_minute_model_for_group(g)
    if m.height>0:
        minute_model_list.append(m)

minute_model = pl.concat(minute_model_list,how="vertical")
mm_pd = minute_model.to_pandas()



#%% ===================================================================================================

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










# #%%
# def backtest_full_points(
#     mm_pd: pd.DataFrame,
#     proba_df: pd.DataFrame,
#     K: float,
#     HOLD: int,
#     BASE_FEE: float
# ):
#     """
#     固定 HOLD、Top-K 策略的完整回測（點數）
#     回傳：
#         sharpe_daily
#         df_bt（含 pnl / cost）
#         summary（dict）
#     """

#     FEE = BASE_FEE

#     #============ 1) score ============
#     score = proba_df["p_up"] - proba_df["p_down"]

#     #============ 2) Top-K ============
#     th = score.quantile(1 - K)
#     sig_raw = (score > th).astype(int)

#     #============ 3) 對齊測試資料 ============
#     df = mm_pd.loc[proba_df.index].copy()

#     #============ 4) 訊號後移 ============
#     df["signal_shift"] = sig_raw.shift(1).fillna(0).astype(int)

#     #============ 5) entry / exit 價格 ============
#     df["entry_price"] = df["price_open"]

#     df["exit_price"] = (
#         df.groupby(["商品代號", "到期月份(週別)"])["price_close"]
#           .shift(-(HOLD - 1))
#     )

#     #============ 6) 損益（點） ============
#     df["pnl_point"] = df["signal_shift"] * (
#         df["exit_price"] - df["entry_price"]
#     )

#     #============ 7) turnover ============
#     df["turnover"] = np.where(
#         df["signal_shift"] == 1,
#         2,
#         0
#     )

#     #============ 8) 成本 ============
#     df["price"] = df["entry_price"] + df["exit_price"]

#     df["fee_cost_pt"] = df["turnover"] * FEE

#     df["tax_cost_pt"] = np.where(
#         df["signal_shift"] == 1,
#         df["price"] * 2 / 100000,
#         0.0
#     )

#     df["cost_pt"] = df["fee_cost_pt"] + df["tax_cost_pt"]

#     df["net_pnl_pt"] = df["pnl_point"] - df["cost_pt"]

#     #============ 9) 清掉尾端 NaN ============
#     df_bt = df.dropna(subset=["entry_price", "exit_price"]).copy()

#     ret = df_bt["net_pnl_pt"]

#     #============ 10) Daily Sharpe ============
#     daily_pnl = ret.groupby(df_bt["交易日"]).sum()
#     sharpe_daily = (
#         daily_pnl.mean() / daily_pnl.std()
#     ) * np.sqrt(252) if daily_pnl.std() != 0 else 0.0

#     #============ 11) Equity / MaxDD ============
#     eq = ret.cumsum()
#     dd_point = eq.cummax() - eq
#     maxdd_point = dd_point.max()

#     #============ 12) Win Rate ============
#     pnl_pos = ret[df_bt["signal_shift"] == 1]
#     win_rate = (pnl_pos > 0).mean() if len(pnl_pos) > 0 else 0.0

#     #============ 13) Turnover ============
#     total_turnover = df_bt["turnover"].sum()
#     avg_daily_turnover = (
#         df_bt.groupby(df_bt["交易日"].dt.date)["turnover"].sum().mean()
#     )

#     #============ 14) Total PnL ============
#     total_pnl_pt = ret.sum()

#     summary = {
#         "TotalPnL_point": float(total_pnl_pt),
#         "AvgDailyPnL_point": float(daily_pnl.mean()),
#         "StdDailyPnL_point": float(daily_pnl.std()),
#         "Sharpe_daily_point": float(sharpe_daily),
#         "MaxDD_point": float(maxdd_point),
#         "WinRate": float(win_rate),
#         "TotalTurnover": float(total_turnover),
#         "AvgDailyTurnover": float(avg_daily_turnover),
#     }

#     return sharpe_daily, df_bt, summary


# #%%






# #%%

# score_test = proba_df["p_up"] - proba_df["p_down"]


# #%%
# BASE_FEE = 0.4 
# K_list = [0.01,0.02, 0.05, 0.10, 0.20]
# HOLD_list = [10, 20, 30, 45, 60, 90]

# results = []

# for K in K_list:
#     for HOLD in HOLD_list:
#         sharpe_daily, df_bt, summary = backtest_full_points(
#             mm_pd,
#             proba_df,
#             K,
#             HOLD,
#             BASE_FEE
#         )
        
#         results.append({
#             "K": K,
#             "HOLD": HOLD,
#             "TotalPnL_point": summary["TotalPnL_point"],
#             "AvgDailyPnL_point": summary["AvgDailyPnL_point"],
#             "Sharpe_daily_point": summary["Sharpe_daily_point"],
#             "MaxDD_point": summary["MaxDD_point"],
#             "WinRate": summary["WinRate"],
#             "TotalTurnover": summary["TotalTurnover"],
#             "AvgDailyTurnover": summary["AvgDailyTurnover"],
#         })

# results_df = (
#     pd.DataFrame(results)
#     .sort_values(["Sharpe_daily_point","TotalPnL_point"], ascending=[False, False])
#     .reset_index(drop=True)
# )

# print("\n====== Top 10（依日 Sharpe 排序）======")
# print(results_df.head(10))



# #%%
# best_row = results_df.iloc[0]

# BEST_K = best_row["K"]
# BEST_HOLD = int(best_row["HOLD"])

# print("\n最佳參數：")
# print(best_row)

# th_best = score_test.quantile(1 - BEST_K)

# print("\n最佳 K =", BEST_K)
# print("最佳門檻 th =", th_best)




# #%%
# import pandas as pd

# df_params = pd.DataFrame([{
#     "BEST_K": BEST_K,
#     "BEST_HOLD": BEST_HOLD,
#     "th_best": th_best
# }])

# df_params.to_csv("best_params_0109.csv", index=False)

# print("已儲存參數到 best_params.csv")









































































































