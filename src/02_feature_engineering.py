"""
Construct minute-level features from tick data.

Build 1-minute bars and derive OFI, lagged, and rolling features
for short-horizon prediction.
# Corresponds to research_raw/step02_data_feature_ml_model.py
"""

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




