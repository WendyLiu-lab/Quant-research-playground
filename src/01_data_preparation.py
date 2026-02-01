#%% -------------------- 匯入 --------------------
import pandas as pd
import re
from pathlib import Path

#%% -------------------- 參數 --------------------
folder = Path(r"C:/Wendy/TXF")              # Daily_yyyy_mm_dd.csv 的資料夾
pattern = "Daily_*.csv"                     # 檔名格式
# target_month = "2023_01"                    # ★ 只跑 2023/01
product_code = "MTX"                        # 只保留 MTX（全品種改 None）
roll_days_before_ltd = 1                    # LTD 前 N 天提前換月（你案例 = 1）
treat_ltd_as_current = True                 # LTD 當天仍視為當月

ltd_table_path = r"C:/Wendy/Project/TX_MTX_TMF_結算日_整理後.txt"

# 跑 2019/01/01 ~ 2023/12/31 的所有日檔
start_ymd = (2023, 1, 1)
end_ymd   = (2023, 5, 31)

# 輸出
out_path = folder / "MTX_near_month_20171225_20231231.csv"

#%% -------------------- 工具函式 --------------------
def read_daily_csv(path: Path) -> pd.DataFrame:
    """讀單日成交 CSV（big5），轉 '成交日期' 為 Timestamp。"""
    df = pd.read_csv(
        path,
        dtype={"成交時間": str, "到期月份(週別)": str, "商品代號": str},
        encoding="big5"
    )
    if pd.api.types.is_integer_dtype(df["成交日期"]) or pd.api.types.is_object_dtype(df["成交日期"]):
        df["成交日期"] = pd.to_datetime(df["成交日期"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["成交日期"] = pd.to_datetime(df["成交日期"], errors="coerce")
    return df

def load_ltd_lookup(path: str) -> dict[str, pd.Timestamp]:
    """
    讀『TX/MTX/TMF 結算日對照表』，回傳 { 'YYYYMM': Timestamp('YYYY-MM-DD') }，只留純月約。
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df = df.rename(columns=lambda c: c.strip())
    assert "契約月份" in df.columns and "最後結算日" in df.columns, "請確認欄名為『契約月份』『最後結算日』"

    df["契約月份"] = df["契約月份"].astype(str).str.strip()
    df = df[df["契約月份"].str.fullmatch(r"\d{6}")].copy()
    df["最後結算日"] = pd.to_datetime(df["最後結算日"].str.replace("/", "-"), errors="coerce")
    df = df[["契約月份", "最後結算日"]].dropna()
    return dict(zip(df["契約月份"], df["最後結算日"]))

def next_month_yyyymm(yyyymm: str) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    y2, m2 = (y + (m == 12), 1 if m == 12 else m + 1)
    return f"{y2:04d}{m2:02d}"

def pick_near_month_from_lookup(
    trade_date: pd.Timestamp,
    contracts: list[str],
    ltd_lookup: dict[str, pd.Timestamp],
    roll_days_before_ltd: int = 1,
    treat_ltd_as_current: bool = True,
) -> str | None:
    """用查表的實際 LTD 決定近月（只以純月約 YYYYMM）。"""
    monthlies = sorted({s.strip() for s in contracts if isinstance(s, str) and re.fullmatch(r"\d{6}", s.strip())})
    if not monthlies:
        return None

    td = pd.Timestamp(trade_date).normalize()
    base_yyyymm = td.strftime("%Y%m")
    candidates = [m for m in monthlies if m >= base_yyyymm]
    cur_yyyymm = candidates[0] if candidates else monthlies[0]

    ltd = ltd_lookup.get(cur_yyyymm)
    if pd.isna(ltd) or ltd is None:
        # 安全退路：若查不到，用第三個週三估
        first = pd.to_datetime(cur_yyyymm + "01", format="%Y%m%d")
        offset = (2 - first.weekday()) % 7  # 週三=2
        first_wed = first + pd.Timedelta(days=offset)
        ltd = (first_wed + pd.Timedelta(days=14)).normalize()

    roll_threshold = ltd - pd.Timedelta(days=roll_days_before_ltd)
    next_m = next_month_yyyymm(cur_yyyymm)

    if td < roll_threshold:
        return cur_yyyymm
    elif td < ltd:
        return next_m
    elif td == ltd:
        return cur_yyyymm if treat_ltd_as_current else next_m
    else:
        return next_m

def extract_near_month_one_file(
    path: Path,
    product_code: str | None,
    roll_days_before_ltd: int,
    ltd_lookup: dict[str, pd.Timestamp],
    treat_ltd_as_current: bool = True,
) -> pd.DataFrame:
    """單檔：用實際 LTD 查表決定近月 → 過濾近月（與商品），回傳逐筆。"""
    df = read_daily_csv(path)
    if df.empty:
        return df

    trade_date = df["成交日期"].iloc[0]
    if pd.isna(trade_date):
        print(f"[WARN] 成交日期解析失敗，略過：{path.name}")
        return pd.DataFrame()

    contracts_raw = df["到期月份(週別)"].dropna().astype(str).tolist()

    near_yyyymm = pick_near_month_from_lookup(
        trade_date=trade_date,
        contracts=contracts_raw,
        ltd_lookup=ltd_lookup,
        roll_days_before_ltd=roll_days_before_ltd,
        treat_ltd_as_current=treat_ltd_as_current,
    )
    if near_yyyymm is None:
        print(f"[WARN] 找不到月合約碼 YYYYMM，略過：{path.name}")
        return pd.DataFrame()

    # 只保留純月碼，並等於近月
    tmp = df["到期月份(週別)"].astype(str).str.strip()
    df_pure = df[tmp.str.fullmatch(r"\d{6}")].copy()
    out = df_pure[df_pure["到期月份(週別)"].astype(str).str.strip() == near_yyyymm].copy()

    if product_code:
        out = out[out["商品代號"].astype(str).str.strip() == product_code]

    out["近月yyyymm"] = near_yyyymm
    out["來源檔名"] = path.name
    return out

#%% -------------------- --------------------
# 依檔名 "Daily_YYYY_MM_DD.csv" s
def in_range_by_filename(fname: str) -> bool:
    m = re.search(r"Daily_(\d{4})_(\d{2})_(\d{2})\.csv$", fname)
    if not m:
        return False
    y, mo, d = map(int, m.groups())
    ymd = (y, mo, d)
    return (ymd >= start_ymd) and (ymd <= end_ymd)

all_files = sorted(f for f in folder.glob(pattern) if in_range_by_filename(f.name))

#%% -------------------- 執行 --------------------
# 1) 載入結算日查表
ltd_lookup = load_ltd_lookup(ltd_table_path)

# 2) 跑批
dfs = []
for i, fp in enumerate(all_files, 1):
    try:
        res = extract_near_month_one_file(
            path=fp,
            product_code=product_code,
            roll_days_before_ltd=roll_days_before_ltd,
            ltd_lookup=ltd_lookup,
            treat_ltd_as_current=treat_ltd_as_current,
        )
        if not res.empty:
            dfs.append(res)
        print(f"[{i}/{len(all_files)}] {fp.name} → rows: {len(res)}")
    except Exception as e:
        print(f"[ERROR] {fp.name}: {e}")

combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
if not combined.empty:
    combined = combined.sort_values(["成交日期", "成交時間"], kind="stable").reset_index(drop=True)
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成！輸出：{out_path}  總筆數：{len(combined)}")
else:
    print("\n⚠️ 沒有任何輸出資料，請檢查來源檔或過濾條件。")



