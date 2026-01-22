#%%
import pandas as pd
import re
from pathlib import Path
#%%

# 讀取一份 txt 檔案
def read_txt_file(txt_path):
    """
    讀取指定的 txt 檔案，回傳字串內容。
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

txt_path = r"C:/Wendy/Project/TX_MTX_TMF_結算日.txt"
content = read_txt_file(txt_path)

#%%
# 嘗試自動判斷分隔符號並整理成 DataFrame
import io

# 假設每行代表一筆資料，且以空白、tab或逗號等分隔，這裡自動推測最可能的分隔符
# 嘗試用常見分隔符號分割
lines = content.strip().splitlines()
# 嘗試尋找首行最常見的分隔
first_line = lines[0]
possible_delims = [",", "\t", "|", ";", " "]
delim_counts = {d: first_line.count(d) for d in possible_delims}
guessed_delim = max(delim_counts, key=delim_counts.get) if max(delim_counts.values()) > 0 else None

if guessed_delim is not None:
    df = pd.read_csv(io.StringIO(content), sep=guessed_delim, engine="python")
else:
    # 退而求其次，用預設 split 處理
    split_lines = [re.split(r"\s+", l) for l in lines]
    df = pd.DataFrame(split_lines[1:], columns=split_lines[0] if len(split_lines)>1 else None)

df.head()
# %%
# 將 df 輸出成 txt 檔
output_txt_path = r"C:/Wendy/Project/TX_MTX_TMF_結算日_整理後.txt"
df.to_csv(output_txt_path, sep='\t', index=False, encoding='utf-8')
print(f"✅ 已匯出為 txt：{output_txt_path}")

# %%
