#!/usr/bin/env python3
"""SOXL / GOOGL / SNDK 散戶情緒噪音：每日自動更新（由 Colab notebook 轉成，供 GitHub Actions 執行）。

流程：逐檔增量下載 Yahoo 資料 → 存 data/{代號}_history.csv → 分析 → 圖存 output/*_latest.png
      → 文字報告存 output/report_latest.txt → 每日摘要追加到 data/summary_log.csv
"""
import os, sys, io
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")                                   # 無螢幕環境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yfinance as yf

# 路徑：預設在腳本所在資料夾；可用環境變數 RN_BASE_DIR 覆寫（測試用）
BASE_DIR  = os.environ.get("RN_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "output")
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"   # workflow 會用 apt 安裝
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 使用者參數 ============
TICKERS         = ["SOXL", "GOOGL", "SNDK"]   # 要同時產生報告的標的，可自由增減
SCALES          = {"SOXL": 1.0}       # 各標的「分級尺度」；未列出的標的依自身波動自動估計（見 scale_for）
REF_HV          = 100                 # 基準門檻是依 SOXL（HV 約 100%）校準：自動尺度 = 該標的 HV 中位數 ÷ REF_HV
HISTORY_START   = "2023-01-01"        # 首次下載的起點（越早，HV 暖機越充分、之後越不用重抓）
OVERLAP_DAYS    = 10                  # 每次更新往回重抓幾個「日曆日」×2，用來修正舊資料並偵測拆股/除息
FORCE_FULL      = False               # True = 忽略快取、整份重新下載
LOOKBACK_DAYS   = 250                 # 分析/繪圖使用最近幾個交易日
RECENT_N        = 20                  # 「最近期間」天數
HV_WINDOW       = 20                  # 歷史波動率視窗
SAVE_FIGS       = True                # 圖檔是否存到 Drive outputs/
TODAY_COLOR     = "dodgerblue"        # 「今日」標示色（熱力圖外框、散佈圖星形標記）
MARKET_TZ       = "America/New_York"  # 所有時間顯示與「今日」判斷用的時區：美股收盤時區（自動處理 EST/EDT）

# 基準分級門檻（以 SOXL 為準；其他標的會乘上各自的尺度）
NOISE_CUTS      = [-20, 0, 30]            # 噪音分級切點
NOISE_LABELS    = ["機構控盤(極度冷清)", "情緒平淡(理性)", "散戶活躍(情緒升溫)", "史詩級散戶噪音(瘋狂踩踏)"]
RET_CUTS        = [-12, -4, 4, 12]        # 當日報酬分級切點（%）
NOISE_ACTIVE    = 30                      # 活躍臨界線
NOISE_EPIC      = 100                     # 「史詩級噪音」門檻

# 前瞻報酬研究（事件後 N 日的表現）
FWD_HORIZONS    = [1, 3, 5]               # 看事件後 1、3、5 個交易日
FWD_LOOKBACK_DAYS = None                  # 前瞻研究用最近幾個交易日；None = 全部歷史（樣本比 250 日視窗多）
FWD_NN          = 15                      # 「最相似 N 天」：以噪音、報酬標準化後距離今日最近的 N 天
FWD_BOOT        = 2000                    # 信賴區間的重抽樣次數
# ====================================


# --- 字型 ---
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

COLS = ["Open", "High", "Low", "Close", "Volume"]

# --- 字型 ---
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ---------- 資料層 ----------
def csv_path(ticker):
    """每檔標的各自一份快取檔。"""
    return os.path.join(DATA_DIR, f"{ticker}_history.csv")


def load_cache(path):
    """讀取快取；相容舊版 yfinance 多層標頭 CSV（會自動濾掉 'Ticker'/'Date' 說明列）。"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df[idx.notna()].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[COLS].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    return df.sort_index()


def fetch(ticker, start):
    """從 Yahoo 下載 start 之後的日線（未調整，欄位攤平），失敗重試 3 次。"""
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    last_err = None
    for _ in range(3):
        try:
            raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
            if raw is not None and len(raw):
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw[COLS].dropna(subset=["Close"])
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                raw.index.name = "Date"
                return raw
        except Exception as e:          # 網路/限流
            last_err = e
    if last_err:
        print("⚠️ 下載失敗：", last_err)
    return pd.DataFrame(columns=COLS)


def market_now():
    """目前的美股時區（美東）時間，含 EST/EDT 夏令時間。"""
    from zoneinfo import ZoneInfo
    return pd.Timestamp.now(tz=ZoneInfo(MARKET_TZ))


def fmt_market_time(ts=None):
    """格式化為『2026-10-26 17:45 EDT（美東）』；不傳參數 = 現在。"""
    ts = market_now() if ts is None else ts
    return ts.strftime("%Y-%m-%d %H:%M %Z") + "（美東）"


def day_tag(last_date, now=None):
    """最新資料就是美股『今天』→「今日」；否則（週末/假日/尚未開盤）→「最新交易日」。"""
    now = market_now() if now is None else now
    return "今日" if pd.Timestamp(last_date).date() == now.date() else "最新交易日"


def drop_incomplete_today(df):
    """美股盤中（美東時間 16:05 前）下載到的今日 K 棒尚未收盤，先丟掉，避免污染統計。"""
    try:
        now = market_now()
        if len(df) and df.index[-1].date() == now.date() and now.time() < dt.time(16, 5):
            print(f"ℹ️ 美股尚未收盤，略過未完成的 {df.index[-1].date()} K 棒")
            return df.iloc[:-1]
    except Exception:
        pass
    return df


def save_cache(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp)
    os.replace(tmp, path)               # 原子寫入，不會寫到一半壞檔


def update_cache(ticker, path, start, overlap_days, force=False):
    """增量更新：只抓『快取最後一天往前 overlap 區間』之後的資料，並自動偵測調整（拆股/除息）。"""
    cached = None if force else load_cache(path)
    if cached is None or cached.empty:
        print(f"📥 無快取（或強制重載），完整下載 {ticker}（{start} 起）...")
        merged = drop_incomplete_today(fetch(ticker, start))
    else:
        last = cached.index.max()
        new = drop_incomplete_today(fetch(ticker, last - pd.Timedelta(days=overlap_days * 2)))
        if new.empty:
            print("ℹ️ 沒有取得新資料，沿用本地快取")
            return cached
        # 檢查重疊區間的舊資料是否被 Yahoo 調整過 → 若是，快取已失真，整份重抓
        common = cached.index.intersection(new.index)
        common = common[common < last]
        if len(common) and (new.loc[common, "Close"] / cached.loc[common, "Close"] - 1).abs().max() > 0.01:
            print("⚠️ 偵測到歷史價格被調整（可能拆股/除息），整份重新下載")
            merged = drop_incomplete_today(fetch(ticker, start))
        else:
            added = (new.index > last).sum()
            merged = pd.concat([cached.drop(new.index, errors="ignore"), new]).sort_index()
            print(f"✅ 增量更新：新增 {added} 個交易日，重疊區間 {len(common)} 日已用新值覆蓋")
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if merged.empty:
        raise RuntimeError("沒有任何資料可用，請檢查網路或 ticker")
    save_cache(merged, path)
    return merged


def update_all(tickers, force=False):
    """逐檔更新；某一檔失敗（代號錯誤/抓不到）不會影響其他檔。回傳 ({代號: 資料}, {代號: 錯誤})。"""
    raws, errors = {}, {}
    for t in tickers:
        print(f"\n━━━ {t} ━━━")
        try:
            raws[t] = update_cache(t, csv_path(t), HISTORY_START, OVERLAP_DAYS, force=force)
        except Exception as e:
            errors[t] = e
            print(f"❌ {t} 更新失敗：{e}")
    return raws, errors


# ---------- 特徵層 ----------
def build_features(raw):
    """在『全部歷史』上計算特徵（HV 暖機才充分），之後再截取最近 LOOKBACK_DAYS。"""
    d = raw.copy()
    d["Log_Return"]          = np.log(d["Close"] / d["Close"].shift(1))
    d["HV_20d"]              = d["Log_Return"].rolling(HV_WINDOW).std() * np.sqrt(252) * 100
    d["Intraday_Volatility"] = (d["High"] - d["Low"]) / d["Close"] * np.sqrt(252) * 100
    d["Retail_Noise"]        = d["Intraday_Volatility"] - d["HV_20d"]
    d["Daily_Return"]        = d["Log_Return"] * 100
    d["Absolute_Return"]     = d["Daily_Return"].abs()
    return d.dropna()


# ---------- 分級層：不同波動度的標的用不同尺度 ----------
def scale_for(ticker, hv_median):
    """分級尺度：SCALES 有指定就用；否則 = HV 中位數 ÷ REF_HV，四捨五入到 0.05，限制在 0.1~1.5。"""
    if ticker in SCALES:
        return float(SCALES[ticker])
    return round(float(np.clip(round(hv_median / REF_HV / 0.05) * 0.05, 0.1, 1.5)), 2)


def make_levels(scale=1.0):
    """依尺度產生分級切點與標籤（scale=1 時與 SOXL 原本的分級完全相同）。"""
    f = lambda v: round(v * scale, 2)
    n = [f(v) for v in NOISE_CUTS]
    r = [f(v) for v in RET_CUTS]
    return SimpleNamespace(
        noise_bins=[-np.inf] + n + [np.inf],
        noise_labels=list(NOISE_LABELS),
        ret_bins=[-np.inf] + r + [np.inf],
        ret_labels=[f"暴跌 ({r[0]:g}%以下)", f"大跌 ({r[1]:g}%~{r[0]:g}%)", f"微幅震盪 ({r[1]:g}%~+{r[2]:g}%)",
                    f"大漲 (+{r[2]:g}%~+{r[3]:g}%)", f"暴漲 (+{r[3]:g}%以上)"],
        active=f(NOISE_ACTIVE), epic=f(NOISE_EPIC))


def noise_level(noise, lv):
    return ("史詩級噪音" if noise > lv.epic else
            "散戶活躍" if noise > lv.active else
            "平淡" if noise > 0 else "冷清")


def corr_table(d):
    """前 3 列是『當日同步』關係（暴跌當天盤中波動必然放大）；後面『未來 N 日報酬』才是預測力。"""
    t = {
        "噪音 × 每日報酬率": d.Retail_Noise.corr(d.Daily_Return),
        "噪音 × 報酬絕對值(波幅)": d.Retail_Noise.corr(d.Absolute_Return),
        "噪音 × HV_20d": d.Retail_Noise.corr(d.HV_20d),
    }
    for k in FWD_HORIZONS:
        if f"Fwd_{k}d" in d:
            t[f"噪音 × 未來{k}日報酬"] = d.Retail_Noise.corr(d[f"Fwd_{k}d"])
    return t


# ---------- 前瞻報酬層：這類日子之後 1/3/5 日表現如何？----------
BASE_GROUP = "全部交易日（基準）"


def add_forward_returns(df, horizons):
    """加上 Fwd_{k}d = 之後第 k 個交易日相對今日收盤的對數報酬(%)；最後 k 天沒有未來，留 NaN（不可當樣本）。"""
    d = df.copy()
    for k in horizons:
        d[f"Fwd_{k}d"] = np.log(d["Close"].shift(-k) / d["Close"]) * 100
    return d


def episode_ids(pos, gap):
    """事件的列位置（已排序）→ 事件編號。相隔 ≤ gap 個交易日的事件前瞻視窗重疊，算同一波（不是獨立樣本）。"""
    pos = np.asarray(pos)
    if len(pos) == 0:
        return np.array([], dtype=int)
    return np.concatenate([[0], np.cumsum(np.diff(pos) > gap)])


def fwd_stats(vals, ep, base_mean, n_boot=2000, seed=0):
    """一組事件的前瞻報酬統計；信賴區間以『獨立事件』為單位重抽樣（不是以天），事件 < 3 個不給區間。"""
    vals = np.asarray(vals, float)
    nan = np.nan
    out = dict(n=len(vals), episodes=nan if ep is None else len(np.unique(ep)),
               mean=nan, median=nan, win=nan, lo=nan, hi=nan, excess=nan)
    if len(vals) == 0:
        return out
    out.update(mean=vals.mean(), median=float(np.median(vals)), win=(vals > 0).mean() * 100,
               excess=vals.mean() - base_mean)
    if ep is not None and out["episodes"] >= 3:
        ids = np.unique(ep)
        sums = np.array([vals[ep == e].sum() for e in ids])
        cnts = np.array([(ep == e).sum() for e in ids])
        pick = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
        means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
        out["lo"], out["hi"] = np.percentile(means, [2.5, 97.5])
    return out


def evidence(st):
    """證據強度：事件太少 → 樣本不足；基準值落在區間外 → 異於基準；否則 → 與基準無顯著差異。"""
    if np.isnan(st["episodes"]):
        return "基準"
    if st["episodes"] < 5:
        return "樣本不足"
    base = st["mean"] - st["excess"]
    return "異於基準" if (base < st["lo"] or base > st["hi"]) else "與基準無顯著差異"


def forward_groups(c):
    """各組事件的布林遮罩（對齊 c.df_fwd）。今日是最後一列，沒有未來資料，統計時會自動排除。"""
    f, lv = c.df_fwd, c.lv
    last = f.iloc[-1]
    kmax = max(FWD_HORIZONS)
    ok = f[f"Fwd_{kmax}d"].notna().values                       # 有完整未來視窗的列
    z = lambda s: (s - last[s.name]) / s.std()                  # 以今日為原點、標準化
    dist = np.sqrt(z(f.Retail_Noise) ** 2 + z(f.Daily_Return) ** 2).values
    cand = np.flatnonzero(ok)
    nn = np.zeros(len(f), bool)
    nn[cand[np.argsort(dist[cand])[:FWD_NN]]] = True
    cut = lv.ret_bins[2]
    return {
        BASE_GROUP: np.ones(len(f), bool),
        f"噪音活躍（>{lv.active:g}）": (f.Retail_Noise > lv.active).values,
        f"大跌日（報酬≤{cut:g}%）": (f.Daily_Return <= cut).values,
        "今日同格": ((f["噪音分級"] == last["噪音分級"]) & (f["股價走勢"] == last["股價走勢"])).values,
        f"最相似{int(nn.sum())}天": nn,
    }


def forward_table(c):
    """每組 × 每個天期：樣本天數、獨立事件數、平均/中位數/勝率、95% 信賴區間、較基準超額、證據強度。"""
    f, gap = c.df_fwd, max(FWD_HORIZONS)
    rows = []
    for name, mask in forward_groups(c).items():
        pos = np.flatnonzero(mask)
        ep = None if name == BASE_GROUP else episode_ids(pos, gap)
        for k in FWD_HORIZONS:
            col = f[f"Fwd_{k}d"].values
            ok = ~np.isnan(col[pos])
            st = fwd_stats(col[pos][ok], None if ep is None else ep[ok], np.nanmean(col), FWD_BOOT, seed=k)
            rows.append({"組別": name, "天期": k, "樣本天數": st["n"], "獨立事件數": st["episodes"],
                         "平均%": st["mean"], "中位數%": st["median"], "勝率%": st["win"],
                         "CI下限%": st["lo"], "CI上限%": st["hi"], "較基準%": st["excess"], "證據": evidence(st)})
    t = pd.DataFrame(rows).set_index(["組別", "天期"]).round(2)
    num = t.select_dtypes("number").columns
    t[num] = t[num] + 0.0                                    # 避免顯示 -0.0
    return t


def extremity(c):
    """今日相對歷史有多極端；超出歷史範圍時，回歸線/熱力圖都是外推或單一格子的少數樣本，不可靠。"""
    v = c.df_view
    last, hist = v.iloc[-1], v.iloc[:-1]
    return dict(
        noise=last.Retail_Noise, ret=last.Daily_Return, n_hist=len(hist),
        noise_pct=(hist.Retail_Noise < last.Retail_Noise).mean() * 100,
        ret_pct=(hist.Daily_Return < last.Daily_Return).mean() * 100,
        noise_max=hist.Retail_Noise.max(), ret_min=hist.Daily_Return.min(), ret_max=hist.Daily_Return.max(),
        beyond_noise=bool(last.Retail_Noise > hist.Retail_Noise.max()),
        beyond_ret=bool(last.Daily_Return < hist.Daily_Return.min() or last.Daily_Return > hist.Daily_Return.max()))


def cell_stats(c):
    """今日所在熱力圖格子：(歷史天數, 獨立事件數)。連續出現在同一波行情的天數只算一個事件。"""
    d, last = c.d, c.d.iloc[-1]
    mask = ((d["噪音分級"] == last["噪音分級"]) & (d["股價走勢"] == last["股價走勢"])).values
    return int(mask.sum()), int(len(np.unique(episode_ids(np.flatnonzero(mask), max(FWD_HORIZONS)))))


# ---------- 分析層：一檔標的 → 一份 ctx（報告所需的一切）----------
def analyze(ticker, raw):
    """純計算（不連網、不畫圖）：特徵、分級、熱力圖計數、相關係數。"""
    df_all = add_forward_returns(build_features(raw), FWD_HORIZONS)   # 全歷史計算（含前瞻報酬欄）
    df_view = df_all.tail(LOOKBACK_DAYS).copy()       # 分析視窗
    recent = df_view.tail(RECENT_N)
    scale = scale_for(ticker, df_view.HV_20d.median())
    lv = make_levels(scale)
    d = df_view.copy()
    d["噪音分級"] = pd.cut(d["Retail_Noise"], bins=lv.noise_bins, labels=lv.noise_labels)
    d["股價走勢"] = pd.cut(d["Daily_Return"], bins=lv.ret_bins, labels=lv.ret_labels)
    heat = (pd.crosstab(d["股價走勢"], d["噪音分級"])
              .reindex(index=lv.ret_labels, columns=lv.noise_labels, fill_value=0)
              .iloc[::-1])                            # 暴漲在上、暴跌在下
    corr_recent, corr_all = corr_table(recent), corr_table(df_view)
    corr_df = pd.DataFrame({f"近{RECENT_N}日": corr_recent, f"近{len(df_view)}日": corr_all}).round(4)
    # 前瞻報酬研究用的資料（預設全部歷史，事件樣本比 250 日視窗多）
    df_fwd = (df_all if FWD_LOOKBACK_DAYS is None else df_all.tail(FWD_LOOKBACK_DAYS)).copy()
    df_fwd["噪音分級"] = pd.cut(df_fwd["Retail_Noise"], bins=lv.noise_bins, labels=lv.noise_labels)
    df_fwd["股價走勢"] = pd.cut(df_fwd["Daily_Return"], bins=lv.ret_bins, labels=lv.ret_labels)
    c = SimpleNamespace(ticker=ticker, df_all=df_all, df_view=df_view, recent=recent, scale=scale, lv=lv,
                        d=d, heat=heat, corr_recent=corr_recent, corr_all=corr_all, corr_df=corr_df, df_fwd=df_fwd)
    c.fwd_table = forward_table(c)
    c.extreme = extremity(c)
    return c


def analyze_all(raws):
    """逐檔分析（依 TICKERS 順序）；某檔失敗不影響其他檔。"""
    ctxs, errors = {}, {}
    for t in [t for t in TICKERS if t in raws] + [t for t in raws if t not in TICKERS]:
        try:
            ctxs[t] = analyze(t, raws[t])
        except Exception as e:
            errors[t] = e
            print(f"❌ {t} 分析失敗：{e}")
    return ctxs, errors


def summarize(c):
    last, v = c.df_view.iloc[-1], c.df_view
    return {"標的": c.ticker, "日期": f"{v.index[-1]:%Y-%m-%d}", "收盤": round(last.Close, 2),
            "日報酬%": round(last.Daily_Return, 2), "HV20": round(last.HV_20d, 1),
            "Retail_Noise": round(last.Retail_Noise, 1), "噪音等級": noise_level(last.Retail_Noise, c.lv),
            f"近{RECENT_N}日活躍天數": int((c.recent.Retail_Noise > c.lv.active).sum()),
            f"近{RECENT_N}日平均噪音": round(c.recent.Retail_Noise.mean(), 1), "分級尺度": c.scale}


def summary_table(ctxs):
    """多檔並排摘要，一列一檔。"""
    return pd.DataFrame([summarize(c) for c in ctxs.values()]).set_index("標的")


def print_summary(c):
    s, v = summarize(c), c.df_view
    print(f"=== {c.ticker} {day_tag(v.index[-1])}（{v.index[-1]:%Y-%m-%d} 美股收盤）｜現在 {fmt_market_time()} ===")
    print(f"收盤 {s['收盤']:.2f}｜日報酬 {s['日報酬%']:+.2f}%｜HV20 {s['HV20']:.1f}｜"
          f"Retail_Noise {s['Retail_Noise']:+.1f}（{s['噪音等級']}）")
    print(f"近 {RECENT_N} 日：噪音 > {c.lv.active:g} 的天數 = {s[f'近{RECENT_N}日活躍天數']}/{RECENT_N}，"
          f"平均噪音 {s[f'近{RECENT_N}日平均噪音']:+.1f}｜分級尺度 ×{c.scale:g}")
    e = c.extreme
    print(f"極端度：今日報酬在前 {e['n_hist']} 個交易日的第 {e['ret_pct']:.0f} 百分位、噪音第 {e['noise_pct']:.0f} 百分位")
    if e["beyond_ret"]:
        lo = e["ret"] < e["ret_min"]
        print(f"⚠️ 今日報酬 {e['ret']:+.2f}% 比近 {e['n_hist']} 日{'最低' if lo else '最高'}的 "
              f"{e['ret_min' if lo else 'ret_max']:+.2f}% 還極端：散佈圖的擬合線在此是外推，不可靠")
    if e["beyond_noise"]:
        print(f"⚠️ 今日噪音 {e['noise']:+.1f} 超過近 {e['n_hist']} 日最大值 {e['noise_max']:+.1f}")


# ---------- 繪圖層 ----------
def today_cell(d, heat):
    """最新一天落在熱力圖的哪一格，回傳 (row, col)（座標與 heat 的列/欄順序一致）。"""
    last = d.iloc[-1]
    return list(heat.index).index(last["股價走勢"]), list(heat.columns).index(last["噪音分級"])


def _show_df(df):
    print(df.to_string())


def _finish(fig, c, name):
    fig.tight_layout()
    note = f"產圖時間 {fmt_market_time()}｜資料截至 {c.df_view.index[-1]:%Y-%m-%d} 美股收盤"
    if c.scale != 1:
        note += f"｜分級尺度 ×{c.scale:g}"
    fig.text(0.005, 0.003, note, ha="left", va="bottom", fontsize=8, color="gray")
    if SAVE_FIGS:
        path = os.path.join(OUT_DIR, f"{c.ticker}_{name}_latest.png")   # 固定檔名覆蓋，repo 不會越長越大
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("🖼️ 已存檔：", path)
    plt.close(fig)


def plot_heatmap(c):
    """熱力圖：噪音分級 × 當日報酬；今日所在格子用不同顏色粗框標示。"""
    import seaborn as sns
    from matplotlib.patches import Rectangle, Patch
    d, heat, lv = c.d, c.heat, c.lv
    fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
    sns.heatmap(heat, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={"label": "歷史交易天數 (頻率)"}, ax=ax)
    ax.set_title(f"{c.ticker} 真實行情：散戶情緒噪音(IV-HV) ╳ 每日股價震盪熱力圖\n"
                 f"({d.index[0]:%Y-%m-%d} ~ {d.index[-1]:%Y-%m-%d}，{len(d)} 個交易日)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("散戶隨機噪音強度 (盤中異常波動 - 歷史基準波動)", fontsize=12)
    ax.set_ylabel(f"{c.ticker} 當日收盤回報率", fontsize=12)
    # 標註位置依「格子索引」計算，不寫死座標
    ax.text(len(lv.noise_labels) - 1 + 0.05, len(lv.ret_labels) - 1 + 0.25, "散戶多頭踩踏區\n(追高遭機構反手倒貨)",
            color="darkred", weight="bold", fontsize=10)
    ax.text(0.05, 0.4, "隨機嘎空區\n(散戶割肉後籌碼變輕)", color="blue", weight="bold", fontsize=10)
    # 今日（最新交易日）所在格子：不同顏色粗框 + 標注 + 圖例說明
    t_row, t_col = today_cell(d, heat)
    t = d.iloc[-1]
    tag = day_tag(d.index[-1])
    ax.add_patch(Rectangle((t_col, t_row), 1, 1, fill=False, edgecolor=TODAY_COLOR, lw=4, zorder=10))
    ax.text(t_col + 0.95, t_row + 0.92, tag, color=TODAY_COLOR, weight="bold", fontsize=11,
            ha="right", va="bottom", zorder=11)
    n_cell, n_ev = cell_stats(c)
    ax.legend(handles=[Patch(fill=False, edgecolor=TODAY_COLOR, linewidth=3,
                             label=f"{tag} {d.index[-1]:%Y-%m-%d}（美東）：噪音 {t.Retail_Noise:+.1f}、報酬 {t.Daily_Return:+.2f}%"
                                   f"｜同格歷史 {n_cell} 天、約 {n_ev} 個獨立事件")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=11)
    _finish(fig, c, "heatmap")


def plot_scatter(c):
    """散佈圖：噪音 × 每日報酬（紅點 = 最近 N 日、藍色星形 = 今日）。"""
    v, recent = c.df_view, c.recent
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.scatter(v.Retail_Noise, v.Daily_Return, s=25, alpha=0.5, color="steelblue",
               label=f"整體歷史數據 (近 {len(v)} 個交易日)")
    ax.scatter(recent.Retail_Noise, recent.Daily_Return, s=80, alpha=0.9, color="red",
               edgecolor="black", label=f"最近 {RECENT_N} 個交易日")
    # 今日超出歷史範圍時，它會把擬合線拉過去（高槓桿點）→ 擬合線只用之前的歷史來畫
    out = c.extreme["beyond_ret"] or c.extreme["beyond_noise"]
    fit = v.iloc[:-1] if out else v
    slope, intercept = np.polyfit(fit.Retail_Noise, fit.Daily_Return, 1)
    xs = np.linspace(fit.Retail_Noise.min(), fit.Retail_Noise.max(), 50)
    ax.plot(xs, slope * xs + intercept, "--", lw=2, color="darkorange",
            label=f"整體趨勢擬合線 (Corr: {c.corr_all['噪音 × 每日報酬率']:.2f}" + ("；線僅用今日以前的歷史擬合" if out else "") + ")")
    ax.axvline(0, color="gray", ls=":", alpha=0.7)
    ax.axhline(0, color="gray", ls=":", alpha=0.7)
    # 今日（最新交易日）：不同顏色的大星形 + 箭頭標注（標注位置自動避開圖框邊緣）
    tdy = v.iloc[-1]
    tag = day_tag(v.index[-1])
    ax.scatter([tdy.Retail_Noise], [tdy.Daily_Return], s=380, marker="*", color=TODAY_COLOR,
               edgecolor="black", linewidth=1.5, zorder=5,
               label=f"{tag} {v.index[-1]:%Y-%m-%d}（美東；噪音 {tdy.Retail_Noise:+.1f}、報酬 {tdy.Daily_Return:+.2f}%）"
                     + ("｜超出歷史範圍，擬合線在此為外推" if out else ""))
    xl, yl = ax.get_xlim(), ax.get_ylim()
    dx = -50 if tdy.Retail_Noise > xl[0] + 0.7 * (xl[1] - xl[0]) else 50
    dy = -50 if tdy.Daily_Return > yl[0] + 0.7 * (yl[1] - yl[0]) else 50
    ax.annotate(tag, (tdy.Retail_Noise, tdy.Daily_Return), xytext=(dx, dy), textcoords="offset points",
                ha="center", va="center", color=TODAY_COLOR, weight="bold", fontsize=12, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=TODAY_COLOR, lw=2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TODAY_COLOR))
    ax.set_title(f"{c.ticker} 散戶情緒噪音 ╳ 每日回報率 關係散佈圖", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("散戶隨機噪音強度 (Retail_Noise)", fontsize=12)
    ax.set_ylabel(f"{c.ticker} 每日收盤回報率 (%)", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)
    _finish(fig, c, "scatter")


def plot_timeseries(c):
    """時間序列：Retail_Noise 趨勢（紅色陰影 = 最近 N 日）。"""
    v, recent = c.df_view, c.recent
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    ax.plot(v.index, v.Retail_Noise, color="steelblue", alpha=0.8, lw=1.5, label="散戶情緒噪音 (Retail_Noise)")
    ax.axvspan(recent.index.min(), recent.index.max(), color="red", alpha=0.15,
               label=f"最近 {RECENT_N} 個交易日 (極端波動期)")
    ax.axhline(0, color="black", ls="--", alpha=0.6, label="無超額噪音基準線 (Noise = 0)")
    ax.axhline(c.lv.active, color="orange", ls=":", alpha=0.8, label=f"活躍臨界線 (Noise = {c.lv.active:g})")
    ax.set_title(f"{c.ticker} 散戶情緒噪音 (Retail_Noise) 歷史時間序列趨勢圖", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("日期 (Date)", fontsize=12)
    ax.set_ylabel("情緒噪音強度 (IV - HV)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    _finish(fig, c, "timeseries")


def plot_forward(c):
    """類似日子之後 1/3/5 日的平均報酬；誤差線 = 以獨立事件重抽樣的 95% 信賴區間，樣本不足的組別畫斜線。"""
    from matplotlib.patches import Patch
    t, ks = c.fwd_table, FWD_HORIZONS
    groups = list(dict.fromkeys(t.index.get_level_values("組別")))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=100)
    w, x = 0.8 / len(ks), np.arange(len(groups))
    for i, k in enumerate(ks):
        sub = t.xs(k, level="天期").reindex(groups)
        m, lo, hi = sub["平均%"].values, sub["CI下限%"].values, sub["CI上限%"].values
        err = np.nan_to_num(np.vstack([m - lo, hi - m]))
        bars = ax.bar(x + (i - (len(ks) - 1) / 2) * w, m, w, yerr=err, capsize=3, color=colors[i % len(colors)],
                      label=f"{k} 日後")
        for b, ev in zip(bars, sub["證據"]):
            if ev == "樣本不足":
                b.set_hatch("//"); b.set_alpha(0.55)
    ax.axhline(0, color="black", lw=1)
    last = t.xs(max(ks), level="天期").reindex(groups)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{int(n)} 天／{'—' if np.isnan(e) else int(e)} 事件"
                        for g, n, e in zip(groups, last["樣本天數"], last["獨立事件數"])], fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(facecolor="white", edgecolor="gray", hatch="//", label="獨立事件 < 5（樣本不足）")],
              l + ["獨立事件 < 5（樣本不足）"], loc="best", fontsize=10)
    ax.set_title(f"{c.ticker} 類似日子之後 {'/'.join(map(str, ks))} 日的平均報酬（對數報酬 %）\n"
                 f"誤差線 = 以獨立事件重抽樣的 95% 信賴區間｜資料 {c.df_fwd.index[0]:%Y-%m-%d} ~ {c.df_fwd.index[-1]:%Y-%m-%d}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("事件後平均報酬 (%)", fontsize=12)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    _finish(fig, c, "forward")


def forward_notes(c):
    """前瞻報酬表的使用警語（自動依樣本數產生）。"""
    t, kmax = c.fwd_table, max(FWD_HORIZONS)
    notes = ["ℹ️ 今日尚無未來資料，不納入統計；表中是「歷史上類似的日子」之後的表現。",
             "ℹ️ 相隔不到 %d 個交易日的事件視為同一波（獨立事件）；信賴區間以事件為單位重抽樣，事件少時區間會很寬。" % kmax]
    for g, row in t.xs(kmax, level="天期").iterrows():
        if row["證據"] == "樣本不足":
            notes.append(f"⚠️ 「{g}」只有 {int(row['獨立事件數'])} 個獨立事件（{int(row['樣本天數'])} 天），樣本不足，結果僅供參考。")
    return notes


def report(c):
    """一檔標的的完整報告：摘要 → 相關係數 → 三張圖 → 前瞻報酬（表 + 圖 + 警語）。"""
    print("\n" + "═" * 70)
    print_summary(c)
    print("\n相關係數：前 3 列是『當日同步』關係（暴跌當天盤中波動必然放大，所以噪音高），不代表隔天會反彈；"
          "『未來 N 日報酬』列才是預測力。")
    _show_df(c.corr_df)
    plot_heatmap(c)
    plot_scatter(c)
    plot_timeseries(c)
    print(f"\n── {c.ticker} 前瞻報酬：類似日子之後的表現（對數報酬 %，基準 = 全部交易日）──")
    _show_df(c.fwd_table)
    for n in forward_notes(c):
        print(n)
    plot_forward(c)

# ---------- 主程式 ----------
class _Tee:
    """同時輸出到終端機（Actions log）與檔案。"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def append_summary_log(summary_df):
    """每日摘要追加到 data/summary_log.csv；同一(日期, 標的)只保留最新一筆。"""
    path = os.path.join(DATA_DIR, "summary_log.csv")
    new = summary_df.reset_index()
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"日期": str})
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["日期", "標的"], keep="last").sort_values(["日期", "標的"])
    new.to_csv(path, index=False)


def main():
    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, buf)
    try:
        raws, update_errors = update_all(TICKERS, force=FORCE_FULL)
        print("\n" + "─" * 70)
        for t, r in raws.items():
            print(f"{t}：快取共 {len(r)} 筆，{r.index.min().date()} → {r.index.max().date()}")
        print(f"更新時間：{fmt_market_time()}")

        ctxs, analyze_errors = analyze_all(raws)
        if ctxs:
            summary = summary_table(ctxs)
            print("\n" + summary.to_string())
            append_summary_log(summary)
        for t in TICKERS:
            if t in ctxs:
                report(ctxs[t])
            else:
                print(f"\n⚠️ {t} 沒有可用資料，略過")
        errors = {**update_errors, **analyze_errors}
        if errors:
            print("\n⚠️ 有標的失敗：", {k: str(v) for k, v in errors.items()})
    finally:
        sys.stdout = real_stdout
        with open(os.path.join(OUT_DIR, "report_latest.txt"), "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
    # 全部失敗才讓 Actions 顯示紅燈；部分失敗仍保留其他標的的結果
    return 1 if not ctxs else 0


if __name__ == "__main__":
    sys.exit(main())
