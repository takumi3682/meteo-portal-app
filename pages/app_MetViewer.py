import io
from pathlib import Path
import duckdb
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from meteo_lib import core_v4 as cv4

# ==============================================================================
# ページ設定 & フォント設定（Mac / Linux / Streamlit Cloud 両対応）
# ==============================================================================
st.set_page_config(
    page_title="気象要素時系列ビューア",
    page_icon="🌦️",
    layout="wide",
)


def setup_japanese_font() -> None:
    """MacおよびLinux(Streamlit Cloud)環境で日本語フォントを自動設定する"""
    # リポジトリ内に同梱されたフォントがあれば優先登録（例: fonts/ipaexg.ttf 等）
    font_candidates = [
        Path("fonts/ipaexg.ttf"),
        Path("fonts/NotoSansJP-Regular.ttf"),
        Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"),
        Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
    ]
    custom_names = []
    for fp in font_candidates:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            prop = font_manager.FontProperties(fname=str(fp))
            custom_names.append(prop.get_name())

    plt.rcParams["font.sans-serif"] = custom_names + [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "AppleGothic",
        "IPAexGothic",
        "Noto Sans CJK JP",
        "Yu Gothic",
        "Meiryo",
    ]
    plt.rcParams["axes.unicode_minus"] = False


setup_japanese_font()

# ==============================================================================
# ストレージ環境の自動判定（ローカル外付けSSD vs Streamlit Cloud）
# ==============================================================================
LOCAL_JMA_DIR = Path("/Volumes/Transcend/projects/MetData/JMA")
IS_LOCAL_STORAGE = LOCAL_JMA_DIR.exists()
BASE_DIR = LOCAL_JMA_DIR if IS_LOCAL_STORAGE else None

WIND_DIR_DICT = {
    "北": 360.0, "北北東": 22.5, "北東": 45.0, "東北東": 67.5,
    "東": 90.0, "東南東": 112.5, "南東": 135.0, "南南東": 157.5,
    "南": 180.0, "南南西": 202.5, "南西": 225.0, "西南西": 247.5,
    "西": 270.0, "西北西": 292.5, "北西": 315.0, "北北西": 337.5,
    "静穏": 0.0,
}

# ==============================================================================
# ユーティリティ関数
# ==============================================================================
# リポジトリ内の data/地点コード.csv を指す（無ければ None でデフォルトにフォールバック）
APP_ROOT = Path(__file__).resolve().parent
LOCAL_AMD_CSV = APP_ROOT / "data" / "地点コード.csv"
AMD_CSV_PATH = str(LOCAL_AMD_CSV) if LOCAL_AMD_CSV.exists() else None


@st.cache_data
def get_station_catalog() -> dict[str, list[str]]:
    """都道府県名 -> 地点名リスト の辞書を作成"""
    amd_df = cv4._load_amd_list(path=AMD_CSV_PATH)
    catalog: dict[str, list[str]] = {}
    for pref_name, prec_no in cv4.prec_list.items():
        places = amd_df.loc[amd_df["prec_no"] == prec_no, "Name"].dropna().tolist()
        if places:
            catalog[pref_name] = places
    return catalog

def to_jst_naive(s: pd.Series) -> pd.Series:
    """
    core_v4.save_dataset() で保存された Parquet は UTC 数値の naive timestamp になっているため、
    naive な場合は UTC とみなして Asia/Tokyo (+9時間) へ変換する。
    """
    dt = pd.to_datetime(s)
    if dt.dt.tz is None:
        return dt.dt.tz_localize("UTC").dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)


def get_default_period() -> tuple[pd.Timestamp, pd.Timestamp]:
    """常に最新の1日分（前日 00:00 〜 当日 00:00）をデフォルトにする"""
    today_00 = pd.Timestamp.now().normalize()
    return today_00 - pd.Timedelta(days=1), today_00


def calc_dewpoint(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """気温(℃)と相対湿度(%)から露点温度(℃)を算出（Magnus式）"""
    a, b = 17.625, 243.04
    rh_safe = rh_pct.clip(lower=0.1, upper=100.0)
    alpha = np.log(rh_safe / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * alpha) / (a - alpha)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one_day_cached(
    pref: str,
    place: str,
    prec: int,
    block: int,
    station_class: str,
    freq: str,
    year: int,
    month: int,
    day: int,
) -> tuple[pd.DataFrame, dict]:
    """オンメモリキャッシュ付き1日取得関数（クラウド環境・再描画時の高速化）"""
    st_obj = cv4.Station(pref=pref, place=place, prec=prec, block=block, station_class=station_class)
    df_d, meta = cv4.fetch_one_day(
        station=st_obj,
        freq=freq,
        year=year,
        month=month,
        day=day,
        sleep_sec=0.4,
    )
    df_d = cv4.ensure_datetime_column(df_d, "datetime")
    return df_d, meta


# ==============================================================================
# データ取得関数（ローカルParquet優先＋不足分JMA補完 / クラウド自動対応）
# ==============================================================================
def load_met_data_st(
    station: cv4.Station,
    interval: str,
    start_dt: str,
    end_dt: str,
    save_cache: bool = True,
) -> pd.DataFrame:
    b_str = f"b{int(station.block):05d}"
    t0 = pd.to_datetime(start_dt)
    t1 = pd.to_datetime(end_dt)

    # ------------------------------------------------------------------
    # A. 毎時データ (hourly)
    # ------------------------------------------------------------------
    if interval == "hourly":
        df_local = pd.DataFrame()
        station_dir = (BASE_DIR / "hourly" / b_str) if IS_LOCAL_STORAGE else None
        parquet_file = (station_dir / f"{b_str}.parquet") if station_dir else None

        # 1) ローカル環境かつ統合Parquetがあれば抽出
        if parquet_file is not None and parquet_file.exists():
            q_start = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            q_end = (t1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            con = duckdb.connect()
            query = f"""
            SELECT *
            FROM read_parquet('{parquet_file.resolve()}')
            WHERE CAST(datetime AS TIMESTAMP) BETWEEN '{q_start}' AND '{q_end}'
            ORDER BY datetime;
            """
            df_local = con.execute(query).df()
            con.close()

        # 2) ローカルでカバーできていない日付を特定
        req_days = set(pd.date_range(t0.normalize(), t1.normalize(), freq="D").date)
        have_days = set()
        if not df_local.empty:
            local_dates = to_jst_naive(df_local["datetime"])
            counts = local_dates.dt.date.value_counts()
            have_days = set(counts[counts >= 24].index)

        missing_days = sorted(req_days - have_days)

        # 3) 不足日があれば気象庁サイトから補完取得
        dfs_hourly = [df_local] if not df_local.empty else []
        if missing_days:
            if IS_LOCAL_STORAGE and station_dir is not None:
                station_dir.mkdir(parents=True, exist_ok=True)

            progress_text = f"気象庁サイトから毎時データを取得中 ({station.place})..."
            bar = st.progress(0.0, text=progress_text)

            for idx, d in enumerate(missing_days):
                day_pq = (station_dir / f"{d:%Y-%m-%d}.parquet") if station_dir else None
                if day_pq is not None and day_pq.exists():
                    df_d = pd.read_parquet(day_pq)
                else:
                    df_d, meta = fetch_one_day_cached(
                        station.pref, station.place, station.prec, station.block, station.station_class,
                        "hourly", d.year, d.month, d.day
                    )
                    if IS_LOCAL_STORAGE and save_cache and station_dir is not None and not df_d.empty:
                        cv4.save_dataset(
                            df_d,
                            out_dir=station_dir,
                            basename=f"{d:%Y-%m-%d}",
                            export_csv=False,
                            parquet_metadata=meta,
                        )
                        df_d = pd.read_parquet(day_pq)
                    elif not df_d.empty:
                        df_d = df_d.copy()
                        df_d["datetime"] = pd.to_datetime(df_d["datetime"]).dt.tz_localize("Asia/Tokyo")

                dfs_hourly.append(df_d)
                bar.progress((idx + 1) / len(missing_days), text=f"{progress_text} [{idx + 1}/{len(missing_days)}日]")

            bar.empty()

        if not dfs_hourly:
            return pd.DataFrame()

        df = pd.concat(dfs_hourly, ignore_index=True)
        df["date"] = to_jst_naive(df["datetime"])
        df = (
            df[(df["date"] >= t0) & (df["date"] <= t1)]
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if df.empty:
            return df

        df["ws"] = pd.to_numeric(df.get("wind_speed"), errors="coerce")
        df["wind_dir_str"] = df.get("wind_dir")
        df["gust"] = np.nan
        return df

    # ------------------------------------------------------------------
    # B. 10分値データ (10min)
    # ------------------------------------------------------------------
    elif interval == "10min":
        dfs = []

        if IS_LOCAL_STORAGE and BASE_DIR is not None:
            # 【ローカルモード】月次Parquetキャッシュを利用・作成
            monthly_dir = BASE_DIR / "10min_monthly" / b_str
            monthly_dir.mkdir(parents=True, exist_ok=True)
            months = pd.period_range(t0.to_period("M"), t1.to_period("M"), freq="M")

            for ym in months:
                pq_month = monthly_dir / f"{ym}.parquet"
                if pq_month.exists():
                    df_m = pd.read_parquet(pq_month)
                else:
                    m_start = ym.to_timestamp()
                    m_end = min((ym + 1).to_timestamp() - pd.Timedelta(days=1), pd.Timestamp.now().normalize())
                    days = pd.date_range(m_start, m_end, freq="D")
                    day_dfs = []

                    progress_text = f"気象庁サイトから10分データを取得中 ({station.place} / {ym})..."
                    bar = st.progress(0.0, text=progress_text)

                    for idx, d in enumerate(days):
                        df_d, _ = fetch_one_day_cached(
                            station.pref, station.place, station.prec, station.block, station.station_class,
                            "10min", d.year, d.month, d.day
                        )
                        day_dfs.append(df_d)
                        bar.progress((idx + 1) / len(days), text=f"{progress_text} [{idx + 1}/{len(days)}日]")

                    bar.empty()
                    df_m = pd.concat(day_dfs, ignore_index=True).sort_values("datetime")

                    if save_cache and not df_m.empty:
                        cv4.save_dataset(
                            df_m,
                            out_dir=monthly_dir,
                            basename=str(ym),
                            export_csv=False,
                            parquet_metadata={"prec_no": station.prec, "block_no": station.block, "freq": "10min"},
                        )
                        df_m = pd.read_parquet(pq_month)
                        st.toast(f"キャッシュ保存完了: {ym}.parquet", icon="💾")
                    else:
                        df_m = df_m.copy()
                        df_m["datetime"] = pd.to_datetime(df_m["datetime"]).dt.tz_localize("Asia/Tokyo")

                dfs.append(df_m)
        else:
            # 【クラウドモード】指定期間の日数分だけオンデマンド取得（高速）
            days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
            progress_text = f"気象庁サイトから10分データを取得中 ({station.place})..."
            bar = st.progress(0.0, text=progress_text)

            for idx, d in enumerate(days):
                df_d, _ = fetch_one_day_cached(
                    station.pref, station.place, station.prec, station.block, station.station_class,
                    "10min", d.year, d.month, d.day
                )
                if not df_d.empty:
                    df_d = df_d.copy()
                    df_d["datetime"] = pd.to_datetime(df_d["datetime"]).dt.tz_localize("Asia/Tokyo")
                    dfs.append(df_d)
                bar.progress((idx + 1) / len(days), text=f"{progress_text} [{idx + 1}/{len(days)}日]")

            bar.empty()

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df["date"] = to_jst_naive(df["datetime"])
        df = (
            df[(df["date"] >= t0) & (df["date"] <= t1)]
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

        if df.empty:
            return df

        df["ws"] = pd.to_numeric(df.get("wind_speed_mean"), errors="coerce")
        df["wind_dir_str"] = df.get("wind_dir_mean")
        df["gust"] = pd.to_numeric(df.get("wind_gust"), errors="coerce")
        return df

    else:
        raise ValueError(f"未対応の interval: {interval}")


# ==============================================================================
# グラフ描画関数
# ==============================================================================
def create_met_figure(
    df: pd.DataFrame,
    station: cv4.Station,
    interval: str,
    start_dt: str,
    moisture_mode: str,
    show_gust: bool,
    barb_step: int,
) -> plt.Figure:
    df = df.copy()
    df["wd"] = df["wind_dir_str"].astype(str).str.strip().map(WIND_DIR_DICT)
    df["precip_clean"] = pd.to_numeric(df.get("precip"), errors="coerce").fillna(0.0)
    df["temp_clean"] = pd.to_numeric(df.get("temp"), errors="coerce")
    df["rh_clean"] = pd.to_numeric(df.get("rh"), errors="coerce")

    if "dewpoint" in df.columns:
        td_raw = pd.to_numeric(df["dewpoint"], errors="coerce")
        df["td_clean"] = td_raw.fillna(calc_dewpoint(df["temp_clean"], df["rh_clean"]))
    else:
        df["td_clean"] = calc_dewpoint(df["temp_clean"], df["rh_clean"])

    has_pres = False
    pres_label = ""
    if "pres_msl" in df.columns and df["pres_msl"].notna().any():
        df["pres_clean"] = pd.to_numeric(df["pres_msl"], errors="coerce")
        pres_label = "海面気圧 (hPa)"
        has_pres = True
    elif "pres_local" in df.columns and df["pres_local"].notna().any():
        df["pres_clean"] = pd.to_numeric(df["pres_local"], errors="coerce")
        pres_label = "現地気圧 (hPa)"
        has_pres = True

    plot_gust = (interval == "10min") and show_gust and df["gust"].notna().any()

    fig, ax_temp = plt.subplots(figsize=(12, 5.4), dpi=300)
    dates = df["date"]
    lines_for_legend = []

    # --- 1) 左軸（第1軸）: 気温 + 露点温度 ---
    p_temp, = ax_temp.plot(
        dates, df["temp_clean"], color="red", linewidth=1.6, label="気温 (℃)", zorder=5
    )
    lines_for_legend.append(p_temp)

    if moisture_mode == "dewpoint" and df["td_clean"].notna().any():
        p_td, = ax_temp.plot(
            dates, df["td_clean"], color="darkorange", linestyle="--", linewidth=1.3, label="露点温度 (℃)", zorder=4
        )
        lines_for_legend.append(p_td)
        ax_temp.set_ylabel("気温 / 露点温度 (℃)", color="red", fontsize=11)
        y_min_val = min(df["temp_clean"].min(), df["td_clean"].min())
        y_max_val = df["temp_clean"].max()
    else:
        ax_temp.set_ylabel("気温 (℃)", color="red", fontsize=11)
        y_min_val = df["temp_clean"].min()
        y_max_val = df["temp_clean"].max()

    ax_temp.tick_params(axis="y", labelcolor="red")
    ax_temp.grid(True, linestyle="--", alpha=0.4)

    t_span = max(y_max_val - y_min_val, 5.0)
    y_temp_bottom = np.floor(y_min_val - t_span * 0.22)
    y_temp_top = np.ceil(y_max_val + t_span * 0.45)
    ax_temp.set_ylim(y_temp_bottom, y_temp_top)

    # --- 2) 右第1軸: 降水量 + [最大瞬間風速 or 相対湿度] ---
    ax_right1 = ax_temp.twinx()
    bar_width = (1 / 24 * 0.8) if interval == "hourly" else (1 / 144 * 0.8)
    p_precip = ax_right1.bar(
        dates,
        df["precip_clean"],
        width=bar_width,
        color="green",
        alpha=0.85,
        label=f"降水量 ({'1時間' if interval == 'hourly' else '10分間'} mm)",
        zorder=2,
    )
    lines_for_legend.append(p_precip)

    right1_labels = ["降水量 (mm)"]
    right1_color = "green"
    max_r1 = max(df["precip_clean"].max() * 3.0, 10.0)

    if moisture_mode == "rh" and df["rh_clean"].notna().any():
        p_rh, = ax_right1.plot(
            dates, df["rh_clean"], color="teal", linestyle=":", linewidth=1.4, label="相対湿度 (%)", zorder=3
        )
        lines_for_legend.append(p_rh)
        right1_labels.append("相対湿度 (%)")
        right1_color = "teal"
        max_r1 = max(max_r1, 105.0)

    if plot_gust:
        p_gust, = ax_right1.plot(
            dates, df["gust"], color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.85, label="最大瞬間風速 (m/s)", zorder=3
        )
        lines_for_legend.append(p_gust)
        right1_labels.append("最大瞬間風速 (m/s)")
        right1_color = "tab:blue"
        max_r1 = max(max_r1, df["gust"].max() * 1.15)

    ax_right1.set_ylim(0, np.ceil(max_r1))
    ax_right1.set_ylabel(" / ".join(right1_labels), color=right1_color, fontsize=11)
    ax_right1.tick_params(axis="y", labelcolor=right1_color)

    # --- 3) 右第2軸（外側）: 気圧 ---
    if has_pres:
        ax_pres = ax_temp.twinx()
        ax_pres.spines["right"].set_position(("outward", 55))
        p_pres, = ax_pres.plot(
            dates, df["pres_clean"], color="purple", linewidth=1.5, linestyle="-.", label=pres_label, zorder=3
        )
        ax_pres.set_ylabel(pres_label, color="purple", fontsize=11)
        ax_pres.tick_params(axis="y", labelcolor="purple")

        p_min, p_max = df["pres_clean"].min(), df["pres_clean"].max()
        p_span = max(p_max - p_min, 4.0)
        ax_pres.set_ylim(p_min - p_span * 0.2, p_max + p_span * 0.35)
        lines_for_legend.append(p_pres)

    # --- 4) 上部: 気象庁地上基準の矢羽 ---
    barb_y_level = y_temp_bottom + (y_temp_top - y_temp_bottom) * 0.82
    ax_temp.axhline(barb_y_level, color="black", linewidth=0.8, zorder=1)

    df_barb = df.iloc[::barb_step].dropna(subset=["wd", "ws"]).copy()
    rad = np.radians(df_barb["wd"])
    u = -df_barb["ws"] * np.sin(rad)
    v = -df_barb["ws"] * np.cos(rad)

    is_calm = (df_barb["ws"] == 0) | (df_barb["wind_dir_str"] == "静穏")
    u = np.where(is_calm, 0.0, u)
    v = np.where(is_calm, 0.0, v)

    x_nums = mdates.date2num(df_barb["date"])
    ax_temp.barbs(
        x_nums,
        np.full(len(df_barb), barb_y_level),
        u,
        v,
        length=6.0,
        barbcolor="black",
        flagcolor="black",
        linewidth=0.9,
        barb_increments=dict(half=2.5, full=5.0, flag=25.0),
        zorder=6,
    )

    ax_temp.text(
        0.015, 0.93,
        "風向・風速（気象庁地上基準：長矢羽=5m/s, 短矢羽=2.5m/s, 〇=静穏）",
        transform=ax_temp.transAxes,
        fontsize=9,
        va="top",
    )

    # --- 5) X軸・タイトル・1行横並び凡例 ---
    total_hours = (dates.max() - dates.min()).total_seconds() / 3600
    if total_hours <= 36:
        ax_temp.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n(%m/%d)"))
    else:
        ax_temp.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    pad_td = pd.Timedelta(hours=1) if interval == "hourly" else pd.Timedelta(minutes=20)
    ax_temp.set_xlim(dates.min() - pad_td, dates.max() + pad_td)

    labels = [l.get_label() for l in lines_for_legend]
    leg = ax_temp.legend(
        lines_for_legend,
        labels,
        loc="upper right",
        ncol=len(lines_for_legend),
        columnspacing=1.0,
        handletextpad=0.4,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=8.5,
    )
    leg.set_zorder(10)

    interval_jp = "毎時" if interval == "hourly" else "10分値"
    start_str = start_dt[:10]
    ax_temp.set_title(
        f"{station.place}（{station.pref}）気象要素時系列図 [{interval_jp}] {start_str}",
        fontsize=13,
        pad=12,
    )

    fig.tight_layout()
    return fig


# ==============================================================================
# Streamlit UI（サイドバー & メイン画面）
# ==============================================================================
st.title("🌦️ 気象要素時系列ビューア")

catalog = get_station_catalog()
pref_list = list(catalog.keys())

with st.sidebar:
    st.header("1. 地点・期間設定")

    default_pref_idx = pref_list.index("宮城県") if "宮城県" in pref_list else 0
    selected_pref = st.selectbox("都道府県 / 振興局", pref_list, index=default_pref_idx)

    places = catalog[selected_pref]
    default_place_idx = places.index("仙台") if "仙台" in places else 0
    selected_place = st.selectbox("地点名", places, index=default_place_idx)

    interval = st.radio(
        "表示間隔",
        options=["10min", "hourly"],
        format_func=lambda x: "10分値" if x == "10min" else "毎時",
        index=1,
    )

    def_start_ts, def_end_ts = get_default_period()

    col_d1, col_t1 = st.columns([1.4, 1])
    with col_d1:
        start_date = st.date_input("開始日", value=def_start_ts.date())
    with col_t1:
        start_hour = st.selectbox("開始時", [f"{h:02d}:00" for h in range(24)], index=def_start_ts.hour)

    col_d2, col_t2 = st.columns([1.4, 1])
    with col_d2:
        end_date = st.date_input("終了日", value=def_end_ts.date())
    with col_t2:
        end_hour = st.selectbox("終了時", [f"{h:02d}:00" for h in range(24)], index=def_end_ts.hour)

    st.divider()
    st.header("2. 描画オプション")

    moisture_mode = st.radio(
        "湿潤要素の表示",
        options=["dewpoint", "rh"],
        format_func=lambda x: "露点温度 (℃) [推奨・左軸]" if x == "dewpoint" else "相対湿度 (%) [右軸]",
        index=0,
    )

    show_gust = st.checkbox(
        "最大瞬間風速を表示 (10分値のみ)",
        value=True,
        disabled=(interval != "10min"),
    )

    default_step = 3 if interval == "10min" else 1
    barb_step = st.slider("矢羽の間引き間隔（プロット毎）", min_value=1, max_value=12, value=default_step)

    if IS_LOCAL_STORAGE:
        save_cache = st.checkbox("新規取得したデータをローカル保存", value=True)
    else:
        save_cache = False
        st.caption("☁️ クラウド環境（一時メモリキャッシュ動作）")

    run_btn = st.button("グラフを描画", type="primary", use_container_width=True)

# 実行処理
start_dt_str = f"{start_date} {start_hour}:00"
end_dt_str = f"{end_date} {end_hour}:00"

if pd.to_datetime(start_dt_str) >= pd.to_datetime(end_dt_str):
    st.error("終了日時は開始日時より後に設定してください。")
    st.stop()

station = cv4.get_station(
    pref=selected_pref,
    place=selected_place,
    amd_list_path=AMD_CSV_PATH,
)
if station is None:
    st.error(f"地点情報が見つかりませんでした: {selected_pref} {selected_place}")
    st.stop()

try:
    with st.spinner("データを読み込み中..."):
        df = load_met_data_st(
            station=station,
            interval=interval,
            start_dt=start_dt_str,
            end_dt=end_dt_str,
            save_cache=save_cache,
        )

    if df.empty:
        st.warning("指定された期間のデータが見つかりませんでした。")
        st.stop()

    fig = create_met_figure(
        df=df,
        station=station,
        interval=interval,
        start_dt=start_dt_str,
        moisture_mode=moisture_mode,
        show_gust=show_gust,
        barb_step=barb_step,
    )

    st.pyplot(fig, use_container_width=True)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    b_str = f"b{int(station.block):05d}"
    base_fname = f"{b_str}_{station.place}_{interval}_{start_date}"

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.download_button(
            label="📥 グラフ画像を保存 (PNG)",
            data=buf_png.getvalue(),
            file_name=f"{base_fname}.png",
            mime="image/png",
            use_container_width=True,
        )
    with c2:
        csv_bytes = df.drop(columns=["date"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📄 抽出データを保存 (CSV)",
            data=csv_bytes,
            file_name=f"{base_fname}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("抽出データプレビュー"):
        st.dataframe(df.drop(columns=["date"], errors="ignore"), use_container_width=True)

except FileNotFoundError as e:
    st.error(f"ローカルファイルが見つかりません: {e}")
except Exception as e:
    st.exception(e)