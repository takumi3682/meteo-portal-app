import streamlit as st

# ページ設定は app.py に1回だけ記述する
st.set_page_config(
    page_title="気象解析ポータル",
    page_icon="⛅",
    layout="wide",
)

# 各アプリのページ定義
pages = {
    "実況・観測データ": [
        st.Page("pages/app_surface.py", title="地上天気図", icon="🌤️"),
        st.Page("pages/app_MetViewer.py", title="アメダス時系列", icon="📈"),
        st.Page("pages/app_SkewT.py", title="エマグラム", icon="🎈"),
    ],
    "数値予報モデル": [
        st.Page("pages/app_MeteoSynoptic.py", title="高層天気図 (GSM/MSM)", icon="🌐"),
    ],
}

# ナビゲーションの実行
pg = st.navigation(pages)
pg.run()