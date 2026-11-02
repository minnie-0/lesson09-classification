
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_예측_실습실.json"

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_notes():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"meanings": meanings, "source": source},
            f,
            ensure_ascii=False,
            indent=2
        )


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


notes = load_notes()

st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 관련 데이터를 살펴보고 분석하는 실습 공간입니다.")

# 주요 통계
st.header("1. 데이터 한눈에 보기")

total = len(df)
columns = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 사람 수", f"{total:,}명")
c2.metric("열 개수", f"{columns}개")
c3.metric("뇌졸중 발생 인원", f"{stroke_count:,}명")
c4.metric("뇌졸중 발생 비율", f"{stroke_ratio:.2f}%")

# 열 정보
st.header("2. 데이터 열 정보")

metadata = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        notes.get("meanings", {}).get(col, "")
        for col in df.columns
    ],
    "값의 종류": [
        df[col].nunique()
        for col in df.columns
    ],
    "빈 값 개수": [
        df[col].isna().sum()
        for col in df.columns
    ]
})

edited = st.data_editor(
    metadata,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"]
)

# 원본 데이터
st.header("3. 데이터 처음 다섯 줄")
st.dataframe(df.head(), use_container_width=True)

# 출처
st.header("4. 데이터 출처")

source = st.text_area(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=notes.get("source", "")
)

if st.button("기록 저장", type="primary"):
    meanings = dict(zip(edited["열 이름"], edited["우리말 뜻"]))
    save_notes(meanings, source)
    st.success("기록을 저장했습니다.")
