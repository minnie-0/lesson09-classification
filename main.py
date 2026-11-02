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

st.title("🧠 뇌졸중 예측 실습실")
st.caption("뇌졸중 관련 데이터를 살펴보고 각 열의 의미를 정리하는 실습 공간입니다.")


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# 저장된 기록 불러오기
def load_notes():
    if RECORD_FILE.exists():
        try:
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                saved = json.load(file)

            if isinstance(saved, dict):
                return saved

        except (json.JSONDecodeError, OSError):
            st.warning("저장된 기록을 읽지 못했습니다.")

    return {
        "meanings": {},
        "source": ""
    }


# 기록 저장하기
def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "meanings": meanings,
        "source": source
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# 데이터 로드
try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(error)
    st.stop()


notes = load_notes()
columns = list(df.columns)


# 1. 데이터 소개
st.header("1. 데이터 소개")

st.write(
    "이 데이터는 사람들의 기본 정보와 건강 관련 정보를 담고 있으며, "
    "뇌졸중 발생 여부를 나타내는 stroke 열이 포함되어 있습니다."
)


# 2. 주요 통계 카드
st.header("2. 데이터 한눈에 보기")

total_people = len(df)
total_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 사람 수",
    f"{total_people:,}명"
)

col2.metric(
    "열 개수",
    f"{total_columns:,}개"
)

col3.metric(
    "stroke가 1인 사람 수",
    f"{stroke_count:,}명"
)

col4.metric(
    "stroke가 1인 비율",
    f"{stroke_ratio:.2f}%"
)


# 3. 열 정보 표
st.header("3. 데이터 열 정보")

st.write(
    "교재를 참고하여 각 열의 우리말 뜻을 직접 입력하세요. "
    "값의 종류와 빈 값 개수는 데이터에서 자동으로 계산됩니다."
)

metadata = pd.DataFrame({
    "열 이름": columns,
    "우리말 뜻": [
        str(notes.get("meanings", {}).get(col, ""))
        for col in columns
    ],
    "값의 종류": [
        int(df[col].nunique(dropna=True))
        for col in columns
    ],
    "빈 값 개수": [
        int(df[col].isna().sum())
        for col in columns
    ]
})

edited_metadata = st.data_editor(
    metadata,
    hide_index=True,
    use_container_width=True,
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            disabled=True
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 직접 입력하세요."
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류",
            disabled=True
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            disabled=True
        )
    },
    key="column_metadata_editor"
)


# 4. 데이터 미리 보기
st.header("4. 데이터 미리 보기")

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True
)


# 5. 데이터 출처
st.header("5. 데이터 출처")

source_text = st.text_area(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=str(notes.get("source", "")),
    height=100,
    placeholder="데이터 출처를 입력하세요."
)


# 6. 저장 버튼
if st.button("우리말 뜻과 출처 저장", type="primary"):

    meanings = {}

    for _, row in edited_metadata.iterrows():
        meaning = row["우리말 뜻"]

        if pd.isna(meaning):
            meaning = ""

        meanings[str(row["열 이름"])] = str(meaning)

    try:
        save_notes(meanings, source_text)

        st.success("우리말 뜻과 데이터 출처를 저장했습니다.")

    except OSError as error:
        st.error("기록을 저장하지 못했습니다.")
        st.exception(error)


st.caption(
    "저장 파일 위치: 내_기록/뇌졸중_예측_실습실.json"
)
