
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 데이터 주소
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 기록 파일
SAVE_DIR = Path("내_기록")
SAVE_FILE = SAVE_DIR / "stroke_info.json"

# 데이터 열
COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke"
]


# 저장된 기록 불러오기
def load_saved_info():
    if SAVE_FILE.exists():
        try:
            with SAVE_FILE.open("r", encoding="utf-8") as file:
                saved = json.load(file)

            return saved.get("meanings", {}), saved.get("source", "")

        except (json.JSONDecodeError, OSError):
            return {}, ""

    return {}, ""


# 기록 저장하기
def save_info(meanings, source):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source
    }

    with SAVE_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df = df[COLUMNS].copy()

    # BMI 빈 값 처리
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    return df


# 앱 제목
st.title("🧠 뇌졸중 예측 실습실")

st.caption(
    "뇌졸중 관련 데이터를 살펴보고 "
    "각 열의 의미를 직접 기록하는 실습 공간입니다."
)


# 데이터 불러오기
try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(error)
    st.stop()


# 데이터 소개
st.header("데이터 소개")

st.write(
    "이 데이터는 사람들의 기본 정보와 건강 관련 항목, "
    "그리고 뇌졸중 발생 여부를 담고 있습니다. "
    "데이터의 크기와 각 열의 구성, 빈 값 등을 확인하고 "
    "뇌졸중 예측 실습에 사용할 자료를 이해합니다."
)


# 통계 계산
total_people = len(df)

column_count = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)


# 숫자 카드 4개
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 사람 수",
    f"{total_people:,}명"
)

col2.metric(
    "열 개수",
    f"{column_count:,}개"
)

col3.metric(
    "stroke가 1인 사람 수",
    f"{stroke_count:,}명"
)

col4.metric(
    "stroke가 1인 비율",
    f"{stroke_ratio:.2f}%"
)


st.divider()


# 열 정보
st.header("열 정보 기록")

st.write(
    "우리말 뜻과 데이터 출처는 교재를 확인한 뒤 "
    "직접 입력하세요."
)


# 기존 기록 불러오기
saved_meanings, saved_source = load_saved_info()


# 열 정보 표 만들기
info_rows = []

for column in COLUMNS:

    values = df[column].dropna().unique()

    if len(values) <= 8:
        value_text = ", ".join(
            str(value) for value in values
        )

    else:
        value_text = f"고유값 {len(values)}개"

    info_rows.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_text,
        "빈 값 개수": int(df[column].isna().sum())
    })


# 표 출력
info_df = pd.DataFrame(info_rows)

st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True
)


# 우리말 뜻과 출처 입력
st.subheader("우리말 뜻 및 출처 입력")

with st.form("meaning_form"):

    edited_meanings = {}

    input_cols = st.columns(2)

    for index, column in enumerate(COLUMNS):

        with input_cols[index % 2]:

            edited_meanings[column] = st.text_input(
                column,
                value=saved_meanings.get(column, "")
            )

    source_text = st.text_area(
        "데이터 출처",
        value=saved_source,
        height=100
    )

    submitted = st.form_submit_button("기록 저장")


    # 저장 버튼
    if submitted:

        save_info(
            edited_meanings,
            source_text
        )

        st.success(
            "내_기록/stroke_info.json에 저장했습니다."
        )

        st.rerun()


st.divider()


# 처음 다섯 줄
st.header("데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# 데이터 출처 안내
st.header("데이터 출처")

st.write(
    "교재에 적힌 데이터 출처를 위의 입력란에 작성하고 "
    "기록 저장을 눌러 주세요."
)

st.caption(
    "우리말 뜻과 출처는 내_기록/stroke_info.json에 저장됩니다."
)
