
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(
    page_title="데이터 탐색",
    page_icon="🔎",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


st.title("🔎 데이터 탐색")
st.write("뇌졸중 데이터의 다양한 특징을 그래프로 살펴봅니다.")

# 뇌졸중 여부
df["뇌졸중 여부"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# 1. 나이와 평균 혈당 분포
st.header("1. 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )
    st.plotly_chart(fig, use_container_width=True)


# 2. 뇌졸중 여부에 따른 비교
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교"
    )
    st.plotly_chart(fig, use_container_width=True)


# 평균값 표
st.subheader("그룹별 평균값")

mean_table = (
    df.groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .round(2)
    .reset_index()
)

mean_table.columns = [
    "뇌졸중 여부",
    "평균 나이",
    "평균 혈당"
]

st.dataframe(
    mean_table,
    hide_index=True,
    use_container_width=True
)


# 3. 고혈압과 심장병
st.header("3. 질병 여부에 따른 뇌졸중 비율")


def draw_stroke_rate(column, title, labels):
    rate = (
        df.groupby(column)["stroke"]
        .mean()
        .mul(100)
        .reset_index()
    )

    rate[title] = rate[column].map(labels)

    fig = px.bar(
        rate,
        x=title,
        y="stroke",
        text="stroke",
        title=title + "에 따른 뇌졸중 비율",
        labels={
            title: title,
            "stroke": "뇌졸중 비율(%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="뇌졸중 비율(%)"
    )

    st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
    draw_stroke_rate(
        "hypertension",
        "고혈압 여부",
        {0: "고혈압 없음", 1: "고혈압 있음"}
    )

with col2:
    draw_stroke_rate(
        "heart_disease",
        "심장병 여부",
        {0: "심장병 없음", 1: "심장병 있음"}
    )


# 4. BMI 결측값
st.header("4. BMI 결측값과 뇌졸중 비율")

missing_bmi = df[df["bmi"].isna()]

missing_count = len(missing_bmi)
missing_stroke = int(missing_bmi["stroke"].sum())

missing_rate = (
    missing_stroke / missing_count * 100
    if missing_count > 0 else 0
)

total_rate = df["stroke"].mean() * 100

bmi_table = pd.DataFrame({
    "구분": ["BMI 결측값이 있는 사람", "전체 사람"],
    "사람 수": [missing_count, len(df)],
    "뇌졸중 발생 인원": [missing_stroke, int(df["stroke"].sum())],
    "뇌졸중 비율(%)": [
        round(missing_rate, 2),
        round(total_rate, 2)
    ]
})

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True
)


# 5. 흡연 상태
st.header("5. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_table,
    hide_index=True,
    use_container_width=True
)


st.divider()
st.caption("데이터 출처: GitHub stroke.csv")
