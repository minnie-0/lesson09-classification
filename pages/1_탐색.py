
import pandas as pd
import streamlit as st
import plotly.express as px

# 데이터 주소
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(error)
    st.stop()


# 제목
st.title("🔎 데이터 탐색")
st.write("뇌졸중 데이터의 분포와 여러 요인에 따른 차이를 살펴봅니다.")

# 뇌졸중 여부 표시
df["뇌졸중 여부"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# 1. 나이와 평균 혈당의 분포
st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    st.subheader("나이 분포")

    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이에 따른 사람 수",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)


with col2:
    st.subheader("평균 혈당 분포")

    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당에 따른 사람 수",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
st.header("2. 뇌졸중 여부에 따른 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이 및 평균 혈당 분포를 비교합니다."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("나이 비교")

    fig_age_box = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이",
        labels={
            "뇌졸중 여부": "뇌졸중 여부",
            "age": "나이"
        }
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:
    st.subheader("평균 혈당 비교")

    fig_glucose_box = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "뇌졸중 여부": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 평균값 계산
mean_table = (
    df.groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "뇌졸중 여부": "뇌졸중 여부",
    "age": "평균 나이",
    "avg_glucose_level": "평균 혈당"
})

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("뇌졸중 여부에 따른 평균값")

st.dataframe(
    mean_table,
    hide_index=True,
    use_container_width=True
)


# 3. 고혈압과 심장병에 따른 뇌졸중 비율
st.header("3. 질병 여부에 따른 뇌졸중 비율")

st.write(
    "고혈압과 심장병이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)


# 고혈압에 따른 뇌졸중 비율
hypertension_rate = (
    df.groupby("hypertension")["stroke"]
    .mean()
    .reset_index()
)

hypertension_rate["stroke"] *= 100

hypertension_rate["고혈압 여부"] = hypertension_rate[
    "hypertension"
].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

fig_hypertension = px.bar(
    hypertension_rate,
    x="고혈압 여부",
    y="stroke",
    text="stroke",
    title="고혈압 여부에 따른 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "stroke": "뇌졸중 비율(%)"
    }
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_hypertension.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    yaxis_range=[
        0,
        max(hypertension_rate["stroke"]) * 1.2
    ]
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


# 심장병에 따른 뇌졸중 비율
heart_rate = (
    df.groupby("heart_disease")["stroke"]
    .mean()
    .reset_index()
)

heart_rate["stroke"] *= 100

heart_rate["심장병 여부"] = heart_rate[
    "heart_disease"
].map({
    0: "심장병 없음",
    1: "심장병 있음"
})

fig_heart = px.bar(
    heart_rate,
    x="심장병 여부",
    y="stroke",
    text="stroke",
    title="심장병 여부에 따른 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "stroke": "뇌졸중 비율(%)"
    }
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_heart.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    yaxis_range=[
        0,
        max(heart_rate["stroke"]) * 1.2
    ]
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


# 4. BMI 결측값과 뇌졸중 비율
st.header("4. BMI 결측값과 뇌졸중 비율")

st.write(
    "BMI 값이 비어 있는 사람들을 따로 구분하여 "
    "전체 사람들의 뇌졸중 비율과 비교합니다."
)

bmi_missing = df[df["bmi"].isna()]

missing_count = len(bmi_missing)
missing_stroke_count = int(bmi_missing["stroke"].sum())

missing_stroke_rate = (
    missing_stroke_count / missing_count * 100
    if missing_count > 0
    else 0
)

total_stroke_rate = df["stroke"].mean() * 100

bmi_table = pd.DataFrame({
    "구분": [
        "BMI 결측값이 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_count,
        len(df)
    ],
    "뇌졸중 발생 인원": [
        missing_stroke_count,
        int(df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(missing_stroke_rate, 2),
        round(total_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True
)


# 5. 흡연 상태별 사람 수
st.header("5. 흡연 상태별 사람 수")

st.write(
    "흡연 상태의 값에 따라 사람 수를 구분합니다."
)

smoking_count = (
    df["smoking_status"]
    .fillna("결측값")
    .value_counts()
    .reset_index()
)

smoking_count.columns = [
    "흡연 상태",
    "사람 수"
]

st.dataframe(
    smoking_count,
    hide_index=True,
    use_container_width=True
)


# 마무리
st.divider()

st.caption(
    "데이터 출처: https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)
