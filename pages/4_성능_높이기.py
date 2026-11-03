from pathlib import Path

code = r'''# pages/4_성능_높이기.py
# 가중치를 맞추기 전과 맞춘 뒤의 분류 성능을 비교하는 페이지

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

BINARY = {"hypertension", "heart_disease"}

NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"

COLOR_NO = "#3F7CAC"
COLOR_YES = "#D84315"
COLOR_BEFORE = "#424242"
COLOR_AFTER = "#2E7D32"

GRID_N = 150
DEFAULT_DEPTH = 3

st.set_page_config(
    page_title=f"성능 높이기 · {APP_TITLE}",
    page_icon="🚀",
    layout="wide",
)


# ---------------------------------------------------------------
# 데이터 불러오기 및 계산 함수
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중이에요...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def confusion(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 혼동행렬의 네 값을 계산한다."""
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def calculate_metrics(cm):
    """혼동행렬에서 정확도, 재현율, 정밀도, F1을 계산한다."""
    tp = cm["TP"]
    fp = cm["FP"]
    fn = cm["FN"]
    tn = cm["TN"]

    total = tp + fp + fn + tn
    actual_positive = tp + fn
    predicted_positive = tp + fp

    accuracy = (tp + tn) / total if total else None
    recall = tp / actual_positive if actual_positive else None
    precision = tp / predicted_positive if predicted_positive else None

    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def format_metric(value):
    return "계산할 수 없음" if value is None else f"{value:.4f}"


def axis_range(data, column):
    """그래프 축의 범위를 정한다."""
    if column in BINARY:
        return -0.4, 1.4

    low = float(data[column].min())
    high = float(data[column].max())
    padding = (high - low) * 0.05

    if padding == 0:
        padding = 1.0

    return low - padding, high + padding


def fixed_features_text(columns_to_skip, selected, medians):
    """그래프 축으로 고르지 않은 속성의 고정값을 설명한다."""
    parts = []

    for column in selected:
        if column in columns_to_skip:
            continue

        value = float(medians[column])

        if column in BINARY:
            rounded = int(round(value))
            state = "있음" if rounded == 1 else "없음"
            parts.append(f"{KOR[column]} {rounded}({state})")
        else:
            parts.append(f"{KOR[column]} {value:.1f}")

    return ", ".join(parts)


def add_test_points(fig, test_data, y_test, x_column, y_column):
    """실제 뇌졸중 여부에 따라 테스트 데이터 점을 다른 색으로 표시한다."""
    for label, name, color, size, opacity in [
        (0, "실제 뇌졸중 아님", COLOR_NO, 6, 0.5),
        (1, "실제 뇌졸중", COLOR_YES, 9, 0.9),
    ]:
        mask = y_test == label
        part = test_data.loc[mask]

        fig.add_trace(
            go.Scatter(
                x=part[x_column],
                y=part[y_column],
                mode="markers",
                name=name,
                marker={
                    "color": color,
                    "size": size,
                    "opacity": opacity,
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertemplate=(
                    f"{KOR[x_column]}: %{{x}}<br>"
                    f"{KOR[y_column]}: %{{y}}<extra>{name}</extra>"
                ),
            )
        )


# ---------------------------------------------------------------
# 화면 제목 및 데이터 불러오기
# ---------------------------------------------------------------
st.title("🚀 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어요. "
    "뇌졸중과 아님, 양쪽의 가중치를 같게 맞추면 결과가 어떻게 달라지는지 살펴봐요."
)

try:
    raw = load_data()
except Exception as error:
    st.error(
        "데이터를 불러오지 못했어요. 인터넷 연결이나 데이터 주소를 확인해 주세요.\n\n"
        f"오류 내용: {error}"
    )
    st.stop()


# ---------------------------------------------------------------
# 입력 속성 선택
# ---------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ],
    format_func=lambda column: KOR[column],
)

selected = [column for column in FEATURES if column in chosen]

if len(selected) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요.")
    st.stop()


# ---------------------------------------------------------------
# 채점표 페이지와 같은 방식으로 데이터 나누기
# 번호 순으로 정렬한 뒤, 열 명 중 앞 세 명을 테스트용으로 사용한다.
# ---------------------------------------------------------------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3

train_df = df.loc[~is_test].copy()
test_df = df.loc[is_test].copy()

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)
y_test_array = y_test.to_numpy()

# 체질량지수의 결측값은 훈련 데이터의 중앙값으로 채운다.
if "bmi" in selected:
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# ---------------------------------------------------------------
# 두 모델을 가중치 적용 전·후로 학습한다.
# 로지스틱 회귀에는 표준화를 적용하고, 의사결정트리에는 적용하지 않는다.
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    NAME_LOGIT: {
        "그대로": LogisticRegression(max_iter=2000).fit(
            X_train_scaled, y_train
        ),
        "맞춤": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ).fit(X_train_scaled, y_train),
    },
    NAME_TREE: {
        "그대로": DecisionTreeClassifier(
            max_depth=DEFAULT_DEPTH,
            min_samples_leaf=5,
            random_state=0,
        ).fit(X_train, y_train),
        "맞춤": DecisionTreeClassifier(
            max_depth=DEFAULT_DEPTH,
            min_samples_leaf=5,
            random_state=0,
            class_weight="balanced",
        ).fit(X_train, y_train),
    },
}


# ---------------------------------------------------------------
# 가중치 적용 전·후 성능 비교
# ---------------------------------------------------------------
st.subheader("⚖️ 가중치를 맞추기 전과 맞춘 뒤")

st.write(
    f"테스트용 **{len(y_test):,}명** 가운데 "
    f"실제 뇌졸중인 사람은 **{int((y_test == 1).sum()):,}명**이에요. "
    "가중치를 맞추면 적은 쪽인 뇌졸중 사례가 학습에서 더 크게 반영돼요."
)

for model_name in [NAME_LOGIT, NAME_TREE]:
    before_prediction = models[model_name]["그대로"].predict(
        X_test_scaled if model_name == NAME_LOGIT else X_test
    )
    after_prediction = models[model_name]["맞춤"].predict(
        X_test_scaled if model_name == NAME_LOGIT else X_test
    )

    before_cm = confusion(y_test_array, before_prediction)
    after_cm = confusion(y_test_array, after_prediction)

    before_metrics = calculate_metrics(before_cm)
    after_metrics = calculate_metrics(after_cm)

    rows = [
        ("찾아낸 환자 (TP)", f"{before_cm['TP']:,}명", f"{after_cm['TP']:,}명"),
        ("놓친 환자 (FN)", f"{before_cm['FN']:,}명", f"{after_cm['FN']:,}명"),
        ("헛짚은 사람 (FP)", f"{before_cm['FP']:,}명", f"{after_cm['FP']:,}명"),
        ("아님을 맞힘 (TN)", f"{before_cm['TN']:,}명", f"{after_cm['TN']:,}명"),
    ]

    for metric_name in ["정확도", "재현율", "정밀도", "F1"]:
        rows.append(
            (
                metric_name,
                format_metric(before_metrics[metric_name]),
                format_metric(after_metrics[metric_name]),
            )
        )

    st.markdown(f"**{model_name}**")
    st.dataframe(
        pd.DataFrame(
            rows,
            columns=["항목", "가중치 그대로 (전)", "가중치 맞춤 (후)"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        f"찾아낸 환자 {before_cm['TP']:,}명 → {after_cm['TP']:,}명 · "
        f"놓친 환자 {before_cm['FN']:,}명 → {after_cm['FN']:,}명 · "
        f"헛짚은 사람 {before_cm['FP']:,}명 → {after_cm['FP']:,}명"
    )


# ---------------------------------------------------------------
# 두 속성으로 본 로지스틱 회귀의 0.5 경계선
# ---------------------------------------------------------------
st.subheader("🗺️ 두 속성으로 본 0.5 경계선")

axis_col1, axis_col2 = st.columns(2)

x_column = axis_col1.selectbox(
    "가로축",
    selected,
    index=0,
    format_func=lambda column: KOR[column],
    key="boundary_x",
)
y_column = axis_col2.selectbox(
    "세로축",
    selected,
    index=1,
    format_func=lambda column: KOR[column],
    key="boundary_y",
)

medians = X_train.median()

if x_column == y_column:
    st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    x_low, x_high = axis_range(X_test, x_column)
    y_low, y_high = axis_range(X_test, y_column)

    x_values = np.linspace(x_low, x_high, GRID_N)
    y_values = np.linspace(y_low, y_high, GRID_N)

    grid = pd.DataFrame(
        {
            column: np.full(GRID_N * GRID_N, float(medians[column]))
            for column in selected
        }
    )
    grid[x_column] = np.tile(x_values, GRID_N)
    grid[y_column] = np.repeat(y_values, GRID_N)

    grid_scaled = scaler.transform(grid)

    fig = go.Figure()
    add_test_points(fig, X_test, y_test, x_column, y_column)

    boundary_notes = []

    for key, color, dash, label in [
        ("그대로", COLOR_BEFORE, "dash", "가중치를 맞추기 전"),
        ("맞춤", COLOR_AFTER, "solid", "가중치를 맞춘 뒤"),
    ]:
        model = models[NAME_LOGIT][key]
        probabilities = model.predict_proba(grid_scaled)[:, 1]
        probability_min = float(probabilities.min())
        probability_max = float(probabilities.max())

        if probability_min <= 0.5 <= probability_max:
            probability_grid = probabilities.reshape(GRID_N, GRID_N)

            fig.add_trace(
                go.Contour(
                    x=x_values,
                    y=y_values,
                    z=probability_grid.T,
                    contours={
                        "start": 0.5,
                        "end": 0.5,
                        "size": 0.1,
                        "coloring": "lines",
                    },
                    colorscale=[[0, color], [1, color]],
                    line={"width": 3, "dash": dash},
                    showscale=False,
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{label} 0.5 경계선",
                )
            )
        else:
            boundary_notes.append(
                f"{label}의 0.5 경계선은 그림 밖에 있어요. "
                f"그림 안의 뇌졸중 예측 확률은 "
                f"{probability_min:.3f}~{probability_max:.3f}예요."
            )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{label} 0.5 경계선",
                line={"color": color, "width": 3, "dash": dash},
            )
        )

    fig.update_layout(
        title=f"{KOR[x_column]} × {KOR[y_column]}",
        xaxis={"title": KOR[x_column], "range": [x_low, x_high]},
        yaxis={"title": KOR[y_column], "range": [y_low, y_high]},
        legend={"orientation": "h", "y": -0.2},
        height=650,
        margin={"l": 10, "r": 10, "t": 50, "b": 80},
    )

    st.plotly_chart(fig, use_container_width=True)

    fixed_text = fixed_features_text(
        [x_column, y_column],
        selected,
        medians,
    )
    if fixed_text:
        st.caption(
            "가로축과 세로축이 아닌 속성은 훈련 데이터의 중앙값으로 고정했어요: "
            + fixed_text
        )

    for note in boundary_notes:
        st.info(note)


# ---------------------------------------------------------------
# 세 속성으로 본 3차원 산점도와 로지스틱 회귀의 0.5 평면
# ---------------------------------------------------------------
if len(selected) >= 3:
    st.subheader("🧊 세 속성으로 본 0.5 평면 (3차원)")

    col_x, col_y, col_z = st.columns(3)

    axis_x = col_x.selectbox(
        "가로축",
        selected,
        index=0,
        format_func=lambda column: KOR[column],
        key="plane_x",
    )
    axis_y = col_y.selectbox(
        "세로축",
        selected,
        index=1,
        format_func=lambda column: KOR[column],
        key="plane_y",
    )
    axis_z = col_z.selectbox(
        "높이축",
        selected,
        index=2,
        format_func=lambda column: KOR[column],
        key="plane_z",
    )

    axes = [axis_x, axis_y, axis_z]

    if len(set(axes)) != 3:
        st.warning("3차원 그래프의 세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        ranges = {
            column: axis_range(X_test, column)
            for column in axes
        }

        # 표준화된 로지스틱 회귀식을 원래 속성 단위의 식으로 변환한다.
        before_model = models[NAME_LOGIT]["그대로"]
        after_model = models[NAME_LOGIT]["맞춤"]

        def plane_coordinates(model, number=40):
            coefficients_scaled = model.coef_[0]
            coefficients = coefficients_scaled / scaler.scale_
            intercept = model.intercept_[0] - np.sum(
                coefficients_scaled * scaler.mean_ / scaler.scale_
            )

            axis_indices = [selected.index(column) for column in axes]
            axis_coefficients = [
                coefficients[index] for index in axis_indices
            ]

            fixed_sum = intercept
            for index, column in enumerate(selected):
                if column not in axes:
                    fixed_sum += coefficients[index] * float(medians[column])

            solve_index = int(np.argmax(np.abs(axis_coefficients)))
            solve_coefficient = axis_coefficients[solve_index]

            if abs(solve_coefficient) < 1e-12:
                return "none", None

            other_indices = [
                index for index in range(3)
                if index != solve_index
            ]

            first_axis = axes[other_indices[0]]
            second_axis = axes[other_indices[1]]
            solved_axis = axes[solve_index]

            first_values = np.linspace(
                ranges[first_axis][0],
                ranges[first_axis][1],
                number,
            )
            second_values = np.linspace(
                ranges[second_axis][0],
                ranges[second_axis][1],
                number,
            )

            first_grid, second_grid = np.meshgrid(
                first_values,
                second_values,
            )

            solved_grid = -(
                fixed_sum
                + axis_coefficients[other_indices[0]] * first_grid
                + axis_coefficients[other_indices[1]] * second_grid
            ) / solve_coefficient

            solved_low, solved_high = ranges[solved_axis]
            solved_grid = np.where(
                (solved_grid >= solved_low)
                & (solved_grid <= solved_high),
                solved_grid,
                np.nan,
            )

            if np.isnan(solved_grid).all():
                return "outside", None

            coordinate_grids = [None, None, None]
            coordinate_grids[other_indices[0]] = first_grid
            coordinate_grids[other_indices[1]] = second_grid
            coordinate_grids[solve_index] = solved_grid

            return "ok", coordinate_grids

        fig3 = go.Figure()

        for label, name, color, size, opacity in [
            (0, "실제 뇌졸중 아님", COLOR_NO, 3, 0.35),
            (1, "실제 뇌졸중", COLOR_YES, 5, 0.9),
        ]:
            mask = y_test == label
            part = X_test.loc[mask]

            fig3.add_trace(
                go.Scatter3d(
                    x=part[axis_x],
                    y=part[axis_y],
                    z=part[axis_z],
                    mode="markers",
                    name=name,
                    marker={
                        "color": color,
                        "size": size,
                        "opacity": opacity,
                    },
                )
            )

        plane_notes = []

        for key, color, label in [
            ("그대로", COLOR_BEFORE, "가중치를 맞추기 전"),
            ("맞춤", COLOR_AFTER, "가중치를 맞춘 뒤"),
        ]:
            status, coordinates = plane_coordinates(models[NAME_LOGIT][key])

            if status == "ok":
                fig3.add_trace(
                    go.Surface(
                        x=coordinates[0],
                        y=coordinates[1],
                        z=coordinates[2],
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=0.4,
                        name=f"{label} 0.5 평면",
                        hoverinfo="skip",
                    )
                )
            elif status == "outside":
                plane_notes.append(
                    f"{label}의 0.5 평면은 현재 그래프 범위 밖에 있어요."
                )
            else:
                plane_notes.append(
                    f"{label}의 0.5 평면을 만들 수 없어요."
                )

            fig3.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    name=f"{label} 0.5 평면",
                    marker={
                        "size": 7,
                        "color": color,
                        "symbol": "square",
                    },
                )
            )

        fig3.update_layout(
            title=(
                f"{KOR[axis_x]} × {KOR[axis_y]} × {KOR[axis_z]}"
            ),
            scene={
                "xaxis": {
                    "title": KOR[axis_x],
                    "range": list(ranges[axis_x]),
                },
                "yaxis": {
                    "title": KOR[axis_y],
                    "range": list(ranges[axis_y]),
                },
                "zaxis": {
                    "title": KOR[axis_z],
                    "range": list(ranges[axis_z]),
                },
            },
            legend={"orientation": "h", "y": -0.1},
            height=700,
            margin={"l": 0, "r": 0, "t": 50, "b": 50},
        )

        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "마우스로 그래프를 끌면 회전할 수 있고, 휠로 확대하거나 축소할 수 있어요."
        )

        fixed_text_3d = fixed_features_text(axes, selected, medians)
        if fixed_text_3d:
            st.caption(
                "세 축이 아닌 속성은 훈련 데이터의 중앙값으로 고정했어요: "
                + fixed_text_3d
            )

        for note in plane_notes:
            st.info(note)
'''
path = Path("/mnt/data/4_성능_높이기.py")
path.write_text(code, encoding="utf-8")
compile(code, str(path), "exec")
print("전체 코드 생성 완료. Python 문법 검사 통과.")
