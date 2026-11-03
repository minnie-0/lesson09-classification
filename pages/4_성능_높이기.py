        title=f"{KOR[x_col]} × {KOR[y_col]} (테스트 데이터 {len(X_test):,}명)",
        xaxis=dict(title=KOR[x_col], range=[x_lo, x_hi]),
        yaxis=dict(title=KOR[y_col], range=[y_lo, y_hi]),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=680,
    )
    st.plotly_chart(fig, use_container_width=True)

    others_text = fixed_text([x_col, y_col])
    if others_text:
        st.write(f"가로축과 세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {others_text}")
    for note in notes:
        st.write(note)

# ---------------------------------------------------------------
# 4) 세 속성으로 본 3차원 그림 (0.5 평면, 마우스로 돌려 볼 수 있다)
# ---------------------------------------------------------------
if len(selected) >= 3:
    st.subheader("🧊 세 속성으로 본 0.5 평면 (3차원)")
    t1, t2, t3 = st.columns(3)
    ax_x = t1.selectbox("가로축", selected, index=0, format_func=lambda c: KOR[c], key="axis3_x")
    ax_y = t2.selectbox("세로축", selected, index=1, format_func=lambda c: KOR[c], key="axis3_y")
    ax_z = t3.selectbox("높이축", selected, index=2, format_func=lambda c: KOR[c], key="axis3_z")
    axes3 = [ax_x, ax_y, ax_z]

    if len(set(axes3)) < 3:
        st.warning("세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        ranges3 = {c: axis_range(c) for c in axes3}

        def plane_for(model, n: int = 40):
            """세 축을 가로지르는 0.5 평면. 로지스틱 회귀는 확률 0.5가 판별식 0인 평면이다."""
            base = pd.DataFrame([{c: float(medians[c]) for c in selected}])
            for a in axes3:
                base[a] = 0.0
            f0 = float(model.decision_function(scaler.transform(base))[0])
            coefs = []
            for a in axes3:
                row = base.copy()
                row[a] = 1.0
                coefs.append(float(model.decision_function(scaler.transform(row))[0]) - f0)
            k = int(np.argmax(np.abs(coefs)))
            if abs(coefs[k]) < 1e-12:
                return "none", None
            i, j = [m for m in range(3) if m != k]
            g1 = np.linspace(*ranges3[axes3[i]], n)
            g2 = np.linspace(*ranges3[axes3[j]], n)
            G1, G2 = np.meshgrid(g1, g2)
            Zk = -(f0 + coefs[i] * G1 + coefs[j] * G2) / coefs[k]
            lo, hi = ranges3[axes3[k]]
            Zk = np.where((Zk >= lo) & (Zk <= hi), Zk, np.nan)
            if np.isnan(Zk).all():
                return "outside", None
            arrays = [None, None, None]
            arrays[i], arrays[j], arrays[k] = G1, G2, Zk
            return "ok", arrays

        fig3 = go.Figure()
        for label, name, color, size, opacity in [
            (0, "실제 뇌졸중 아님", C_NO, 3, 0.35),
            (1, "실제 뇌졸중", C_YES, 5, 0.95),
        ]:
            part = X_test[(y_test == label).values]
            fig3.add_trace(
                go.Scatter3d(
                    x=part[ax_x].tolist(),
                    y=part[ax_y].tolist(),
                    z=part[ax_z].tolist(),
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=size, opacity=opacity),
                )
            )

        notes3 = []
        for key, color, label in [("그대로", C_BEFORE, "가중치를 맞추기 전"), ("맞춤", C_AFTER, "가중치를 맞춘 뒤")]:
            status, arrays = plane_for(top_models[NAME_LOGIT][key])
            plane_name = f"{label}의 {NAME_LOGIT} 0.5 평면"
            if status == "ok":
                fig3.add_trace(
                    go.Surface(
                        x=to_list_with_none(arrays[0]),
                        y=to_list_with_none(arrays[1]),
                        z=to_list_with_none(arrays[2]),
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=0.45,
                        hoverinfo="skip",
                    )
                )
                visible = float(np.isfinite(arrays[0] * arrays[1] * arrays[2]).mean())
                if visible < 0.05:
                    notes3.append(
                        f"{label}의 {NAME_LOGIT} 0.5 평면은 거의 다 이 그림 밖에 있고, 아주 작은 조각만 안에 걸쳐 있어요."
                    )
            elif status == "outside":
                notes3.append(f"{label}의 {NAME_LOGIT} 0.5 평면은 이 그림 밖에 있어요.")
            else:
                notes3.append(f"{label}의 {NAME_LOGIT}은 고른 세 속성으로는 0.5 평면이 만들어지지 않아요.")
            fig3.add_trace(  # 범례용 안내 항목
                go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers", name=plane_name,
                    marker=dict(size=7, color=color, symbol="square"),
                )
            )

        fig3.update_layout(
            title=f"{KOR[ax_x]} × {KOR[ax_y]} × {KOR[ax_z]} (테스트 데이터 {len(X_test):,}명)",
            scene=dict(
                xaxis=dict(title=KOR[ax_x], range=list(ranges3[ax_x])),
                yaxis=dict(title=KOR[ax_y], range=list(ranges3[ax_y])),
                zaxis=dict(title=KOR[ax_z], range=list(ranges3[ax_z])),
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.02, x=0),
            margin=dict(l=0, r=0, t=50, b=0),
            height=720,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("마우스로 끌면 돌려 볼 수 있고, 휠로 크게 작게 볼 수 있어요.")

        fixed3 = fixed_text(axes3)
        if fixed3:
            st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed3}")
        for note in notes3:
            st.write(note)
  
