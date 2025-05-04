import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

st.title("📊 2025년 이후 지역별 가정폭력 신고 예측 및 필요 시설 수 계산 (2023년 기준 통합 격차 지수가 부족, 심각 지역만 해당하고 해당 년도에 주의 미만이 되기 위해 필요한 수(총 시설 수로 기존의 시설 수를 고려하지 않음음))")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("data/report_data.csv", encoding="utf-8")
    df = df.melt(id_vars=["신고건수 연도"], var_name="지역", value_name="신고건수")
    df = df.rename(columns={"신고건수 연도": "연도"})
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    df["신고건수"] = pd.to_numeric(df["신고건수"], errors="coerce")

    gap_df = pd.read_csv("data/gap_grades.csv", encoding="cp949")
    grade_map = gap_df.iloc[6, 1:].to_dict()

    return df, grade_map

df_long, grade_map = load_data()

# 계산 함수
def calculate_facilities(report_count, ratio_counsel=0.987):
    required_capacity = report_count / 1.99
    counsel_target = required_capacity * ratio_counsel
    shelter_target = required_capacity * (1 - ratio_counsel)
    counsel_count = int((counsel_target / 1000) + 0.999)
    shelter_count = int((shelter_target / 16.74) + 0.999)
    return shelter_count, counsel_count

# 사용자 입력
target_year = st.number_input("예측할 연도 (2026 이상)", min_value=2026, step=1, value=2026)

if st.button("예측 실행"):
    rows = []
    regions = df_long["지역"].unique()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, region in enumerate(regions):
        region_df = df_long[df_long["지역"] == region].copy().sort_values("연도")
        pred_yr_dict = {}

        for year in range(2026, target_year + 1):
            X = region_df[["연도"]]
            y = region_df["신고건수"]

            if len(X) < 2:
                pred_yr_dict[year] = "데이터 부족"
                continue

            r2 = 0
            retry = 0
            while r2 < 0.7:
                model = RandomForestRegressor(n_estimators=100, random_state=retry)
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                retry += 1

            pred = model.predict([[year]])[0]
            pred_yr_dict[year] = round(pred)
            region_df = pd.concat(
                [region_df, pd.DataFrame({"연도": [year], "신고건수": [pred]})],
                ignore_index=True
            )

        final_prediction = pred_yr_dict.get(target_year)
        grade = grade_map.get(region, None)

        if grade in ["부족", "심각"] and isinstance(final_prediction, (int, float)):
            shelter, counsel = calculate_facilities(final_prediction)
        else:
            shelter, counsel = "-", "-"

        rows.append({
            "지역": region,
            "2023 격차등급": grade,
            f"{target_year}년 예상 신고건수": final_prediction,
            "필요 보호소 수": shelter,
            "필요 상담소 수": counsel
        })

        # 진행률 업데이트
        progress = (idx + 1) / len(regions)
        progress_bar.progress(progress)
        status_text.text(f"{region} 처리 중... ({idx + 1}/{len(regions)})")

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df)

    st.download_button(
        label="📥 결과 다운로드",
        data=result_df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"{target_year}_예상신고_및_필요시설.csv",
        mime="text/csv"
    )
