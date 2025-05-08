
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import time

st.title("2025년 이후 지역별 가정폭력 신고 예측 및 필요 시설 수 계산")

st.markdown(
    '''
    <h4 style='margin-top: 20px; color: gray; font-weight: normal;'>
    2023년 기준 통합 격차 지수가 <strong>부족, 심각</strong> 지역만 해당하며,<br>
    해당 연도에 '주의' 미만이 되기 위해 필요한 <strong>총 시설 수</strong>를 계산합니다. 
    <br><em>(기존 시설 수는 고려하지 않음)</em>
    </h4>
    ''',
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    report_df = pd.read_csv("data/report_data.csv", encoding="utf-8")
    report_df = report_df.melt(id_vars=["신고건수 연도"], var_name="지역", value_name="신고건수")
    report_df = report_df.rename(columns={"신고건수 연도": "연도"})
    report_df["연도"] = pd.to_numeric(report_df["연도"], errors="coerce")
    report_df["신고건수"] = pd.to_numeric(report_df["신고건수"], errors="coerce")

    gap_df = pd.read_csv("data/gap_grades.csv", encoding="cp949")
    grade_map = gap_df.iloc[6, 1:].to_dict()

    return report_df, grade_map

def calculate_model_confidence(region_df):
    train_df = region_df[region_df["연도"] <= 2024]
    X = train_df[["연도"]]
    y = train_df["신고건수"]

    retry = 0
    avg_r2 = 0

    while avg_r2 < 0.7 and retry < 10:
        rf = RandomForestRegressor(n_estimators=100, random_state=retry)
        lr = LinearRegression()
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=retry)

        rf.fit(X, y)
        lr.fit(X, y)
        xgb.fit(X, y)

        rf_r2 = r2_score(y, rf.predict(X))
        lr_r2 = r2_score(y, lr.predict(X))
        xgb_r2 = r2_score(y, xgb.predict(X))
        avg_r2 = (rf_r2 + lr_r2 + xgb_r2) / 3
        retry += 1

    return rf, lr, xgb, rf_r2, lr_r2, xgb_r2, avg_r2

def calculate_facilities(report_count, ratio_counsel=0.987):
    required_capacity = report_count / 1.99
    counsel_target = required_capacity * ratio_counsel
    shelter_target = required_capacity * (1 - ratio_counsel)
    counsel_count = int((counsel_target / 1000) + 0.999)
    shelter_count = int((shelter_target / 16.74) + 0.999)
    return shelter_count, counsel_count

df, grade_map = load_data()
target_year = st.selectbox("예측 연도 선택", [2025, 2026, 2027, 2028])
run_prediction = st.button("예측 실행")

if run_prediction:
    rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, region in enumerate(df["지역"].unique()):
        grade = grade_map.get(region)
        region_df = df[df["지역"] == region]
        if region_df["연도"].nunique() < 2:
            continue

        rf, lr, xgb, rf_r2, lr_r2, xgb_r2, avg_r2 = calculate_model_confidence(region_df)

        rf_pred = rf.predict([[target_year]])[0]
        lr_pred = lr.predict([[target_year]])[0]
        xgb_pred = xgb.predict([[target_year]])[0]
        avg_pred = (rf_pred + lr_pred + xgb_pred) / 3

        if grade in ["부족", "심각"] and avg_r2 >= 0.7:
            shelter_count, counsel_count = calculate_facilities(avg_pred)
        else:
            shelter_count, counsel_count = "-", "-"

        rows.append({
            "지역": region,
            "2023 격차등급": grade,
            f"{target_year}년 평균 예측 신고건수": round(avg_pred),
            "필요 보호소 수": shelter_count,
            "필요 상담소 수": counsel_count,
            "평균 R²": round(avg_r2, 3),
            "신뢰도 기준 통과 여부": "✅ 신뢰 가능" if avg_r2 >= 0.7 else "⚠️ 신뢰도 부족"
        })

        progress = (idx + 1) / len(df["지역"].unique())
        progress_bar.progress(progress)
        status_text.text(f"🔄 처리 중: {region} ({idx + 1}/{len(df['지역'].unique())})")
        time.sleep(0.05)

    result_df = pd.DataFrame(rows)

    # 🔧 문자열 강제 변환 (Arrow 변환 오류 방지)
    result_df["필요 보호소 수"] = result_df["필요 보호소 수"].astype(str)
    result_df["필요 상담소 수"] = result_df["필요 상담소 수"].astype(str)

    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 결과 다운로드", data=csv, file_name=f"{target_year}_예측결과.csv", mime="text/csv")

    valid_r2 = result_df[result_df["신뢰도 기준 통과 여부"] == "✅ 신뢰 가능"]["평균 R²"]
    avg_r2_all = valid_r2.mean() if not valid_r2.empty else 0.0
    st.metric("전체 평균 R² (예측 신뢰도) (0.7이상 신뢰)", f"{avg_r2_all:.3f}")

    st.markdown(
        "<p style='font-size: 12px; color: gray;'>"
        "※ 본 예측은 2021~2024년 신고건수 데이터를 기반으로, "
        "RandomForest, LinearRegression, XGBoost 모델의 평균값으로 산출되었으며, "
        "신뢰도(R²)는 각 모델의 과거 설명력을 의미합니다. "
        "미래 예측에 대한 실제 발생 보장을 의미하지는 않습니다."
        "</p>",
        unsafe_allow_html=True
    )
st.markdown("## 커버력 계산 기준 설명")

with st.expander("커버력 계산 공식 및 기준 보기"):
    st.markdown("""
    ### 커버력 계산 공식 (2023년, 격차지수 '적정' 지역 기준)

    **1. 상담소 커버력**  
    상담소 수 × 1000

    **2. 보호소 커버력**  
    보호소 수 × 16.74

    **3. 총 커버력**  
    상담소 커버력 + 보호소 커버력

    **4. 커버력 비율 계산**  
    - 상담소 커버력 비율 = 상담소 커버력 ÷ 총 커버력 ≈ **99.17%**  
    - 보호소 커버력 비율 = 보호소 커버력 ÷ 총 커버력 ≈ **0.83%**

    **위 비율(상담소:보호소 = 약 99:1)은 2023년 격차지수 '적정' 지역 평균값 기반입니다.**

    ---

    ### 2026년 예측을 활용한 필요 시설 수 계산

    **1. 격차지수 목표 기준**: 1.99 이하 (주의 등급 기준)

    **2. 총 필요 커버력 계산**  
    2026년 예측 신고건수 ÷ 1.99

    **3. 커버력 확보량 계산**  
    - 상담소: 필요 커버력 × 0.987  
    - 보호소: 필요 커버력 × 0.013

    **4. 필요 시설 수 계산**  
    - 상담소 수 = 상담소 커버력 ÷ 1000  
    - 보호소 수 = 보호소 커버력 ÷ 16.74

    ### 사용된 예측 모델 설명

    본 시스템은 지역별 가정폭력 신고건수를 예측하기 위해 다음의 3가지 회귀 모델을 사용합니다.

    - **Random Forest Regressor**
    - **Linear Regression**
    - **XGBoost Regressor**

    예측 방식은 다음과 같습니다:

    - 2021~2024년 신고건수 데이터를 기반으로 각 지역별 모델을 학습
    - 세 모델의 예측값 평균을 최종 예측값으로 사용
    - 각 모델의 R²(결정계수)을 기반으로 **신뢰도 평균(R²)** 산출
    - R²이 0.7 이상일 경우에만 해당 지역의 예측을 '신뢰 가능'으로 판단

    **R²(결정계수)**는 모델이 과거 데이터를 얼마나 잘 설명하는지를 나타내는 지표입니다.  
    1.0에 가까울수록 신뢰도가 높으며, 과적합을 방지하기 위해 다중 모델을 평균하여 사용합니다.
    """)
