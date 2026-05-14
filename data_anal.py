import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0. 기본 설정
# -----------------------------

DATA_DIR = "output"

FILE_NAME = "중학생_자치구_연도별_4개영역_통합점수_성별포함_BMI_1_2_3_4_5등급비율포함.csv"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")

print("[원본 데이터 확인]")
print(df.head())
print("\n[컬럼 목록]")
print(df.columns.tolist())

# -----------------------------
# 2. 필요한 컬럼 확인
# -----------------------------
required_cols = [
    "자치구", "연도", "성별",
    "심폐지구력", "근력", "평균BMI",
    "1~2등급비율", "3등급비율", "4~5등급비율"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"필수 열이 없습니다: {missing_cols}")

# -----------------------------
# 3. 숫자형 변환 및 결측 제거
# -----------------------------
numeric_cols = [
    "연도", "심폐지구력", "근력", "평균BMI",
    "1~2등급비율", "3등급비율", "4~5등급비율"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "자치구", "연도", "성별",
        "1~2등급비율", "3등급비율", "4~5등급비율"
    ]
).copy()

df["연도"] = df["연도"].astype(int)

# -----------------------------
# 4. 자치구-연도-성별 기준 정리
# -----------------------------
analysis_df = (
    df.groupby(["자치구", "연도", "성별"], as_index=False)[
        [
            "심폐지구력", "근력", "평균BMI",
            "1~2등급비율", "3등급비율", "4~5등급비율"
        ]
    ]
    .mean()
)

# -----------------------------
# 5. 양극화 지표 추가
# -----------------------------
analysis_df["상하위격차"] = (
    analysis_df["4~5등급비율"] - analysis_df["1~2등급비율"]
)

analysis_df["극단비율합"] = (
    analysis_df["1~2등급비율"] + analysis_df["4~5등급비율"]
)

analysis_df["하위권쏠림지수"] = (
    analysis_df["4~5등급비율"] / (analysis_df["1~2등급비율"] + 1e-6)
)

print("\n[분석용 데이터]")
print(analysis_df.head())

# -----------------------------
# 6. 성별 평균 → 자치구-연도 단위 통합
# -----------------------------
gu_year_df = (
    analysis_df
    .groupby(["자치구", "연도"], as_index=False)["4~5등급비율"]
    .mean()
)

print("\n[자치구-연도별 4~5등급 비율 데이터]")
print(gu_year_df.head())

# -----------------------------
# 7. 최근 3년 4~5등급 비율 순위
# -----------------------------
recent_df = gu_year_df[gu_year_df["연도"].between(2023, 2025)].copy()

recent_avg_df = (
    recent_df
    .groupby("자치구", as_index=False)["4~5등급비율"]
    .mean()
    .rename(columns={"4~5등급비율": "최근3년평균_4~5등급비율"})
    .sort_values("최근3년평균_4~5등급비율", ascending=False)
    .reset_index(drop=True)
)

print("\n==============================")
print("[최근 3년(2023~2025) 4~5등급 비율 높은 자치구 순위]")
print("==============================")

for i, row in recent_avg_df.iterrows():
    print(f"{i+1:>2}. {row['자치구']:<6}  {row['최근3년평균_4~5등급비율']:.2f}%")

# -----------------------------
# 8. 코로나 전후 4~5등급 비율 변화
# -----------------------------
before_df = gu_year_df[gu_year_df["연도"].between(2016, 2019)].copy()
after_df = gu_year_df[gu_year_df["연도"].between(2023, 2025)].copy()

before_avg = (
    before_df
    .groupby("자치구", as_index=False)["4~5등급비율"]
    .mean()
    .rename(columns={"4~5등급비율": "코로나전평균"})
)

after_avg = (
    after_df
    .groupby("자치구", as_index=False)["4~5등급비율"]
    .mean()
    .rename(columns={"4~5등급비율": "코로나후평균"})
)

change_df = pd.merge(before_avg, after_avg, on="자치구", how="inner")
change_df["변화폭"] = change_df["코로나후평균"] - change_df["코로나전평균"]
change_df["절대변화폭"] = change_df["변화폭"].abs()

change_rank_df = (
    change_df
    .sort_values("변화폭", ascending=False)
    .reset_index(drop=True)
)

print("\n============================================")
print("[코로나 전후 4~5등급 비율 변화폭이 큰 자치구 순위]")
print("  코로나후평균 - 코로나전평균")
print("============================================")

for i, row in change_rank_df.iterrows():
    sign = "+" if row["변화폭"] >= 0 else ""
    print(
        f"{i+1:>2}. {row['자치구']:<6}  "
        f"전:{row['코로나전평균']:.2f}%  "
        f"후:{row['코로나후평균']:.2f}%  "
        f"변화:{sign}{row['변화폭']:.2f}%p"
    )

# -----------------------------
# 9. 코로나 전후 양극화 비교 함수
# -----------------------------
def make_compare_table(data, gender_label="전체"):
    if gender_label != "전체":
        data = data[data["성별"] == gender_label].copy()

    if data.empty:
        print(f"\n[{gender_label}] 데이터가 없습니다.")
        return pd.DataFrame()

    pre = data[data["연도"].between(2016, 2019)]
    post = data[data["연도"].between(2023, 2024)]

    if pre.empty or post.empty:
        print(f"\n[{gender_label}] 코로나 전후 비교 데이터가 부족합니다.")
        return pd.DataFrame()

    value_cols = [
        "심폐지구력", "근력", "평균BMI",
        "1~2등급비율", "3등급비율", "4~5등급비율",
        "상하위격차", "극단비율합", "하위권쏠림지수"
    ]

    pre_avg = pre.groupby("자치구")[value_cols].mean()
    post_avg = post.groupby("자치구")[value_cols].mean()

    compare = pre_avg.join(
        post_avg,
        lsuffix="_코로나전",
        rsuffix="_코로나후"
    ).reset_index()

    for col in value_cols:
        compare[f"{col}_변화량"] = (
            compare[f"{col}_코로나후"] - compare[f"{col}_코로나전"]
        )

    compare = (
        compare
        .sort_values("4~5등급비율_변화량", ascending=False)
        .reset_index(drop=True)
    )

    return compare

# -----------------------------
# 10. 전체 / 남 / 여 비교표 생성
# -----------------------------
compare_all = make_compare_table(analysis_df, "전체")
compare_male = make_compare_table(analysis_df, "남")
compare_female = make_compare_table(analysis_df, "여")

print("\n[전체] 코로나 이후 변화")
print(compare_all.head(10))

print("\n[남] 코로나 이후 변화")
print(compare_male.head(10))

print("\n[여] 코로나 이후 변화")
print(compare_female.head(10))

# -----------------------------
# 11. 그래프 함수
# -----------------------------
def save_and_show(filename):
    save_path = os.path.join(DATA_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[저장 완료] {save_path}")


def plot_recent_avg(recent_avg_df):
    plt.figure(figsize=(12, 8))

    plot_df = recent_avg_df.sort_values(
        "최근3년평균_4~5등급비율",
        ascending=True
    )

    plt.barh(
        plot_df["자치구"],
        plot_df["최근3년평균_4~5등급비율"]
    )

    for i, v in enumerate(plot_df["최근3년평균_4~5등급비율"]):
        plt.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9)

    plt.title("최근 3년(2023~2025) 자치구별 4~5등급 비율 평균")
    plt.xlabel("4~5등급 비율(%)")
    plt.ylabel("자치구")
    plt.grid(axis="x", alpha=0.3)

    save_and_show("최근3년_자치구별_4_5등급비율평균.png")


def plot_covid_change(change_df):
    plot_change_df = (
        change_df
        .sort_values("변화폭", ascending=True)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(12, 9))

    y_pos = np.arange(len(plot_change_df))

    for i, row in plot_change_df.iterrows():
        plt.plot(
            [row["코로나전평균"], row["코로나후평균"]],
            [i, i],
            linewidth=2,
            alpha=0.8
        )

    plt.scatter(
        plot_change_df["코로나전평균"],
        y_pos,
        s=60,
        label="코로나 전 평균(2016~2019)"
    )

    plt.scatter(
        plot_change_df["코로나후평균"],
        y_pos,
        s=60,
        label="코로나 후 평균(2023~2025)"
    )

    for i, row in plot_change_df.iterrows():
        sign = "+" if row["변화폭"] >= 0 else ""
        x_max = max(row["코로나전평균"], row["코로나후평균"])

        plt.text(
            x_max + 0.3,
            i,
            f"{sign}{row['변화폭']:.1f}%p",
            va="center",
            fontsize=9
        )

    plt.yticks(y_pos, plot_change_df["자치구"])
    plt.xlabel("4~5등급 비율(%)")
    plt.ylabel("자치구")
    plt.title("코로나 전후 자치구별 4~5등급 비율 변화\n전: 2016~2019 / 후: 2023~2025")
    plt.grid(axis="x", alpha=0.3)
    plt.legend()

    save_and_show("코로나전후_자치구별_4_5등급비율_변화.png")

# -----------------------------
# 12. 그래프 실행
# -----------------------------
plot_recent_avg(recent_avg_df)
plot_covid_change(change_df)

# -----------------------------
# 13. 결과 저장
# -----------------------------
recent_rank_path = os.path.join(
    DATA_DIR,
    "최근3년_자치구별_4_5등급비율평균_순위.csv"
)

change_rank_path = os.path.join(
    DATA_DIR,
    "코로나전후_자치구별_4_5등급비율변화_순위.csv"
)

analysis_path = os.path.join(
    DATA_DIR,
    "중학생_자치구_연도별_성별_양극화분석_기본지표.csv"
)

compare_all_path = os.path.join(
    DATA_DIR,
    "중학생_자치구별_코로나전후_양극화비교_전체.csv"
)

compare_male_path = os.path.join(
    DATA_DIR,
    "중학생_자치구별_코로나전후_양극화비교_남.csv"
)

compare_female_path = os.path.join(
    DATA_DIR,
    "중학생_자치구별_코로나전후_양극화비교_여.csv"
)

recent_avg_df.to_csv(recent_rank_path, index=False, encoding="utf-8-sig")
change_rank_df.to_csv(change_rank_path, index=False, encoding="utf-8-sig")
analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
compare_all.to_csv(compare_all_path, index=False, encoding="utf-8-sig")
compare_male.to_csv(compare_male_path, index=False, encoding="utf-8-sig")
compare_female.to_csv(compare_female_path, index=False, encoding="utf-8-sig")

print("\n저장 완료")
print(recent_rank_path)
print(change_rank_path)
print(analysis_path)
print(compare_all_path)
print(compare_male_path)
print(compare_female_path)

print("\n분석 완료.")