import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# -----------------------------
# 기본 설정
# -----------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = "data"

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
all_data = []

for year in range(2016, 2026):
    file_path = DATA_DIR / f"kyrbs{year}.sav"

    temp = pd.read_spss(file_path)
    temp["year"] = year

    # 서울 중학생만
    temp = temp[
        (temp["CITY"] == "서울") &
        (temp["MH"] == "중학교")
    ].copy()

    cols = [
        "year", "SEX", "GRADE",
        "M_SLP_HR", "M_SLP_MM",
        "M_WK_HR", "M_WK_MM",
        "INT_SPWD_TM", "INT_SPWK_TM",
        "M_STR", "PA_TOT", "PA_MSC"
    ]

    temp = temp[[c for c in cols if c in temp.columns]].copy()

    all_data.append(temp)

df = pd.concat(all_data, ignore_index=True)

print("10년치 서울 중학생 데이터 크기:", df.shape)

# -----------------------------
# 2. 숫자형 변환
# -----------------------------
num_cols = [
    "M_SLP_HR", "M_SLP_MM",
    "M_WK_HR", "M_WK_MM",
    "INT_SPWD_TM", "INT_SPWK_TM",
    "M_STR", "PA_TOT", "PA_MSC"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 3. 파생변수 생성
# -----------------------------
bed = df["M_SLP_HR"] + df["M_SLP_MM"] / 60
wake = df["M_WK_HR"] + df["M_WK_MM"] / 60

df["sleep_hours"] = np.where(
    wake < bed,
    24 - bed + wake,
    wake - bed
)

df["smartphone_avg"] = (
    df["INT_SPWD_TM"] * 5 +
    df["INT_SPWK_TM"] * 2
) / 7

df["stress_raw"] = df["M_STR"]

# -----------------------------
# 4. 성별 / 학년 정리
# -----------------------------
def normalize_sex(x):
    s = str(x).strip()

    if s in ["1", "1.0"] or "남" in s:
        return "남자"

    if s in ["2", "2.0"] or "여" in s:
        return "여자"

    return np.nan


def normalize_grade(x):
    s = str(x).strip()

    if s in ["1", "1.0", "1학년", "중1"]:
        return "1학년"

    if s in ["2", "2.0", "2학년", "중2"]:
        return "2학년"

    if s in ["3", "3.0", "3학년", "중3"]:
        return "3학년"

    return np.nan


df["성별"] = df["SEX"].apply(normalize_sex)
df["학년"] = df["GRADE"].apply(normalize_grade)

# -----------------------------
# 5. 분석용 데이터
# -----------------------------
analysis_cols = [
    "year",
    "PA_TOT",
    "PA_MSC",
    "sleep_hours",
    "smartphone_avg",
    "stress_raw",
    "성별",
    "학년"
]

analysis_df = df[analysis_cols].dropna().copy()

print("분석용 데이터 크기:", analysis_df.shape)

# -----------------------------
# 6. 상관계수
# -----------------------------
corr_vars = [
    "PA_TOT",
    "PA_MSC",
    "sleep_hours",
    "smartphone_avg",
    "stress_raw"
]

labels = [
    "신체활동",
    "근력운동",
    "수면시간",
    "스마트폰",
    "스트레스"
]

corr_df = analysis_df[corr_vars].corr()

print("\n[10년치 전체 상관계수]")
print(corr_df)

# -----------------------------
# 7. 회귀분석
# -----------------------------
model_pa_tot = smf.ols(
    "PA_TOT ~ sleep_hours + smartphone_avg + stress_raw + C(성별) + C(학년) + C(year)",
    data=analysis_df
).fit()

print("\n[회귀분석: PA_TOT]")
print(model_pa_tot.summary())

model_pa_msc = smf.ols(
    "PA_MSC ~ sleep_hours + smartphone_avg + stress_raw + C(성별) + C(학년) + C(year)",
    data=analysis_df
).fit()

print("\n[회귀분석: PA_MSC]")
print(model_pa_msc.summary())

# 남/여 분리 회귀
for sex in ["남자", "여자"]:
    sex_df = analysis_df[
        analysis_df["성별"] == sex
    ].copy()

    model_sex = smf.ols(
        "PA_TOT ~ sleep_hours + smartphone_avg + stress_raw + C(학년) + C(year)",
        data=sex_df
    ).fit()

    print(f"\n[{sex} 회귀분석: PA_TOT]")
    print(model_sex.summary())

# =========================================================
# 그래프 함수
# =========================================================
def make_scatter_with_reg(
    data,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    ax=None,
    color=None
):
    if ax is None:
        ax = plt.gca()

    temp = data[[x_col, y_col]].dropna().copy()

    if len(temp) > 3000:
        temp = temp.sample(n=3000, random_state=42)

    ax.scatter(
        temp[x_col],
        temp[y_col],
        alpha=0.18,
        s=14,
        color=color,
        edgecolors="none"
    )

    coef = np.polyfit(temp[x_col], temp[y_col], 1)

    x = np.linspace(
        temp[x_col].min(),
        temp[x_col].max(),
        100
    )

    y = np.poly1d(coef)(x)

    ax.plot(x, y, color="black", linewidth=2.2)

    r = temp[[x_col, y_col]].corr().iloc[0, 1]

    ax.text(
        0.05,
        0.92,
        f"r = {r:.3f}\nn = {len(temp):,}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            boxstyle="round,pad=0.4",
            alpha=0.9
        )
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

# =========================================================
# 그래프 1. 상관관계 히트맵
# =========================================================
heatmap_df = corr_df.copy()
heatmap_df.index = labels
heatmap_df.columns = labels

mask = np.triu(
    np.ones_like(heatmap_df, dtype=bool),
    k=1
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    heatmap_df,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "상관계수"},
    annot_kws={"size": 12}
)

plt.title(
    "10년 전체 주요 변수 간 상관관계",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# =========================================================
# 그래프 2. 전체 수면시간 vs 신체활동
# =========================================================
plt.figure(figsize=(7, 5))

make_scatter_with_reg(
    analysis_df,
    "sleep_hours",
    "PA_TOT",
    "수면시간과 신체활동의 관계",
    "수면시간",
    "신체활동일수(PA_TOT)"
)

plt.tight_layout()
plt.show()

# =========================================================
# 그래프 3. 남/여 분리
# =========================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    sharey=True
)

sex_info = {
    "남자": {
        "ax": axes[0],
        "color": "#4C78A8"
    },
    "여자": {
        "ax": axes[1],
        "color": "#F58518"
    }
}

for sex, info in sex_info.items():
    temp = analysis_df[
        analysis_df["성별"] == sex
    ].copy()

    make_scatter_with_reg(
        temp,
        "sleep_hours",
        "PA_TOT",
        sex,
        "수면시간",
        "신체활동일수(PA_TOT)",
        ax=info["ax"],
        color=info["color"]
    )

fig.suptitle(
    "성별에 따른 수면시간과 신체활동의 관계",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()