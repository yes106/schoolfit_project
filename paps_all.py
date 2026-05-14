import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0. 기본 설정
# -----------------------------
DATA_DIR = "data"
OUTPUT_DIR = "output"


os.makedirs(OUTPUT_DIR, exist_ok=True)

INCLUDE_HIGH_SCHOOL = False
KEEP_GENDER = True

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# -----------------------------
# 1. 체력 영역 정의
# -----------------------------
metric_groups = {
    "심폐지구력": [
        "왕복오래달리기(회)",
        "오래달리기걷기(초)",
        "스텝검사(PEI)"
    ],
    "유연성": [
        "앉아윗몸앞으로굽히기(cm)",
        "종합유연성(점)"
    ],
    "근력": [
        "(무릎대고)팔굽혀펴기(회)",
        "윗몸말아올리기(회)",
        "악력(kg)"
    ],
    "순발력": [
        "50m달리기(초)",
        "제자리멀리뛰기(cm)"
    ]
}

reverse_metrics = {
    "오래달리기걷기(초)",
    "50m달리기(초)"
}

target_groups = [
    "심폐지구력",
    "근력"
]

all_metrics = [m for v in metric_groups.values() for m in v]
all_metrics.append("BMI(kg/㎡)")

# -----------------------------
# 2. 파일 찾기
# -----------------------------
patterns = [
    os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(중)_서울특별시교육청*.csv"),
    os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(중)_서울특별시교육청*.xlsx"),
]

if INCLUDE_HIGH_SCHOOL:
    patterns += [
        os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(고)_서울특별시교육청*.csv"),
        os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(고)_서울특별시교육청*.xlsx"),
    ]

files = []
for p in patterns:
    files.extend(glob.glob(p))

files = sorted(files)

if not files:
    raise FileNotFoundError("PAPS 파일을 찾지 못했습니다. DATA_DIR 경로를 확인하세요.")

print("파일 수:", len(files))

# -----------------------------
# 3. 함수 정의
# -----------------------------
def read_paps_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        for enc in ["utf-8", "cp949", "euc-kr"]:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except Exception:
                pass
        raise ValueError(f"CSV 파일 인코딩을 읽을 수 없습니다: {filepath}")

    return pd.read_excel(filepath)


def extract_year(filename):
    m = re.search(r"(\d{4})년도", filename)
    return int(m.group(1)) if m else None


def extract_gu(region_value):
    if pd.isna(region_value):
        return None

    text = str(region_value)

    m = re.search(r"서울특별시\s+(.+?구)", text)
    if m:
        return m.group(1)

    m = re.search(r"(.+?구)", text)
    if m:
        return m.group(1)

    return None


def normalize_gender(x):
    x = str(x).strip()

    if x in ["남", "남자", "남학생"]:
        return "남"
    if x in ["여", "여자", "여학생"]:
        return "여"

    return None


def find_grade_count_cols(columns):
    cols = [c for c in columns if "인원" in str(c)]
    return cols[:5]


def safe_z(series):
    std = series.std(ddof=0)

    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)

    return (series - series.mean()) / std


def add_covid_lines():
    plt.axvline(
        x=2020,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="COVID-19 시작"
    )
    plt.axvline(
        x=2023,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="COVID-19 종료"
    )


def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150)
    plt.show()
    plt.close()

# -----------------------------
# 4. 전체 데이터 통합
# -----------------------------
dfs = []

for f in files:
    df = read_paps_file(f)
    year = extract_year(os.path.basename(f))

    if year is None:
        print(f"[주의] 연도 추출 실패: {f}")
        continue

    keep_cols = ["지역", "성별"]

    for m in all_metrics:
        if m in df.columns:
            keep_cols.append(m)

    current_grade_cols = find_grade_count_cols(df.columns)

    if len(current_grade_cols) < 5:
        print(f"[주의] 인원 열 5개를 다 찾지 못함: {os.path.basename(f)} -> {current_grade_cols}")

    for c in current_grade_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    temp = df[keep_cols].copy()

    temp["연도"] = year
    temp["자치구"] = temp["지역"].apply(extract_gu)
    temp["성별"] = temp["성별"].apply(normalize_gender)

    for col in all_metrics:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    for col in current_grade_cols:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    temp = temp[temp["자치구"].notna()]
    temp = temp[temp["성별"].isin(["남", "여"])]

    dfs.append(temp)

if not dfs:
    raise ValueError("읽을 수 있는 데이터가 없습니다. 파일명/경로를 확인하세요.")

raw_df = pd.concat(dfs, ignore_index=True)

print("원본 크기:", raw_df.shape)

grade_count_cols = find_grade_count_cols(raw_df.columns)
print("등급 인원 열:", grade_count_cols)

# -----------------------------
# 5. 자치구·연도·성별 평균 생성
# -----------------------------
group_cols = ["자치구", "연도"]

if KEEP_GENDER:
    group_cols.append("성별")

mean_df = raw_df.groupby(group_cols, as_index=False).mean(numeric_only=True)

if "BMI(kg/㎡)" in mean_df.columns:
    mean_df = mean_df.rename(columns={"BMI(kg/㎡)": "평균BMI"})
else:
    mean_df["평균BMI"] = np.nan

# -----------------------------
# 6. 등급 비율 계산
# -----------------------------
if len(grade_count_cols) >= 5:
    grade_sum_df = raw_df.groupby(group_cols, as_index=False)[grade_count_cols].sum()

    grade_sum_df["전체등급인원"] = grade_sum_df[grade_count_cols].sum(axis=1)
    grade_sum_df["1_2등급인원"] = grade_sum_df[grade_count_cols[0]] + grade_sum_df[grade_count_cols[1]]
    grade_sum_df["3등급인원"] = grade_sum_df[grade_count_cols[2]]
    grade_sum_df["4_5등급인원"] = grade_sum_df[grade_count_cols[3]] + grade_sum_df[grade_count_cols[4]]

    grade_sum_df["1~2등급비율"] = np.where(
        grade_sum_df["전체등급인원"] > 0,
        grade_sum_df["1_2등급인원"] / grade_sum_df["전체등급인원"] * 100,
        np.nan
    )

    grade_sum_df["3등급비율"] = np.where(
        grade_sum_df["전체등급인원"] > 0,
        grade_sum_df["3등급인원"] / grade_sum_df["전체등급인원"] * 100,
        np.nan
    )

    grade_sum_df["4~5등급비율"] = np.where(
        grade_sum_df["전체등급인원"] > 0,
        grade_sum_df["4_5등급인원"] / grade_sum_df["전체등급인원"] * 100,
        np.nan
    )

    mean_df = pd.merge(
        mean_df,
        grade_sum_df[group_cols + ["1~2등급비율", "3등급비율", "4~5등급비율"]],
        on=group_cols,
        how="left"
    )

else:
    mean_df["1~2등급비율"] = np.nan
    mean_df["3등급비율"] = np.nan
    mean_df["4~5등급비율"] = np.nan

# -----------------------------
# 7. 통합 체력 점수 생성
# -----------------------------
result_list = []

std_groups = ["연도"]

if KEEP_GENDER:
    std_groups.append("성별")

for _, sub in mean_df.groupby(std_groups):
    sub = sub.copy()

    for metric in all_metrics:
        if metric not in sub.columns:
            continue

        z = safe_z(sub[metric])

        if metric in reverse_metrics:
            z = -z

        sub[metric + "_z"] = z

    for group in target_groups:
        usable = [m for m in metric_groups[group] if m in sub.columns]
        zcols = [m + "_z" for m in usable if m + "_z" in sub.columns]

        sub[group] = sub[zcols].mean(axis=1) if zcols else np.nan

    keep_cols = ["자치구", "연도"]

    if KEEP_GENDER:
        keep_cols.append("성별")

    keep_cols += target_groups
    keep_cols += ["평균BMI", "1~2등급비율", "3등급비율", "4~5등급비율"]

    result_list.append(sub[keep_cols])

final_df = pd.concat(result_list, ignore_index=True)

sort_cols = ["자치구", "연도"]

if KEEP_GENDER:
    sort_cols.append("성별")

final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

# -----------------------------
# 8. 저장
# -----------------------------
school_tag = "중고등포함" if INCLUDE_HIGH_SCHOOL else "중학생"
gender_tag = "성별포함" if KEEP_GENDER else "성별통합"

file_name = f"{school_tag}_자치구_연도별_4개영역_통합점수_{gender_tag}_BMI_1_2_3_4_5등급비율포함.csv"
csv_save_path = os.path.join(OUTPUT_DIR, file_name)

final_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

print("\n저장 완료:", csv_save_path)

# -----------------------------
# 9. 그래프용 데이터 생성
# -----------------------------
if KEEP_GENDER is False:
    raise ValueError("그래프는 성별포함 데이터 기준입니다. KEEP_GENDER=True로 실행하세요.")

plot_df = final_df[final_df["연도"].between(2016, 2025)].copy()

year_gender_df = (
    plot_df
    .groupby(["연도", "성별"], as_index=False)[
        target_groups + ["평균BMI", "1~2등급비율", "3등급비율", "4~5등급비율"]
    ]
    .mean()
)

male_df = year_gender_df[year_gender_df["성별"] == "남"].copy()
female_df = year_gender_df[year_gender_df["성별"] == "여"].copy()

# -----------------------------
# 10. 그래프 함수
# -----------------------------
def plot_fitness_score(df, gender_label):
    plt.figure(figsize=(10, 6))

    for col in target_groups:
        plt.plot(df["연도"], df[col], marker="o", label=col)

    add_covid_lines()

    plt.title(f"2016~2025 {gender_label} 체력점수 추이")
    plt.xlabel("연도")
    plt.ylabel("체력 통합점수")
    plt.xticks(sorted(df["연도"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_show(f"2016_2025_{gender_label}_체력점수추이.png")


def plot_bmi(df, gender_label):
    plt.figure(figsize=(8, 5))

    plt.plot(df["연도"], df["평균BMI"], marker="o", label="평균BMI")

    add_covid_lines()

    plt.title(f"2016~2025 {gender_label} 평균 BMI 추이")
    plt.xlabel("연도")
    plt.ylabel("평균 BMI")
    plt.xticks(sorted(df["연도"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_show(f"2016_2025_{gender_label}_BMI추이.png")


def plot_grade_ratio(df, gender_label):
    plt.figure(figsize=(9, 5))

    grade_cols = ["1~2등급비율", "3등급비율", "4~5등급비율"]

    for col in grade_cols:
        plt.plot(df["연도"], df[col], marker="o", label=col)

    add_covid_lines()

    plt.title(f"2016~2025 {gender_label} 등급 비율 추이")
    plt.xlabel("연도")
    plt.ylabel("비율(%)")
    plt.xticks(sorted(df["연도"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_and_show(f"2016_2025_{gender_label}_등급비율추이.png")

# -----------------------------
# 11. 최종 그래프 실행
# -----------------------------
plot_fitness_score(male_df, "남학생")
plot_fitness_score(female_df, "여학생")

plot_bmi(male_df, "남학생")
plot_bmi(female_df, "여학생")

plot_grade_ratio(male_df, "남학생")
plot_grade_ratio(female_df, "여학생")

print("\n그래프 저장 완료:", OUTPUT_DIR)

print("\n평균 데이터 미리보기")
print(mean_df.head())

print("\n최종 데이터 미리보기")
print(final_df.head())