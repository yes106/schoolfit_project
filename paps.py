import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

################################################
# 10년치 PAPS 남녀별 추이 분석 코드
################################################

# -----------------------------
# 0. 기본 설정
# -----------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. 체력 영역별 항목 정의
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
    "근력·근지구력": [
        "(무릎대고)팔굽혀펴기(회)",
        "윗몸말아올리기(회)",
        "악력(kg)"
    ],
    "순발력": [
        "50m달리기(초)",
        "제자리멀리뛰기(cm)"
    ],
    "신체구성": [
        "BMI(kg/㎡)",
        "체지방률(%fat)"
    ]
}

reverse_metrics = {
    "오래달리기걷기(초)",
    "50m달리기(초)",
    "BMI(kg/㎡)",
    "체지방률(%fat)"
}

all_metrics = [m for v in metric_groups.values() for m in v]

# -----------------------------
# 2. 파일 찾기
# -----------------------------
patterns = [
    os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(중)_서울특별시교육청*.csv"),
    os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(중)_서울특별시교육청*.xlsx"),
    os.path.join(DATA_DIR, "*학생의 체력 증진에 관한 사항(중)_서울특별시교육청*.xls"),
]

files = []
for p in patterns:
    files.extend(glob.glob(p))

files = sorted(files)

if not files:
    raise FileNotFoundError("중학생 PAPS 파일을 찾지 못했습니다. DATA_DIR 경로를 확인하세요.")

print("찾은 파일 수:", len(files))
for f in files:
    print("-", os.path.basename(f))

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
        raise ValueError(f"CSV 읽기 실패: {filepath}")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)

    raise ValueError(f"지원하지 않는 형식: {filepath}")


def extract_year(filename):
    m = re.search(r"(\d{4})년도", filename)
    return int(m.group(1)) if m else None


def extract_gu(region_value):
    if pd.isna(region_value):
        return None

    text = str(region_value).strip()

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


def safe_filename(name):
    return (
        name.replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("㎡", "m2")
        .replace("%", "percent")
    )

# -----------------------------
# 4. 전체 파일 통합
# -----------------------------
dfs = []

for f in files:
    df = read_paps_file(f)

    if "지역" not in df.columns or "성별" not in df.columns:
        print(f"[건너뜀] 기본 열 없음: {os.path.basename(f)}")
        continue

    year = extract_year(os.path.basename(f))

    keep_cols = ["지역", "성별"]
    keep_cols += [col for col in all_metrics if col in df.columns]

    temp = df[keep_cols].copy()
    temp["연도"] = year
    temp["자치구"] = temp["지역"].apply(extract_gu)
    temp["성별"] = temp["성별"].apply(normalize_gender)

    for col in all_metrics:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

    temp = temp[temp["자치구"].notna()]
    temp = temp[temp["성별"].isin(["남", "여"])]

    dfs.append(temp)

if not dfs:
    raise ValueError("읽어들인 데이터가 없습니다.")

raw_df = pd.concat(dfs, ignore_index=True)

print("\n통합 데이터 크기:", raw_df.shape)

available_metrics = [c for c in all_metrics if c in raw_df.columns]

# -----------------------------
# 5. 평균 데이터 생성
# -----------------------------
year_gender_mean_df = (
    raw_df
    .groupby(["연도", "성별"], as_index=False)[available_metrics]
    .mean()
)

recent_df = raw_df[(raw_df["연도"] >= 2022) & (raw_df["연도"] <= 2025)].copy()

district_gender_mean_df = (
    recent_df
    .groupby(["자치구", "성별"], as_index=False)[available_metrics]
    .mean()
)

# -----------------------------
# 6. 통합점수 계산
# -----------------------------
def make_group_score(df, group_base_cols):
    result_parts = []

    for gender in ["남", "여"]:
        sub = df[df["성별"] == gender].copy()

        sort_col = [c for c in group_base_cols if c != "성별"][0]
        sub = sub.sort_values(sort_col).reset_index(drop=True)

        for group_name, metrics in metric_groups.items():
            usable = [m for m in metrics if m in sub.columns]
            z_cols = []

            for col in usable:
                s = sub[col].astype(float)
                std = s.std(ddof=0)

                if pd.isna(std) or std == 0:
                    z = pd.Series([0] * len(s), index=s.index)
                else:
                    z = (s - s.mean()) / std

                if col in reverse_metrics:
                    z = -z

                z_cols.append(z.rename(col))

            if z_cols:
                z_df = pd.concat(z_cols, axis=1)
                sub[group_name] = z_df.mean(axis=1)

        keep_cols = group_base_cols + list(metric_groups.keys())
        result_parts.append(sub[keep_cols])

    return pd.concat(result_parts, ignore_index=True)


year_score_df = make_group_score(year_gender_mean_df, ["연도", "성별"])
district_score_df = make_group_score(district_gender_mean_df, ["자치구", "성별"])

year_score_df.to_csv(
    os.path.join(OUTPUT_DIR, "중학생_연도별_성별_5개영역_통합점수.csv"),
    index=False,
    encoding="utf-8-sig"
)

district_score_df.to_csv(
    os.path.join(OUTPUT_DIR, "중학생_최근3년_자치구별_성별_5개영역_통합점수.csv"),
    index=False,
    encoding="utf-8-sig"
)

# -----------------------------
# 7. 그래프 함수
# -----------------------------
def plot_all_in_one(year_score_df, district_score_df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    graph_info = [
        ("연도", year_score_df, "남", "연도별 5개 체력영역 통합점수 추이 (남)"),
        ("연도", year_score_df, "여", "연도별 5개 체력영역 통합점수 추이 (여)"),
        ("자치구", district_score_df, "남", "최근 3년 평균 자치구별 5개 체력영역 통합점수 (남)"),
        ("자치구", district_score_df, "여", "최근 3년 평균 자치구별 5개 체력영역 통합점수 (여)")
    ]

    for ax, (x_col, df, gender, title) in zip(axes, graph_info):
        sub = df[df["성별"] == gender].copy().sort_values(x_col)

        for group_name in metric_groups.keys():
            ax.plot(sub[x_col], sub[group_name], marker="o", linewidth=2, label=group_name)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(x_col)
        ax.set_ylabel("표준화 통합점수")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        if x_col == "자치구":
            ax.tick_params(axis="x", rotation=90)
        else:
            ax.set_xticks(sub[x_col].unique())

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4분할_전체체력그래프.png"), dpi=200)
    plt.show()


def plot_4group_year_trends(year_score_df):
    selected_groups = ["심폐지구력", "유연성", "근력·근지구력", "순발력"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, group_name in zip(axes, selected_groups):
        for gender in ["남", "여"]:
            sub = year_score_df[year_score_df["성별"] == gender].copy().sort_values("연도")
            ax.plot(sub["연도"], sub[group_name], marker="o", linewidth=2, label=gender)

        ax.set_title(f"{group_name} 10년 추이", fontsize=13)
        ax.set_xlabel("연도")
        ax.set_ylabel("표준화 통합점수")
        ax.set_xticks(sub["연도"].unique())
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4분할_주요체력영역_10년추이.png"), dpi=200)
    plt.show()


def calculate_grade_average_from_counts(df):
    grade_cols = ["인원", "인원.1", "인원.2", "인원.3", "인원.4"]
    weights = [1, 2, 3, 4, 5]

    total_students = df[grade_cols].sum(axis=1)
    weighted_sum = sum(df[col] * w for col, w in zip(grade_cols, weights))

    return weighted_sum / total_students


def make_paps_grade_trend(files):
    grade_dfs = []

    for f in files:
        df = read_paps_file(f)
        year = extract_year(os.path.basename(f))

        needed = ["성별", "인원", "인원.1", "인원.2", "인원.3", "인원.4"]

        if not all(col in df.columns for col in needed):
            print(f"[건너뜀] 등급 인원 컬럼 없음: {os.path.basename(f)}")
            continue

        temp = df[needed].copy()
        temp["연도"] = year
        temp["성별"] = temp["성별"].apply(normalize_gender)

        for col in needed[1:]:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

        temp = temp[temp["성별"].isin(["남", "여"])].copy()

        total = temp[["인원", "인원.1", "인원.2", "인원.3", "인원.4"]].sum(axis=1)

        temp["평균등급"] = calculate_grade_average_from_counts(temp)
        temp["4_5등급비율"] = (temp["인원.3"] + temp["인원.4"]) / total

        grade_dfs.append(temp)

    if not grade_dfs:
        return None

    all_grade_df = pd.concat(grade_dfs, ignore_index=True)

    grade_trend_df = (
        all_grade_df
        .groupby(["연도", "성별"], as_index=False)
        .agg({
            "평균등급": "mean",
            "4_5등급비율": "mean"
        })
    )

    return grade_trend_df


def plot_paps_grade_trend(grade_trend_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for gender in ["남", "여"]:
        sub = grade_trend_df[grade_trend_df["성별"] == gender].sort_values("연도")
        axes[0].plot(sub["연도"], sub["평균등급"], marker="o", linewidth=2, label=gender)

    axes[0].set_title("전체 PAPS 평균등급 10년 추이")
    axes[0].set_xlabel("연도")
    axes[0].set_ylabel("평균등급 (낮을수록 좋음)")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for gender in ["남", "여"]:
        sub = grade_trend_df[grade_trend_df["성별"] == gender].sort_values("연도")
        axes[1].plot(sub["연도"], sub["4_5등급비율"] * 100, marker="o", linewidth=2, label=gender)

    axes[1].set_title("전체 PAPS 4~5등급 비율 10년 추이")
    axes[1].set_xlabel("연도")
    axes[1].set_ylabel("4~5등급 비율(%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "전체PAPS_평균등급_45등급비율_추이.png"), dpi=200)
    plt.show()


def plot_raw_metric_trend(raw_df, metric_name, better="high"):
    if metric_name not in raw_df.columns:
        print(f"[건너뜀] {metric_name} 컬럼 없음")
        return

    metric_df = raw_df.groupby(["연도", "성별"], as_index=False)[metric_name].mean()

    plt.figure(figsize=(10, 6))

    for gender in ["남", "여"]:
        sub = metric_df[metric_df["성별"] == gender].sort_values("연도")
        plt.plot(sub["연도"], sub[metric_name], marker="o", linewidth=2, label=gender)

    plt.title(f"{metric_name} 10년 추이")
    plt.xlabel("연도")

    if better == "high":
        plt.ylabel(f"{metric_name} 평균 (높을수록 좋음)")
    else:
        plt.ylabel(f"{metric_name} 평균 (낮을수록 좋음)")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{safe_filename(metric_name)}_10년추이.png"),
        dpi=200
    )
    plt.show()


def plot_covid_index_trend(raw_df, metric_name, baseline_years=(2018, 2019)):
    if metric_name not in raw_df.columns:
        print(f"[건너뜀] {metric_name} 컬럼 없음")
        return

    metric_df = raw_df.groupby(["연도", "성별"], as_index=False)[metric_name].mean()
    result = []

    for gender in ["남", "여"]:
        sub = metric_df[metric_df["성별"] == gender].copy().sort_values("연도")
        baseline = sub[sub["연도"].isin(baseline_years)][metric_name].mean()

        if pd.isna(baseline) or baseline == 0:
            print(f"[경고] {metric_name} {gender} 기준연도 평균이 없어 지수 계산 불가")
            continue

        sub["코로나전대비지수"] = (sub[metric_name] / baseline) * 100
        result.append(sub)

    if not result:
        return

    result_df = pd.concat(result, ignore_index=True)

    plt.figure(figsize=(10, 6))

    for gender in ["남", "여"]:
        sub = result_df[result_df["성별"] == gender].sort_values("연도")
        plt.plot(sub["연도"], sub["코로나전대비지수"], marker="o", linewidth=2, label=gender)

    plt.axhline(100, linestyle="--", linewidth=1.5)
    plt.title(f"{metric_name} 코로나 전(2018~2019) 대비 변화율 지수")
    plt.xlabel("연도")
    plt.ylabel("지수 (2018~2019 평균 = 100)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{safe_filename(metric_name)}_코로나전대비지수.png"),
        dpi=200
    )
    plt.show()


def compare_drop_by_gender(raw_df, metric, pre_years=(2018, 2019), post_years=(2022, 2023)):
    result = []

    for gender in ["남", "여"]:
        sub = raw_df[raw_df["성별"] == gender]

        pre_mean = sub[sub["연도"].isin(pre_years)][metric].mean()
        post_mean = sub[sub["연도"].isin(post_years)][metric].mean()
        drop_rate = (pre_mean - post_mean) / pre_mean * 100

        result.append({
            "성별": gender,
            "코로나전평균": pre_mean,
            "코로나후평균": post_mean,
            "감소율(%)": drop_rate
        })

    return pd.DataFrame(result)

# -----------------------------
# 8. 최종 실행
# -----------------------------
plot_all_in_one(year_score_df, district_score_df)
plot_4group_year_trends(year_score_df)

grade_trend_df = make_paps_grade_trend(files)
if grade_trend_df is not None:
    plot_paps_grade_trend(grade_trend_df)

main_metrics = [
    "왕복오래달리기(회)",
    "윗몸말아올리기(회)",
    "악력(kg)"
]

for metric in main_metrics:
    plot_raw_metric_trend(raw_df, metric, better="high")
    plot_covid_index_trend(raw_df, metric, baseline_years=(2018, 2019))

result_cardio = compare_drop_by_gender(
    raw_df,
    metric="왕복오래달리기(회)"
)

print("\n왕복오래달리기 코로나 전후 성별 감소율")
print(result_cardio)