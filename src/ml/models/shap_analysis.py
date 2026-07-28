"""
shap_analysis.py
Day4 모델 해석 - SHAP(TreeExplainer) 기반 피처 기여도 분석

기존 train_*.py들은 model.feature_importances_(기본 split count)만 출력해
"몇 번 분할에 쓰였는지"는 알 수 있어도 "실제로 예측값에 얼마나/어느 방향으로
기여했는지"는 알 수 없었다. 이 스크립트는 이미 저장된 모델 bundle(data/ml/models/*.pkl)과
feature_table_final.parquet을 그대로 불러와 SHAP value만 추가로 계산하는 신규
분석 스크립트다 — 학습을 다시 하지 않으므로 train_base_model.py, train_center_base_models.py,
train_tweedie_model.py, common.py는 전혀 수정하지 않는다.

핵심 설계:
    1) 평가 데이터: evaluate_pipeline.py와 동일하게 학습에 쓰이지 않은 홀드아웃만 사용한다
       (A는 split=='test', B는 split=='pool'). 센터별 분리 모델(A전용/B전용)은 각자의
       홀드아웃만, A+B 통합 모델(Base/Tweedie)은 두 홀드아웃을 합쳐서 사용한다.
    2) 표본 샘플링: TreeExplainer 자체는 빠르지만 홀드아웃 행 수가 많아
       SAMPLE_SIZE(기본 3000)로 무작위 샘플링 후 explain한다(random_state 고정으로 재현 가능).
    3) 조건부회귀기(reg_bundle, Base/A/B)는 target_h1>0인 행만으로 학습됐으므로
       그 대상행만 필터링해 explain한다. Tweedie는 0 포함 전체 행으로 학습됐으므로
       필터링하지 않는다(train_tweedie_model.py 설계 그대로).
    4) 날씨 피처 컬럼명은 신/구 버전을 모두 candidate로 두어, feature_table_final.parquet을
       아직 재생성하지 않은 상태(강수량_상위5%_flag/count)에서도, 재생성한 이후 상태
       (강수량_호우_flag/count)에서도 그대로 동작한다.
    5) 출력: 모델별 mean(|SHAP|) 기준 전체 Top15를 콘솔에 출력하고, 날씨 피처만 따로
       순위를 짚어준다. 전체 순위표는 MODEL_DIR/shap/shap_<모델명>.csv로 저장해 오프라인
       재검토가 가능하게 한다.
"""

import numpy as np
import pandas as pd
import shap

from common import (
    A_BASE_CLS_PATH,
    A_BASE_REG_PATH,
    B_BASE_CLS_PATH,
    B_BASE_REG_PATH,
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    load_model_bundle,
    prepare_X,
)

TWEEDIE_MODEL_PATH = MODEL_DIR / "tweedie_reg.pkl"
SHAP_OUT_DIR = MODEL_DIR / "shap"

SAMPLE_SIZE = 3000
RANDOM_STATE = 42
TOP_N = 15

# 강수량 피처는 rename(강수량_상위5%_* -> 강수량_호우_*) 전/후 데이터 모두에서 동작하도록
# 후보 이름을 순서대로 시도한다.
WEATHER_COL_CANDIDATES = {
    "평균온도": ["평균온도"],
    "총강수량": ["총강수량"],
    "temp_x_precip": ["temp_x_precip"],
    "강수량_호우_flag": ["강수량_호우_flag", "강수량_상위5%_flag"],
    "강수량_호우_count": ["강수량_호우_count", "강수량_상위5%_count"],
    "공휴일_D-1": ["공휴일_D-1"],
    "공휴일_D0": ["공휴일_D0"],
    "공휴일_D+1": ["공휴일_D+1"],
    "B센터_명절휴무": ["B센터_명절휴무"],
    "center_is_B": ["center_is_B"],
}

# name: 리포트용 라벨 / path: bundle 경로 / kind: cls(분류) | reg(조건부회귀, target>0만) |
# reg_raw(Tweedie, 0 포함 전체) / center: 홀드아웃 데이터 소스(A=A_test, B=B_pool, AB=둘 다)
MODEL_SPECS = [
    {"name": "Base(A+B 통합)_1단계분류",   "path": BASE_CLS_MODEL_PATH, "kind": "cls",     "center": "AB"},
    {"name": "Base(A+B 통합)_2단계조건부회귀", "path": BASE_MODEL_PATH,     "kind": "reg",     "center": "AB"},
    {"name": "A전용_분류",                "path": A_BASE_CLS_PATH,     "kind": "cls",     "center": "A"},
    {"name": "A전용_조건부회귀",           "path": A_BASE_REG_PATH,     "kind": "reg",     "center": "A"},
    {"name": "B전용_분류",                "path": B_BASE_CLS_PATH,     "kind": "cls",     "center": "B"},
    {"name": "B전용_조건부회귀",           "path": B_BASE_REG_PATH,     "kind": "reg",     "center": "B"},
    {"name": "Tweedie(A+B 통합)",         "path": TWEEDIE_MODEL_PATH,  "kind": "reg_raw", "center": "AB"},
]


def get_holdout_df(df: pd.DataFrame, center: str) -> pd.DataFrame:
    a_test = df[(df["center_id"] == "A") & (df["split"] == "test")]
    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")]
    if center == "A":
        out = a_test
    elif center == "B":
        out = b_pool
    else:
        out = pd.concat([a_test, b_pool], ignore_index=True)
    return out[out[TARGET_COL].notna()].copy()


def compute_mean_abs_shap(model, X: pd.DataFrame) -> np.ndarray:
    """이진분류/회귀 어느 쪽이든 (n_rows, n_features) 형태의 mean(|SHAP|) 벡터로 통일해 반환.
    shap 버전에 따라 shap_values()가 리스트([neg, pos])나 3차원 배열((n, feat, class))로
    올 수 있어(이진분류 특히), 양성 클래스 기준으로 항상 2차원으로 맞춘다."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return np.abs(sv).mean(axis=0)


def resolve_weather_col(label: str, feature_cols: list[str]) -> str | None:
    for candidate in WEATHER_COL_CANDIDATES[label]:
        if candidate in feature_cols:
            return candidate
    return None


def run_one_model(spec: dict, df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 80)
    print(f"[{spec['name']}] ({spec['path'].name})")

    bundle = load_model_bundle(spec["path"])
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    eval_df = get_holdout_df(df, spec["center"])
    if spec["kind"] == "reg":
        eval_df = eval_df[eval_df[TARGET_COL] > 0]
    print(f"  홀드아웃 대상행: {len(eval_df):,}행 (center={spec['center']}, kind={spec['kind']})")

    if len(eval_df) > SAMPLE_SIZE:
        eval_df = eval_df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"  SHAP 계산 표본: {len(eval_df):,}행")

    X = prepare_X(eval_df, feature_cols)
    mean_abs_shap = compute_mean_abs_shap(model, X)

    ranking = (
        pd.Series(mean_abs_shap, index=feature_cols)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .to_frame()
    )
    ranking["rank"] = range(1, len(ranking) + 1)

    print(f"  Top{TOP_N}:")
    print(ranking.head(TOP_N).to_string())

    print("  날씨 피처 순위(전체 피처 수 대비):")
    for label in WEATHER_COL_CANDIDATES:
        col = resolve_weather_col(label, feature_cols)
        if col is None:
            print(f"    {label}: (이 모델 feature_cols에 없음)")
            continue
        row = ranking.loc[col]
        print(f"    {label}({col}): {int(row['rank'])}/{len(ranking)}위, mean(|SHAP|)={row['mean_abs_shap']:.5f}")

    SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHAP_OUT_DIR / f"shap_{spec['name']}.csv"
    ranking.to_csv(out_path, encoding="utf-8-sig")
    print(f"  전체 순위표 저장 -> {out_path}")

    ranking["model"] = spec["name"]
    return ranking


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    print(f"[SHAP 분석] feature_table_final.parquet 로드 완료 (전체 {len(df):,}행)")

    completed = []
    for spec in MODEL_SPECS:
        try:
            ranking = run_one_model(spec, df)
            completed.append((spec, ranking))
        except FileNotFoundError:
            print(f"  !! {spec['path']} 없음 — 해당 모델은 건너뜀(먼저 학습 스크립트를 실행해야 함)")

    if not completed:
        print("분석할 모델이 하나도 없습니다. train_base_model.py / train_center_base_models.py / "
              "train_tweedie_model.py를 먼저 실행하세요.")
        return

    print()
    print("=" * 80)
    print("[요약] 모델별 날씨 피처 순위 한눈에 보기")
    summary_rows = []
    for spec, ranking in completed:
        row = {"model": spec["name"], "n_features": len(ranking)}
        for label in WEATHER_COL_CANDIDATES:
            col = resolve_weather_col(label, ranking.index.tolist())
            row[label] = int(ranking.loc[col, "rank"]) if col is not None else None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
