"""
search_threshold_extended.py
통합 Hurdle Model의 hard 모드 threshold 전 구간(0.20~0.95) 탐색 — A/B 센터별로 각각

train_base_model.py는 0.20~0.50 구간만 탐색하는데, 그 구간에서 WAPE가 단조
개선되며 상한 0.50이 채택돼 진짜 최적점이 범위 밖에 있을 가능성이 있었다. 이
스크립트는 재학습 없이(이미 저장된 base_model_cls.pkl/base_model_reg.pkl 재사용,
threshold 탐색은 예측만 필요해 학습 비용이 들지 않음) 0.20~0.95 전 구간을 그때그때
새로 계산해 진짜 WAPE 최저점을 찾는다. (과거 특정 시점의 탐색 결과를 하드코딩해
재사용하지 않음 — 모델이 재학습될 때마다 이 스크립트를 다시 돌리면 항상 그 시점
모델 기준 최신 결과가 나온다.)

핵심 설계 — 센터별 threshold 분리:
    통합(A+B) Base Model은 원래 A val split 하나로만 threshold를 찾아 A/B 모두에
    동일하게 적용해왔다(B는 val split이 없어서 대안이 없었음). 그런데 B도
    train_center_base_models.py 등에서 이미 쓰고 있는 "B train 마지막 4주를 내부
    val로 떼어두는" 관례를 그대로 가져오면, 통합 모델의 예측값 자체는 그대로 두고
    B 자기 기준 최적 threshold를 별도로 찾을 수 있다. 재학습 전혀 불필요 — 추론
    시점에 cls_bundle["threshold_by_center"] = {"A":.., "B":..}로 저장해두면
    common.predict_base_hurdle()이 df["center_id"]별로 자동으로 다른 threshold를
    적용한다(하위호환: 이 키가 없는 옛 bundle은 그대로 스칼라 threshold로 동작).

핵심 설계 — 채택 기준은 WAPE 단독이 아니라 WAPE+α|Bias|:
    RMSE/MAE/WAPE/MASE 4개는 전부 "오차 크기"라는 같은 축을 잰다(부호를 버리고
    절대값/제곱만 보므로 과대추정과 과소추정을 구분 못 함). Bias(%)만 유일하게
    "오차 방향"(체계적 과소/과대추정 여부)을 잰다 — 애초에 Optuna 튜닝(tune_hyperparameters.py)
    을 시작한 이유가 "Hurdle Hard의 -35%~-47% 체계적 음수 Bias 완화"였는데, 정작
    threshold를 WAPE 단독으로 고르면 그 문제를 여기서 되살리게 된다(실측: WAPE
    최저점(th=0.85)은 Bias가 -38%/-42%까지 벌어짐). 그래서 Optuna와 동일하게
    BIAS_ALPHA=1.0을 곱한 combined_score = WAPE + BIAS_ALPHA*|Bias(%)|를 채택
    기준으로 쓴다. RMSE/MAE/MASE는 계속 표에는 출력해 모니터링하되 최적화 타겟에는
    넣지 않는다(WAPE와 같은 축이라 중복).
"""

import numpy as np
import pandas as pd
import joblib

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    WEEK_COL,
    apply_target_date_boundary_filter,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)

THRESHOLD_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)
B_INTERNAL_VAL_WEEKS = 4  # train_center_base_models.py / tune_hyperparameters.py와 동일 관례
BIAS_ALPHA = 1.0  # tune_hyperparameters.py의 Optuna 목적함수와 동일 가중치


def sweep(label: str, eval_df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict) -> pd.DataFrame:
    y_true = eval_df[TARGET_COL].to_numpy()
    naive_pred = eval_df["qty"].to_numpy()
    rows = []
    for th in THRESHOLD_GRID:
        y_pred = predict_base_hurdle(eval_df, cls_bundle, reg_bundle, mode="hard", threshold=th)
        m = compute_metrics(y_true, y_pred, naive_pred)
        combined_score = m["WAPE"] + BIAS_ALPHA * abs(m["Bias(%)"])
        rows.append({"threshold": float(th), **{k: round(v, 3) for k, v in m.items()},
                     "combined_score": round(combined_score, 3)})
        print(f"  [{label}] th={th:.2f} WAPE={m['WAPE']:.3f}% Bias={m['Bias(%)']:+.3f}% "
              f"combined={combined_score:.3f}")
    return pd.DataFrame(rows)


def get_b_internal_val(df: pd.DataFrame) -> pd.DataFrame:
    """B는 val split이 없어(train/pool만 존재), B train의 시간순 마지막 4주를 내부
    검증용으로 뗀다 — B_pool(진짜 최종 홀드아웃)은 threshold 탐색에 쓰지 않는다."""
    b_train = df[(df[CENTER_COL] == "B") & (df["split"] == "train")].copy()
    b_train = apply_target_date_boundary_filter(b_train, horizon_weeks=1)
    b_train = b_train[b_train[TARGET_COL].notna()]
    b_weeks = sorted(b_train[WEEK_COL].unique())
    val_weeks = b_weeks[-B_INTERNAL_VAL_WEEKS:]
    return b_train[b_train[WEEK_COL].isin(val_weeks)]


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    a_val_df = df[df["split"] == "val"].copy()
    a_val_df = a_val_df[a_val_df[TARGET_COL].notna()]
    b_val_df = get_b_internal_val(df)
    print(f"[Threshold 전 구간 탐색] A val(={len(a_val_df):,}행) / "
          f"B 내부val(마지막{B_INTERNAL_VAL_WEEKS}주, {len(b_val_df):,}행)")

    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    print(f"  현재 저장된 threshold: {cls_bundle.get('threshold')}  "
          f"threshold_by_center: {cls_bundle.get('threshold_by_center')}")

    print("=" * 80)
    print("[A센터] threshold 탐색")
    table_a = sweep("A", a_val_df, cls_bundle, reg_bundle)
    print("=" * 80)
    print("[B센터] threshold 탐색 (통합 모델 예측값 기준, B 내부val)")
    table_b = sweep("B", b_val_df, cls_bundle, reg_bundle)

    print()
    print("=" * 80)
    print("[A 전 구간 탐색 결과]")
    print(table_a.to_string(index=False))
    print()
    print("[B 전 구간 탐색 결과]")
    print(table_b.to_string(index=False))

    wape_a = table_a.loc[table_a["WAPE"].idxmin()]
    wape_b = table_b.loc[table_b["WAPE"].idxmin()]
    bias_a = table_a.loc[table_a["Bias(%)"].abs().idxmin()]
    bias_b = table_b.loc[table_b["Bias(%)"].abs().idxmin()]
    best_a = table_a.loc[table_a["combined_score"].idxmin()]
    best_b = table_b.loc[table_b["combined_score"].idxmin()]
    print()
    print(f"[A: WAPE 최저(참고용)] threshold={wape_a['threshold']:.2f} (WAPE={wape_a['WAPE']:.3f}%, Bias={wape_a['Bias(%)']:+.3f}%)")
    print(f"[A: Bias 균형(참고용)] threshold={bias_a['threshold']:.2f} (WAPE={bias_a['WAPE']:.3f}%, Bias={bias_a['Bias(%)']:+.3f}%)")
    print(f"[A: 채택 — WAPE+{BIAS_ALPHA}*|Bias| 최저] threshold={best_a['threshold']:.2f} "
          f"(WAPE={best_a['WAPE']:.3f}%, Bias={best_a['Bias(%)']:+.3f}%, combined={best_a['combined_score']:.3f})")
    print(f"[B: WAPE 최저(참고용)] threshold={wape_b['threshold']:.2f} (WAPE={wape_b['WAPE']:.3f}%, Bias={wape_b['Bias(%)']:+.3f}%)")
    print(f"[B: Bias 균형(참고용)] threshold={bias_b['threshold']:.2f} (WAPE={bias_b['WAPE']:.3f}%, Bias={bias_b['Bias(%)']:+.3f}%)")
    print(f"[B: 채택 — WAPE+{BIAS_ALPHA}*|Bias| 최저] threshold={best_b['threshold']:.2f} "
          f"(WAPE={best_b['WAPE']:.3f}%, Bias={best_b['Bias(%)']:+.3f}%, combined={best_b['combined_score']:.3f})")

    threshold_by_center = {"A": float(best_a["threshold"]), "B": float(best_b["threshold"])}
    changed = (
        threshold_by_center != cls_bundle.get("threshold_by_center")
        or cls_bundle.get("threshold") != threshold_by_center["A"]
    )
    if changed:
        cls_bundle["threshold"] = threshold_by_center["A"]  # 스칼라 폴백(하위호환)은 A 값 유지
        cls_bundle["threshold_by_center"] = threshold_by_center
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(cls_bundle, BASE_CLS_MODEL_PATH)
        print(f"  base_model_cls.pkl threshold_by_center 갱신 완료 -> {threshold_by_center} "
              f"(센터별 WAPE+{BIAS_ALPHA}*|Bias| 최저 기준)")
    else:
        print("  기존 threshold_by_center와 동일 — 갱신 불필요")


if __name__ == "__main__":
    main()
