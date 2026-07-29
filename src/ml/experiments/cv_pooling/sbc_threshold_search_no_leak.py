"""
sbc_threshold_search_no_leak.py
프로덕션 search_threshold_sbc_groups.py의 SBC(ADI축) 2그룹별 threshold 탐색을
그대로 재현하되, "2024는 어떤 탐색 루프에도 유출되면 안 된다"는 새 정책을 반영한
버전. 프로덕션 스크립트는 A의 threshold 탐색에 A `split=='val'`(2024-01~06)을
직접 사용하는데(search_threshold_sbc_groups.py:147-154), 이 파일에서는 그 한
지점만 "A train(2021~2023)의 시간순 마지막 4주"로 교체한다. 그 외 로직(그룹
분리 기준, sweep, 목적함수)은 production 모듈에서 그대로 import해서 재사용하며
production 파일은 전혀 수정하지 않는다.

python sbc_threshold_search_no_leak.py

핵심 설계:
    1) B는 원래부터 B train 마지막 4주를 내부 검증셋으로 쓰고 있어(production
       load_b_internal_val) 이미 2024를 안 본다 — 그대로 재사용.
    2) A도 동일한 관례(train 마지막 N주를 내부 val로)로 통일 — B와 대칭적인
       구조가 되어 일관성도 좋아진다.
    3) 최종 홀드아웃은 A `split.isin(['val','test'])`(=2024 전체, 새 정책이
       요구하는 "2024 전체가 최종 holdout") + B는 기존과 동일한 53-fold
       walk-forward. 1회만 적용하고 재탐색하지 않음(production 관례 그대로).
    4) 모델(base_model_cls/reg.pkl)은 재학습하지 않는다 — "B 학습체계 유지"
       원칙과 동일하게 A도 기존 통합 Base Model을 그대로 쓰고, threshold
       탐색의 검증 데이터 소스만 바꾼다.
    5) 이 스크립트는 아무 파일도 덮어쓰지 않는다(production 스크립트와 동일하게
       print 리포트 + 이 실험 전용 리포트 파일만 생성).
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))

from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    apply_target_date_boundary_filter,
    compute_metrics,
    load_model_bundle,
)
from search_threshold_sbc_groups import (  # noqa: E402
    GROUPS,
    THRESHOLD_GRID,
    base_predict_parts,
    choose_group_thresholds,
    group_mask,
    load_b_internal_val,
    predict_group_threshold,
    sweep_group,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402

REPORT_DIR = cc.CV_DIR / "reports"
A_INTERNAL_VAL_WEEKS = 4


def load_a_internal_val(df: pd.DataFrame) -> pd.DataFrame:
    """A train(2021~2023)의 시간순 마지막 4주를 내부 검증셋으로 쓴다 — production의
    load_b_internal_val과 대칭적인 구조. A split=='val'(2024H1)은 여기서 전혀 안 씀."""
    a_train_full = df[(df["center_id"] == "A") & (df["split"] == "train")].copy()
    a_train_full = apply_target_date_boundary_filter(a_train_full, horizon_weeks=1)
    a_train_full = a_train_full[a_train_full[TARGET_COL].notna()]
    a_weeks = sorted(a_train_full["week_st"].unique())
    val_weeks_a = a_weeks[-A_INTERNAL_VAL_WEEKS:]
    val = a_train_full[a_train_full["week_st"].isin(val_weeks_a)]
    print(f"  A 내부 val(2021~2023 train 마지막 {A_INTERNAL_VAL_WEEKS}주): "
          f"{pd.Timestamp(val_weeks_a[0]).date()} ~ {pd.Timestamp(val_weeks_a[-1]).date()}, {len(val):,}행")
    return val


def final_eval_full_2024(df: pd.DataFrame, cls_bundle, reg_bundle, thresholds: dict) -> pd.DataFrame:
    """production final_eval과 동일하되 A 홀드아웃을 'test'(24H2)가 아니라
    'val'+'test'(=2024 전체)로 확장 — 새 정책("2024 전체가 최종 holdout") 반영."""
    a_holdout = df[(df["center_id"] == "A") & (df["split"].isin(["val", "test"]))].copy()
    a_holdout = a_holdout[a_holdout[TARGET_COL].notna()]
    y_true_a = a_holdout[TARGET_COL].to_numpy()
    naive_a = a_holdout["qty"].to_numpy()
    pred_a = predict_group_threshold(a_holdout, cls_bundle, reg_bundle, thresholds)
    m_a = compute_metrics(y_true_a, pred_a, naive_a)

    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        wf_folds = json.load(f)
    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()
    all_true, all_pred, all_naive = [], [], []
    for train_start, train_end, val_start, val_end in wf_folds:
        mask = (b_pool["week_st"] >= val_start) & (b_pool["week_st"] <= val_end)
        fold_df = b_pool[mask]
        fold_df = fold_df[fold_df[TARGET_COL].notna()]
        if len(fold_df) == 0:
            continue
        all_true.append(fold_df[TARGET_COL].to_numpy())
        all_pred.append(predict_group_threshold(fold_df, cls_bundle, reg_bundle, thresholds))
        all_naive.append(fold_df["qty"].to_numpy())
    m_b = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_naive))

    return pd.DataFrame([
        {"center": "A", "n": len(a_holdout), **{k: round(v, 3) for k, v in m_a.items()}},
        {"center": "B", "n": len(np.concatenate(all_true)), **{k: round(v, 3) for k, v in m_b.items()}},
    ])


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    print(f"[기존 단일 threshold] {cls_bundle.get('threshold')}")

    a_val = load_a_internal_val(df)
    b_val = load_b_internal_val(df)
    print(f"A 내부 val: {len(a_val):,}행 / B 내부 val: {len(b_val):,}행 (둘 다 2024 데이터 미포함)")

    print("=" * 80)
    print("[그룹별 threshold 탐색] (A_WAPE + B_WAPE)/2 최소 기준, 2021~2023 내부 val만 사용")
    chosen = choose_group_thresholds(a_val, b_val, cls_bundle, reg_bundle)
    print(f"\n[채택된 그룹별 threshold] {chosen}")

    print()
    print("=" * 80)
    print("[최종 홀드아웃 1회 적용] A=2024 전체, B=53-fold walk-forward")
    report_grouped = final_eval_full_2024(df, cls_bundle, reg_bundle, chosen)
    print(report_grouped.to_string(index=False))

    single_th = {"SE": cls_bundle.get("threshold", 0.80), "IL": cls_bundle.get("threshold", 0.80)}
    report_single = final_eval_full_2024(df, cls_bundle, reg_bundle, single_th)
    print()
    print(f"[비교: 기존 단일 threshold={single_th['SE']}]")
    print(report_single.to_string(index=False))

    print()
    print("=" * 80)
    print("[개선폭] 그룹별 threshold(2024 미유출 탐색) - 단일 threshold")
    delta_rows = []
    for center in ["A", "B"]:
        g = report_grouped[report_grouped["center"] == center].iloc[0]
        s = report_single[report_single["center"] == center].iloc[0]
        row = {"center": center}
        for metric in ["RMSE", "MAE", "WAPE", "MASE"]:
            delta = g[metric] - s[metric]
            direction = "개선" if delta < 0 else "악화"
            row[f"{metric}_단일"] = s[metric]
            row[f"{metric}_그룹"] = g[metric]
            row[f"{metric}_방향"] = direction
            print(f"  [{center}] {metric}: {s[metric]:.3f} -> {g[metric]:.3f} ({direction})")
        delta_rows.append(row)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_grouped.assign(threshold_mode="group_no_leak").to_csv(
        REPORT_DIR / "sbc_threshold_no_leak_report.csv", index=False, encoding="utf-8-sig"
    )
    md_lines = [
        f"# SBC 그룹별 threshold 탐색 (2024 미유출 버전)",
        "",
        f"채택된 threshold: {chosen} (기존 단일 threshold: {single_th['SE']})",
        "",
        "## 그룹별 threshold 최종 홀드아웃 성능",
        "| " + " | ".join(report_grouped.columns) + " |",
        "| " + " | ".join(["---"] * len(report_grouped.columns)) + " |",
    ]
    for _, r in report_grouped.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        "각주: A 그룹별 threshold 탐색은 A train(2021~2023) 마지막 4주만 사용(2024 미참조). "
        "B는 production과 동일하게 B train 마지막 4주 사용. 최종 홀드아웃은 A=2024 전체(기존 "
        "'test'만 보던 production final_eval과 달리 'val'+'test' 전체로 확장), B=53-fold "
        "walk-forward. base_model_cls/reg.pkl은 재학습하지 않았고 어떤 파일도 덮어쓰지 않음."
    )
    (REPORT_DIR / "sbc_threshold_no_leak_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n저장 완료 -> {REPORT_DIR}/sbc_threshold_no_leak_report.{{csv,md}}")


if __name__ == "__main__":
    main()
