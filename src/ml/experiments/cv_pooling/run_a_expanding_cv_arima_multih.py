"""
run_a_expanding_cv_arima_multih.py
"A센터 CV를 왜 2023년 안에서만 했는지"를 검증하기 위한 ARIMA(no_exog, 확정 채택
variant) 확장 실험 — LightGBM 쪽(run_a_expanding_cv_lgbm_multih.py)과 동일한 folds.A_EXPANDING_H2_FOLDS, 동일 horizon(h=1,2,4)으로 나란히 비교한다.

order=(1,0,1)(Day4 확정값 고정, 재탐색 없음), exog 없음(no_exog가 이미 채택된 variant이므로 재확인 목적으로만 no_exog 하나만 돈다 — 4개 variant 전체 ablation은 이번 실험 범위 밖). arimax_exog.py의 fit_and_eval(rolling-origin, SARIMAX.append (refit=False))을 그대로 재사용한다.

python run_a_expanding_cv_arima_multih.py

산출물: data/ml/models/arimax/a_expanding_cv_arima_multih_report.{csv,md}
프로덕션 파일은 read-only로만 읽는다(feature_table_final.parquet).
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
import arimax_exog as ax  # noqa: E402

HORIZONS = [1, 2, 4]
RESULT_DIR = ax.RESULT_DIR
ROUND = 3


def main():
    print("=" * 80)
    print("[Day5-ext ARIMA] feature_table_final.parquet 로드 및 (center_id, week_st) 집계")
    cols_needed = [ax.CENTER_COL, ax.WEEK_COL, ax.QTY_COL, "split"] + ax.RAW_AGG_CANDIDATES
    df = pd.read_parquet(ax.FEATURE_TABLE_PATH, columns=cols_needed)
    agg = ax.aggregate_to_center_week(df)
    sub = agg[agg[ax.CENTER_COL] == "A"].reset_index(drop=True)
    order = ax.ORDER_FIXED["A"]
    print(f"  order={order}(Day4 확정값 고정), horizons={HORIZONS}, variant=no_exog(확정 채택분만)")

    fold_rows = []
    for f in folds.A_EXPANDING_H2_FOLDS:
        print("=" * 80)
        print(f"[Fold {f['fold']}] train {f['train_start']}~{f['train_end']} -> val {f['val_start']}~{f['val_end']}")
        train_df = sub[(sub[ax.WEEK_COL] >= pd.Timestamp(f["train_start"])) &
                       (sub[ax.WEEK_COL] <= pd.Timestamp(f["train_end"]))].reset_index(drop=True)
        val_df = sub[(sub[ax.WEEK_COL] >= pd.Timestamp(f["val_start"])) &
                     (sub[ax.WEEK_COL] <= pd.Timestamp(f["val_end"]))].reset_index(drop=True)
        train_vals = train_df[ax.QTY_COL].to_numpy(dtype=float)
        val_vals = val_df[ax.QTY_COL].to_numpy(dtype=float)
        print(f"  train {len(train_df)}주 / val {len(val_df)}주")

        out = ax.fit_and_eval(train_vals, val_vals, order, HORIZONS, train_df, val_df, cols=[], center="A")
        perf = out["explanatory"]  # exog 없어 operational==explanatory
        for _, r in perf.iterrows():
            fold_rows.append({"fold": f["fold"], "train_start": f["train_start"], "train_end": f["train_end"],
                               "val_start": f["val_start"], "val_end": f["val_end"], **r.to_dict()})
            print(f"  [Fold {f['fold']} {r['horizon']}] Val WAPE={r['WAPE']:.2f}% RMSE={r['RMSE']:.2f} n={r['n']}")

    fold_df = pd.DataFrame(fold_rows)
    print()
    print("=" * 80)
    print("[Fold별 상세]")
    print(fold_df.to_string(index=False))

    fold1_wape = fold_df[fold_df["fold"] == 1][["horizon", "WAPE"]].rename(columns={"WAPE": "Fold1_WAPE(train 1yr)"})
    rest_mean = (
        fold_df[fold_df["fold"] != 1].groupby("horizon")["WAPE"].mean()
        .reset_index().rename(columns={"WAPE": "Fold2~4_WAPE_mean(train 2~3yr)"})
    )
    stability = fold1_wape.merge(rest_mean, on="horizon")
    print()
    print("[Fold1(train 1년) vs Fold2~4(train 2~3년) 평균 비교 - 데이터 부족 가설 검증]")
    print(stability.to_string(index=False))

    print()
    print("=== 최종 holdout: train='train' split(2021~2023) -> eval='val'+'test'(2024 전체) 연속 rolling-origin ===")
    train_df = sub[sub["split"] == "train"].reset_index(drop=True)
    eval_df = sub[sub["split"].isin(ax.A_HOLDOUT_SPLITS)].reset_index(drop=True)
    train_vals = train_df[ax.QTY_COL].to_numpy(dtype=float)
    eval_vals = eval_df[ax.QTY_COL].to_numpy(dtype=float)
    print(f"  train {len(train_df)}주 / holdout eval {len(eval_df)}주")
    final_out = ax.fit_and_eval(train_vals, eval_vals, order, HORIZONS, train_df, eval_df, cols=[], center="A")
    final_df = final_out["explanatory"]
    for _, r in final_df.iterrows():
        print(f"  [최종 {r['horizon']}] 2024 Test WAPE={r['WAPE']:.2f}% RMSE={r['RMSE']:.2f} n={r['n']}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(RESULT_DIR / "a_expanding_cv_arima_multih_folds.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(RESULT_DIR / "a_expanding_cv_arima_multih_final.csv", index=False, encoding="utf-8-sig")

    md_lines = ["# A센터 ARIMA(no_exog) Expanding CV 확장 실험 (h=1,2,4)", "",
                "## Fold별 상세 (Val WAPE)", "",
                "| " + " | ".join(fold_df.columns) + " |",
                "| " + " | ".join(["---"] * len(fold_df.columns)) + " |"]
    for _, r in fold_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "## Fold1(train 1년) vs Fold2~4(train 2~3년) 평균", "",
                 "| " + " | ".join(stability.columns) + " |",
                 "| " + " | ".join(["---"] * len(stability.columns)) + " |"]
    for _, r in stability.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "## 최종(train 2021~2023) -> 2024 홀드아웃(val+test 연속 rolling-origin)", "",
                 "| " + " | ".join(final_df.columns) + " |",
                 "| " + " | ".join(["---"] * len(final_df.columns)) + " |"]
    for _, r in final_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "각주: order=(1,0,1) 고정, no_exog(확정 채택 variant)만 실행. "
                      "rolling-origin은 origin마다 get_forecast(steps=max_h) 1회로 h=1/2/4를 동시에 추출하고, "
                      "append(refit=False)로 매주 1주씩 실측만 반영(재추정 없음)."]
    (RESULT_DIR / "a_expanding_cv_arima_multih_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n저장 완료 -> {RESULT_DIR}/a_expanding_cv_arima_multih_*.{{csv,md}}")


if __name__ == "__main__":
    main()
