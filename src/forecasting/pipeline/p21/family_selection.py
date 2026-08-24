"""
family_selection.py

P21 horizon별 model family selection.

RF, LightGBM, LSTM, TFT, Informer의 3-seed robustness 결과를 비교해 h1/h2/h4별
최종 family/configuration을 선택한다. model fitting이나 2024 Holdout 접근은 수행하지 않는다.

선택 순서는 Bias gate → 1pp WAPE near-tie → 3-seed Worst-fold WAPE →
seed stability → compute → reproducibility → operational simplicity이며,
각 단계에서 탈락한 family는 이후 단계에서 다시 후보가 되지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.pipeline.common import family_registry as registry
from src.forecasting.common.config import BIAS_GUARDRAIL_ABS_PCT, HORIZONS, NEAR_TIE_WAPE_PCT_POINT

ROBUSTNESS_SEEDS_ORDERED = (42, 123, 456)
OOF_KEY_COLS = ("center_id", "sku_id", "week_st", "target_date")  # common OOF key schema

ELIMINATED_AT_VALUES = (
    "bias_gate", "wape_1pp", "worst_fold", "seed_stability", "compute",
    "reproducibility", "operational_simplicity", "winner", "manual_unresolved_tie",
)

AUDIT_CSV_COLUMNS = (
    "family",
    "seed42_pooled_wape", "seed123_pooled_wape", "seed456_pooled_wape", "three_seed_mean_pooled_wape",
    "seed42_pooled_bias", "seed123_pooled_bias", "seed456_pooled_bias", "three_seed_mean_pooled_bias",
    "bias_gate_pass",
    "minimum_wape_among_bias_pass", "wape_gap_pp", "within_1pp",
    "worst_fold_wape", "worst_fold_id",
    "seed_wape_stability_std",
    "compute_evidence", "compute_rank",
    "reproducibility_assessment", "operational_simplicity_assessment",
    "eliminated_at", "selection_status", "selection_reason",
)

_P12_COMPUTE_REFERENCE_HINT = {
    "rf": "outputs/benchmarks/p12/rf/rf_p13_2023_fold4_cheap.json",
    "lightgbm": "outputs/benchmarks/p12/lightgbm/lightgbm_p13_2023_fold4_cheap.json",
    "lstm": "outputs/benchmarks/p12/lstm/lstm_p13_2023_fold4_supp_high_lb13_high_cost.json",
    "tft": "outputs/benchmarks/p12/tft/tft_p13_2023_fold4_supp_high_lb13_high_cost.json",
    "informer": "outputs/benchmarks/p12/informer/informer_p13_2023_fold4_supp_high_lb13_high_cost.json",
}


def _p12_compute_reference(family: str) -> dict:
    """기존 P12 compute profiling 결과를 불러온다.
    사용 가능한 evidence가 없으면 자동 tie-break에 사용하지 않는다.
    """
    path = Path(_P12_COMPUTE_REFERENCE_HINT.get(family, ""))
    if not path.exists():
        return {"source": str(path), "total_runtime_sec": None, "note": "P12 evidence 파일 없음"}
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    return {"source": str(path), "total_runtime_sec": record.get("timing", {}).get("total_runtime_sec")}


def _load_seed_robustness(robustness_root: Path, family: str, horizon: int) -> dict:
    path = robustness_root / family / f"seed_robustness_{family}_h{horizon}.json"
    if not path.exists():
        raise FileNotFoundError(f"{family} h{horizon}: seed robustness 결과가 없음 - {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _oof_path(hpo_root: Path, family: str, horizon: int, trial_id: int, seed: int) -> Path:
    """family/horizon/trial/seed에 대응하는 common OOF 경로를 반환한다."""
    return hpo_root / family / "oof" / f"oof_{family}_h{horizon}_trial{trial_id}_seed{seed}_common.parquet"


def _expected_fold_ids(hpo_root: Path, family: str, horizon: int) -> set:
    """P13 결과에서 해당 family/horizon이 사용한 fold_id 집합을 읽는다."""
    path = hpo_root / family / f"p13_{family}_h{horizon}.json"
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    return {int(k) for k in record["per_fold_metrics_common"]}


def compute_three_seed_worst_fold(hpo_root: Path, family: str, horizon: int, trial_id: int) -> dict:
    """P21용 3-seed Worst-fold WAPE를 계산한다.
    각 fold에서 seed42/123/456 WAPE를 각각 계산해 평균하고,
    그 fold 평균 중 최댓값을 Worst-fold WAPE로 사용한다.
    세 seed의 evaluation key, fold_id, row 수와 y_true가 일치하지 않으면 fail-fast한다.
    """
    oof_by_seed = {}
    for seed in ROBUSTNESS_SEEDS_ORDERED:
        path = _oof_path(hpo_root, family, horizon, trial_id, seed)
        if not path.exists():
            raise FileNotFoundError(f"{family} h{horizon} seed{seed}: 3-seed worst-fold용 OOF가 없음 - {path}")
        df = pd.read_parquet(path)
        missing = [c for c in (*OOF_KEY_COLS, "fold_id", "y_true", "y_pred") if c not in df.columns]
        if missing:
            raise KeyError(f"{family} h{horizon} seed{seed}: OOF에 필수 컬럼 누락 {missing} ({path})")
        dup = df.duplicated(subset=list(OOF_KEY_COLS)).sum()
        if dup:
            raise ValueError(f"{family} h{horizon} seed{seed}: OOF에 evaluation key 중복 {dup}건 ({path})")
        oof_by_seed[seed] = df

    key_sets = {seed: set(df[list(OOF_KEY_COLS)].itertuples(index=False, name=None))
               for seed, df in oof_by_seed.items()}
    ref_seed = ROBUSTNESS_SEEDS_ORDERED[0]
    for seed in ROBUSTNESS_SEEDS_ORDERED[1:]:
        if key_sets[seed] != key_sets[ref_seed]:
            sym_diff = key_sets[seed] ^ key_sets[ref_seed]
            raise ValueError(
                f"{family} h{horizon}: seed{ref_seed}와 seed{seed}의 evaluation key 집합이 다름 "
                f"(대칭차집합 {len(sym_diff)}건) - intersection으로 넘어가지 않고 fail-fast"
            )

    merged = oof_by_seed[ref_seed][list(OOF_KEY_COLS) + ["fold_id", "y_true", "y_pred"]].rename(
        columns={"fold_id": f"fold_id_{ref_seed}", "y_true": f"y_true_{ref_seed}", "y_pred": f"y_pred_{ref_seed}"}
    )
    for seed in ROBUSTNESS_SEEDS_ORDERED[1:]:
        part = oof_by_seed[seed][list(OOF_KEY_COLS) + ["fold_id", "y_true", "y_pred"]].rename(
            columns={"fold_id": f"fold_id_{seed}", "y_true": f"y_true_{seed}", "y_pred": f"y_pred_{seed}"}
        )
        merged = merged.merge(part, on=list(OOF_KEY_COLS), how="inner", validate="one_to_one")
    if len(merged) != len(oof_by_seed[ref_seed]):
        raise ValueError(f"{family} h{horizon}: merge 후 row 수가 줄어듦(key 불일치 의심) - "
                         f"merged={len(merged)}, seed{ref_seed}={len(oof_by_seed[ref_seed])}")

    fold_id_cols = [f"fold_id_{s}" for s in ROBUSTNESS_SEEDS_ORDERED]
    mismatched_fold = merged[fold_id_cols].nunique(axis=1) > 1
    if mismatched_fold.any():
        raise ValueError(
            f"{family} h{horizon}: 동일 key인데 fold_id가 seed 간 다른 행 {int(mismatched_fold.sum())}건 존재"
        )

    y_true_cols = [f"y_true_{s}" for s in ROBUSTNESS_SEEDS_ORDERED]
    y_true_arr = merged[y_true_cols].to_numpy(dtype=float)
    if not np.all(y_true_arr == y_true_arr[:, [0]]):
        raise ValueError(f"{family} h{horizon}: 동일 key인데 y_true가 seed 간 다른 행 존재")

    for seed in ROBUSTNESS_SEEDS_ORDERED:
        pred = merged[f"y_pred_{seed}"].to_numpy(dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError(f"{family} h{horizon} seed{seed}: y_pred에 NaN/Inf 존재")
    if not np.isfinite(y_true_arr).all():
        raise ValueError(f"{family} h{horizon}: y_true에 NaN/Inf 존재")

    merged["fold_id"] = merged[f"fold_id_{ref_seed}"]
    actual_fold_ids = set(merged["fold_id"].unique().tolist())
    expected_fold_ids = _expected_fold_ids(hpo_root, family, horizon)
    if actual_fold_ids != expected_fold_ids:
        raise ValueError(
            f"{family} h{horizon}: 3-seed OOF의 fold_id 집합 {sorted(actual_fold_ids)}이 P13 fold "
            f"집합 {sorted(expected_fold_ids)}과 다름"
        )

    row_counts = {seed: {} for seed in ROBUSTNESS_SEEDS_ORDERED}
    for seed in ROBUSTNESS_SEEDS_ORDERED:
        raw_counts = oof_by_seed[seed].groupby("fold_id").size().to_dict()
        row_counts[seed] = {int(k): int(v) for k, v in raw_counts.items()}
    for fold_id in actual_fold_ids:
        counts = {seed: row_counts[seed].get(fold_id) for seed in ROBUSTNESS_SEEDS_ORDERED}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{family} h{horizon} fold{fold_id}: seed별 row 수가 다름 - {counts}")

    fold_detail = {}
    for fold_id, grp in merged.groupby("fold_id"):
        per_seed_wape = {}
        for seed in ROBUSTNESS_SEEDS_ORDERED:
            y_true = grp[f"y_true_{seed}"].to_numpy(dtype=float)
            y_pred = grp[f"y_pred_{seed}"].to_numpy(dtype=float)
            per_seed_wape[seed] = ev.compute_metrics(y_true, y_pred)["wape"]
        fold_mean_wape = statistics.mean(per_seed_wape.values())
        fold_detail[int(fold_id)] = {
            "seed42_wape": per_seed_wape[42], "seed123_wape": per_seed_wape[123],
            "seed456_wape": per_seed_wape[456], "three_seed_mean_wape": fold_mean_wape,
            "n_rows": int(len(grp)),
        }

    worst_fold_id = max(fold_detail, key=lambda f: fold_detail[f]["three_seed_mean_wape"])
    worst_fold_wape = fold_detail[worst_fold_id]["three_seed_mean_wape"]
    return {"worst_fold_id": worst_fold_id, "worst_fold_wape": worst_fold_wape, "fold_detail": fold_detail}


def _seed_metric(per_seed_metrics: dict, seed: int, key: str) -> float:
    return (per_seed_metrics.get(str(seed)) or per_seed_metrics[seed])[key]


def _build_family_records(hpo_root: Path, robustness_root: Path, horizon: int) -> list[dict]:
    records = []
    for family in registry.FAMILIES:
        sr = _load_seed_robustness(robustness_root, family, horizon)
        pm = sr["per_seed_metrics"]
        seed_wapes = [_seed_metric(pm, s, "wape") for s in (42, 123, 456)]
        worst_fold = compute_three_seed_worst_fold(hpo_root, family, horizon, sr["trial_id"])
        records.append({
            "family": family, "trial_id": sr["trial_id"], "params": sr["params"],
            "seed42_pooled_wape": seed_wapes[0], "seed123_pooled_wape": seed_wapes[1],
            "seed456_pooled_wape": seed_wapes[2], "three_seed_mean_pooled_wape": sr["three_seed_mean_pooled_wape"],
            "seed42_pooled_bias": _seed_metric(pm, 42, "bias"), "seed123_pooled_bias": _seed_metric(pm, 123, "bias"),
            "seed456_pooled_bias": _seed_metric(pm, 456, "bias"), "three_seed_mean_pooled_bias": sr["three_seed_mean_pooled_bias"],
            "bias_gate_pass": None,
            "minimum_wape_among_bias_pass": None, "wape_gap_pp": None, "within_1pp": None,
            "worst_fold_wape": worst_fold["worst_fold_wape"],
            "worst_fold_id": worst_fold["worst_fold_id"],
            "worst_fold_detail": worst_fold["fold_detail"],  # JSON only
            "seed_wape_stability_std": statistics.pstdev(seed_wapes),
            "compute_evidence": None, "compute_rank": None,
            "reproducibility_assessment": None, "operational_simplicity_assessment": None,
            "eliminated_at": None, "selection_status": "pending", "selection_reason": None,
        })
    return records


def _eliminate(records: list[dict], step: str, reason_fn) -> None:
    for r in records:
        r["eliminated_at"] = step
        r["selection_status"] = "eliminated"
        r["selection_reason"] = reason_fn(r)


def _mark_winner(record: dict, reason: str) -> None:
    record["eliminated_at"] = "winner"
    record["selection_status"] = "winner"
    record["selection_reason"] = reason


def select_horizon_winner(
    hpo_root: Path, robustness_root: Path, horizon: int,
    manual_reproducibility_ranking: list[str] | None = None,
    manual_operational_simplicity_ranking: list[str] | None = None,
) -> dict:
    """Frozen P21 hierarchy를 순서대로 적용하고 family별 audit record와 selection trace를 반환한다."""
    records = _build_family_records(hpo_root, robustness_root, horizon)
    by_family = {r["family"]: r for r in records}
    trace = []

    # Step 1: Bias gate
    for r in records:
        r["bias_gate_pass"] = abs(r["three_seed_mean_pooled_bias"]) <= BIAS_GUARDRAIL_ABS_PCT
    survivors = [r for r in records if r["bias_gate_pass"]]
    failed = [r for r in records if not r["bias_gate_pass"]]
    _eliminate(failed, "bias_gate",
              lambda r: f"|3-seed mean bias|={abs(r['three_seed_mean_pooled_bias']):.4f}% > {BIAS_GUARDRAIL_ABS_PCT}%")
    trace.append({"step": "1_bias_gate", "pass": [r["family"] for r in survivors],
                 "fail": [r["family"] for r in failed]})

    if not survivors:
        return _finalize(records, None, trace, horizon, "no_family_passed_bias_guardrail")

    # Step 2: WAPE minimum + 1pp near-tie
    min_wape = min(r["three_seed_mean_pooled_wape"] for r in survivors)
    min_family = next(r["family"] for r in survivors if r["three_seed_mean_pooled_wape"] == min_wape)
    for r in survivors:
        r["minimum_wape_among_bias_pass"] = min_wape
        r["wape_gap_pp"] = r["three_seed_mean_pooled_wape"] - min_wape
        r["within_1pp"] = r["wape_gap_pp"] <= NEAR_TIE_WAPE_PCT_POINT
    next_survivors = [r for r in survivors if r["within_1pp"]]
    dropped = [r for r in survivors if not r["within_1pp"]]
    _eliminate(dropped, "wape_1pp", lambda r: f"wape_gap_pp={r['wape_gap_pp']:.4f} > {NEAR_TIE_WAPE_PCT_POINT}pp")
    trace.append({"step": "2_wape_1pp", "minimum_wape": min_wape, "minimum_family": min_family,
                 "within_1pp": [r["family"] for r in next_survivors], "eliminated": [r["family"] for r in dropped]})
    survivors = next_survivors

    if len(survivors) == 1:
        _mark_winner(survivors[0], "unique_min_wape_within_bias_guardrail")
        return _finalize(records, survivors[0]["family"], trace, horizon, "unique_min_wape_within_bias_guardrail")

    # Step 3: worst-fold WAPE
    min_wf = min(r["worst_fold_wape"] for r in survivors)
    next_survivors = [r for r in survivors if r["worst_fold_wape"] == min_wf]
    dropped = [r for r in survivors if r["worst_fold_wape"] != min_wf]
    _eliminate(dropped, "worst_fold", lambda r: f"worst_fold_wape={r['worst_fold_wape']:.4f} > min {min_wf:.4f}")
    trace.append({"step": "3_worst_fold", "minimum_worst_fold_wape": min_wf,
                 "remaining": [r["family"] for r in next_survivors], "eliminated": [r["family"] for r in dropped]})
    survivors = next_survivors

    if len(survivors) == 1:
        _mark_winner(survivors[0], "worst_fold_wape_minimum")
        return _finalize(records, survivors[0]["family"], trace, horizon, "worst_fold_wape_minimum")

    # Step 4: seed stability
    min_std = min(r["seed_wape_stability_std"] for r in survivors)
    next_survivors = [r for r in survivors if r["seed_wape_stability_std"] == min_std]
    dropped = [r for r in survivors if r["seed_wape_stability_std"] != min_std]
    _eliminate(dropped, "seed_stability", lambda r: f"seed_wape_stability_std={r['seed_wape_stability_std']:.6f} > min {min_std:.6f}")
    trace.append({"step": "4_seed_stability", "minimum_std": min_std,
                 "remaining": [r["family"] for r in next_survivors], "eliminated": [r["family"] for r in dropped]})
    survivors = next_survivors

    if len(survivors) == 1:
        _mark_winner(survivors[0], "seed_stability_minimum_std")
        return _finalize(records, survivors[0]["family"], trace, horizon, "seed_stability_minimum_std")

    # Step 5: P12 compute evidence
    for r in survivors:
        r["compute_evidence"] = _p12_compute_reference(r["family"])
    runtimes = {r["family"]: r["compute_evidence"]["total_runtime_sec"] for r in survivors}
    all_known = all(v is not None for v in runtimes.values())
    all_distinct = len(set(runtimes.values())) == len(runtimes) if all_known else False
    if all_known and all_distinct:
        ranked = sorted(survivors, key=lambda r: runtimes[r["family"]])
        for rank, r in enumerate(ranked, start=1):
            r["compute_rank"] = rank
        winner_r, dropped = ranked[0], ranked[1:]
        _eliminate(dropped, "compute", lambda r: f"total_runtime_sec={runtimes[r['family']]} > {runtimes[winner_r['family']]}(winner)")
        trace.append({"step": "5_compute", "runtimes_sec": runtimes,
                     "remaining": [winner_r["family"]], "eliminated": [r["family"] for r in dropped]})
        _mark_winner(winner_r, "lowest_p12_compute_cost")
        return _finalize(records, winner_r["family"], trace, horizon, "lowest_p12_compute_cost")

    trace.append({"step": "5_compute", "runtimes_sec": runtimes,
                 "note": "compute evidence 없거나 동률 - 자동 해소 불가, step 6으로 이동",
                 "remaining": [r["family"] for r in survivors]})

    # Step 6: reproducibility
    for r in survivors:
        r["reproducibility_assessment"] = "manual_review_required"
    if manual_reproducibility_ranking:
        ranked_families = [f for f in manual_reproducibility_ranking if f in {r["family"] for r in survivors}]
        if ranked_families:
            winner_family = ranked_families[0]
            for r in survivors:
                r["reproducibility_assessment"] = f"manual_rank={manual_reproducibility_ranking.index(r['family']) + 1}" \
                    if r["family"] in manual_reproducibility_ranking else "not_ranked"
            dropped = [r for r in survivors if r["family"] != winner_family]
            _eliminate(dropped, "reproducibility", lambda r: "manual_reproducibility_ranking에서 후순위")
            winner_r = by_family[winner_family]
            trace.append({"step": "6_reproducibility", "manual_ranking": manual_reproducibility_ranking,
                         "remaining": [winner_family]})
            _mark_winner(winner_r, "manual_reproducibility_ranking")
            return _finalize(records, winner_family, trace, horizon, "manual_reproducibility_ranking")
    trace.append({"step": "6_reproducibility", "note": "manual ranking 미제공 - 자동 해소 불가, step 7으로 이동",
                 "remaining": [r["family"] for r in survivors]})

    # Step 7: operational simplicity
    for r in survivors:
        r["operational_simplicity_assessment"] = "manual_review_required"
    if manual_operational_simplicity_ranking:
        ranked_families = [f for f in manual_operational_simplicity_ranking if f in {r["family"] for r in survivors}]
        if ranked_families:
            winner_family = ranked_families[0]
            for r in survivors:
                r["operational_simplicity_assessment"] = (
                    f"manual_rank={manual_operational_simplicity_ranking.index(r['family']) + 1}"
                    if r["family"] in manual_operational_simplicity_ranking else "not_ranked"
                )
            dropped = [r for r in survivors if r["family"] != winner_family]
            _eliminate(dropped, "operational_simplicity", lambda r: "manual_operational_simplicity_ranking에서 후순위")
            winner_r = by_family[winner_family]
            trace.append({"step": "7_operational_simplicity", "manual_ranking": manual_operational_simplicity_ranking,
                         "remaining": [winner_family]})
            _mark_winner(winner_r, "manual_operational_simplicity_ranking")
            return _finalize(records, winner_family, trace, horizon, "manual_operational_simplicity_ranking")

    # 자동/manual tie-break로 결정되지 않으면 unresolved tie로 종료
    for r in survivors:
        r["eliminated_at"] = "manual_unresolved_tie"
        r["selection_status"] = "tied_manual_review"
        r["selection_reason"] = "step 5~7까지 자동/manual 해소 실패 - 임의 winner 생성 금지"
    trace.append({"step": "7_operational_simplicity", "note": "manual ranking 미제공 - 자동 해소 불가",
                 "remaining": [r["family"] for r in survivors]})
    trace.append({"step": "final", "result": "manual_unresolved_tie", "tied_candidates": [r["family"] for r in survivors]})
    return _finalize(records, None, trace, horizon, "manual_unresolved_tie")


def _finalize(records: list[dict], winner_family: str | None, trace: list[dict], horizon: int, reason: str) -> dict:
    winner = next((r for r in records if r["family"] == winner_family), None) if winner_family else None
    tied = [r for r in records if r["selection_status"] == "tied_manual_review"]
    return {
        "horizon": horizon,
        "winner": {"family": winner["family"], "trial_id": winner["trial_id"], "params": winner["params"]} if winner else None,
        "reason": reason,
        "unresolved_manual_tie": bool(tied),
        "tied_candidates": [r["family"] for r in tied] if tied else [],
        "selection_trace": trace,
        "family_audit_records": records,
    }


def _write_audit_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(AUDIT_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["compute_evidence"] = json.dumps(row["compute_evidence"], ensure_ascii=False) if row["compute_evidence"] else ""
            writer.writerow(row)


def print_selection_trace(horizon: int, result: dict) -> None:
    print(f"\n=== h{horizon} P21 selection trace ===")
    for step in result["selection_trace"]:
        print(f"  {step}")
    if result["winner"]:
        print(f"  FINAL: winner={result['winner']['family']} reason={result['reason']}")
    else:
        print(f"  FINAL: winner=None reason={result['reason']} tied={result['tied_candidates']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P21 horizon별 5-family selection + audit trail (2024 미접근)")
    parser.add_argument("--hpo-root", default="outputs/hpo")
    parser.add_argument("--robustness-root", default="outputs/robustness/seed")
    parser.add_argument("--output-dir", default="outputs/p21")
    args = parser.parse_args()

    hpo_root, robustness_root, out_dir = Path(args.hpo_root), Path(args.robustness_root), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        result = select_horizon_winner(hpo_root, robustness_root, horizon)
        print_selection_trace(horizon, result)

        json_path = out_dir / f"p21_h{horizon}_selection.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        csv_path = out_dir / f"p21_h{horizon}_selection_audit.csv"
        _write_audit_csv(csv_path, result["family_audit_records"])
        print(f"[P21] h{horizon} 저장: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
