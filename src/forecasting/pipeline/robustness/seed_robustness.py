"""
seed_robustness.py

P13 selected configuration의 seed robustness 평가.

각 family×horizon에서 P13 seed42 OOF를 재사용하고 seed123/456만 추가 실행한다.
family와 hyperparameter는 변경하지 않으며, 3-seed Pooled WAPE/Bias를 계산해
P21 family selection에 전달한다. best seed selection은 수행하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.pipeline.common import family_registry as registry
from src.forecasting.pipeline.p13 import hpo_common as hc
from src.forecasting.common.config import HORIZONS, ROBUSTNESS_SEEDS
from src.forecasting.common.data_loader import load_development

ADDITIONAL_SEEDS = tuple(s for s in ROBUSTNESS_SEEDS if s != 42)
if set(ADDITIONAL_SEEDS) != {123, 456} or 42 not in ROBUSTNESS_SEEDS:
    raise RuntimeError(f"common.config.ROBUSTNESS_SEEDS={ROBUSTNESS_SEEDS}가 frozen spec (42,123,456)과 다름")


def _load_p13_selection(hpo_root: Path, family: str, horizon: int) -> dict:
    """P13 결과에서 해당 family×horizon의 selected configuration을 읽는다."""
    path = hpo_root / family / f"p13_{family}_h{horizon}.json"
    if not path.exists():
        raise FileNotFoundError(f"{family} h{horizon}: P13 HPO 결과가 없음 - {path}")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    sel = record["selection"]["selected"]
    if sel is None:
        raise ValueError(f"{family} h{horizon}: P13 HPO에서 guardrail 통과 trial이 없어 seed robustness 대상이 없음")
    return sel


def _seed42_oof_path(hpo_root: Path, family: str, horizon: int, trial_id: int) -> Path:
    return hpo_root / family / "oof" / f"oof_{family}_h{horizon}_trial{trial_id}_seed42_common.parquet"


def run_family_horizon_seed_robustness(
    hpo_root: Path, robustness_root: Path, family: str, horizon: int,
    sub_a: pd.DataFrame, checkpoint_root: Path, common_eval_keys: pd.DataFrame,
) -> dict:
    sel = _load_p13_selection(hpo_root, family, horizon)
    trial_id, hp = sel["trial_id"], sel["params"]
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    seed42_path = _seed42_oof_path(hpo_root, family, horizon, trial_id)
    if not seed42_path.exists():
        raise FileNotFoundError(
            f"{family} h{horizon}: seed42 OOF가 없음 - {seed42_path} (P13 HPO를 먼저 완료해야 함)"
        )

    oof_paths = {42: seed42_path}
    per_seed_metrics = {}
    for seed in (42, *ADDITIONAL_SEEDS):
        if seed != 42:
            trial_label = f"seedrobust_h{horizon}_trial{trial_id}_seed{seed}"
            run = registry.run_single_config(
                family, sub_a, horizon, hp, seed, trial_label,
                stage="p13_seed_robustness", checkpoint_root=checkpoint_root,
            )
            if run["status"] != "ok":
                raise RuntimeError(f"{family} h{horizon} seed{seed}: 학습 실패 - {run['error']}")
            pooled_native_oof = pd.concat(run["oof_frames"], ignore_index=True)
            # seed42와 동일한 common evaluation population으로 맞춘다.
            filtered_oof = hc.filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
            out_path = hpo_root / family / "oof" / f"oof_{family}_h{horizon}_trial{trial_id}_seed{seed}_common.parquet"
            filtered_oof.to_parquet(out_path, index=False)
            oof_paths[seed] = out_path

        oof = pd.read_parquet(oof_paths[seed])
        metrics = ev.compute_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["mase_scale"].to_numpy())
        per_seed_metrics[seed] = {"wape": metrics["wape"], "bias": metrics["bias"]}

    mean_wape = sum(m["wape"] for m in per_seed_metrics.values()) / len(per_seed_metrics)
    mean_bias = sum(m["bias"] for m in per_seed_metrics.values()) / len(per_seed_metrics)

    result = {
        "family": family, "horizon": horizon, "trial_id": trial_id, "params": hp,
        "seeds": sorted(per_seed_metrics),
        "per_seed_metrics": per_seed_metrics,
        "three_seed_mean_pooled_wape": mean_wape,
        "three_seed_mean_pooled_bias": mean_bias,
        "oof_paths": {str(s): str(p) for s, p in oof_paths.items()},
        "note": "best seed selection 없음 - 3-seed 평균만 P21로 전달",
    }

    out_dir = robustness_root / family
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_robustness_{family}_h{horizon}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="P13 family x horizon seed123/456 robustness")
    parser.add_argument("--hpo-root", default="outputs/hpo")
    parser.add_argument("--robustness-root", default="outputs/robustness/seed")
    parser.add_argument("--family", choices=registry.FAMILIES, help="생략하면 5개 family 전부")
    parser.add_argument("--horizon", type=int, choices=HORIZONS, help="생략하면 h1/h2/h4 전부")
    parser.add_argument("--common-eval-keys-path", required=True)
    args = parser.parse_args()

    common_eval_keys = hc.load_common_eval_keys(Path(args.common_eval_keys_path))
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    hpo_root = Path(args.hpo_root)
    robustness_root = Path(args.robustness_root)
    checkpoint_root = robustness_root / "_checkpoints"

    families = [args.family] if args.family else list(registry.FAMILIES)
    horizons = [args.horizon] if args.horizon else list(HORIZONS)

    for family in families:
        for horizon in horizons:
            print(f"[SEED-ROBUSTNESS] {family} h{horizon} 시작...")
            result = run_family_horizon_seed_robustness(
                hpo_root, robustness_root, family, horizon, sub_a, checkpoint_root, common_eval_keys,
            )
            print(
                f"[SEED-ROBUSTNESS] {family} h{horizon}: "
                f"3-seed mean WAPE={result['three_seed_mean_pooled_wape']:.4f} "
                f"Bias={result['three_seed_mean_pooled_bias']:.4f}"
            )


if __name__ == "__main__":
    main()
