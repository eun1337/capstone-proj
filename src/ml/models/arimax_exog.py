"""
arimax_exog.py
통계 트랙 Day5: ARIMAX (외생변수 투입판, Day4 arima_baseline.py 기반)

핵심 설계:
    1) (center_id, week_st) 단위로 qty=sum, 외생변수 후보=first() 집계. 그룹 내
       nunique()==1을 검증하고 어긋나면 즉시 에러.
    2) 실제 투입 exog 컬럼은 select_exog()로 센터별 train 구간 기준 매 실행마다
       런타임 자동 판별(하드코딩 금지).
    3) temp_x_precip은 주간 집계된 평균온도 x 총강수량으로 새로 계산.
    4) order는 Day4 확정값 고정 재사용(A=(1,0,1), B=(0,1,0)), 재탐색 없음. 계수는
       train에서 1회만 추정하고, rolling-origin 평가는 SARIMAX.append(refit=False)로
       진행.
    5) 평가 이원화: 설명적(explanatory, 실제 미래 exog 사용)과 실전(operational,
       미래에 알 수 없는 값은 대체 규칙 사용) 두 모드로 get_forecast(). append()는
       두 모드와 무관하게 항상 실측 exog로만 상태 갱신.
    6) Ablation: 캘린더군은 대체 규칙 대상이 없어 operational==explanatory라 한 번만
       계산, 날씨군은 둘 다 계산.
    7) 계수 해석표: level(원 단위) 모델이므로 이진 변수 상대효과율은
       coef / train qty 평균 x 100으로 정의. B센터는 train 26주라 exploratory 한정.
    8) A센터 exog 조합 선택은 2023 안 4-Window CV(run_center_a_cv)에서 끝내고,
       val/test(2024 H1/H2)는 확정 조합을 딱 한 번 holdout 재확인(run_final_holdout_test)
       하는 용도로만 사용. 기존 통합 결과 파일은 덮어쓰지 않고 별도 저장. B센터는
       이번 분리 산출 대상 아님(기존 pool 평가/53주 walk-forward 그대로).
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import MODEL_DIR, compute_metrics

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"
RESULT_DIR = MODEL_DIR / "arimax"

# 4-Window CV 경계는 cv_pooling/folds.py(OPTION2_FOLDS)를 그대로 재사용한다.
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "experiments" / "cv_pooling"))
import folds  # noqa: E402

CENTER_COL = "center_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

# Day4에서 확정된 order 고정 재사용(재탐색 없음).
ORDER_FIXED = {"A": (1, 0, 1), "B": (0, 1, 0)}
A_HORIZONS = [1, 4, 8]
B_HORIZONS = [1, 4]
# A센터: exog 조합 선택은 2023 안 4-Window Expanding CV(OPTION2_FOLDS)로 하고,
# 2024는 조합 확정 후 딱 한 번만 holdout으로 채점(run_final_holdout_test).
A_CV_WINDOWS = folds.OPTION2_FOLDS
A_HOLDOUT_SPLITS = ["val", "test"]
B_EVAL_SPLITS = ["pool"]

# temp_x_precip은 집계 후 새로 계산하므로 원본 집계 대상에서는 뺀다.
RAW_AGG_CANDIDATES = ["공휴일_W0", "공휴일_W-1", "공휴일_W+1", "평균온도", "총강수량",
                       "강수량_호우_flag", "covid_flag"]

EXOG_CANDIDATES = {
    "A": ["공휴일_W0", "공휴일_W-1", "공휴일_W+1", "평균온도", "총강수량",
          "강수량_호우_flag", "covid_flag", "temp_x_precip"],
    "B": ["공휴일_W0", "공휴일_W-1", "공휴일_W+1", "평균온도", "총강수량",
          "강수량_호우_flag", "covid_flag", "temp_x_precip"],
}

CALENDAR_COLS = ["공휴일_W0", "공휴일_W-1", "공휴일_W+1"]
WEATHER_COLS = ["평균온도", "총강수량", "강수량_호우_flag", "temp_x_precip"]


def select_exog(train_df, candidates, center):
    selected, excluded = [], []
    for col in candidates:
        if train_df[col].nunique() > 1:
            selected.append(col)
        else:
            excluded.append(col)
    for col in excluded:
        print(f"[{center}] {col} excluded: constant in train (nunique<=1)")
    return selected


def aggregate_to_center_week(df: pd.DataFrame) -> pd.DataFrame:
    """(center_id, week_st) 단위로 qty=sum, exog 후보=first() 집계.
    그룹 내 nunique()==1 검증, 어긋나면 즉시 에러."""
    group_keys = [CENTER_COL, WEEK_COL]

    nun = df.groupby(group_keys, observed=True)[RAW_AGG_CANDIDATES].nunique()
    bad_cols = nun.columns[(nun > 1).any(axis=0)].tolist()
    if bad_cols:
        n_bad_groups = int((nun[bad_cols] > 1).any(axis=1).sum())
        raise ValueError(
            f"exog 후보 컬럼이 (center_id, week_st) 그룹 내에서 값이 일치하지 않음: "
            f"{bad_cols} (불일치 그룹 {n_bad_groups}건). 원본 데이터 정합성을 먼저 확인할 것."
        )

    agg_dict = {QTY_COL: "sum", "split": "first"}
    for col in RAW_AGG_CANDIDATES:
        agg_dict[col] = "first"
    agg = (
        df.groupby(group_keys, observed=True)
        .agg(agg_dict)
        .reset_index()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )
    # temp_x_precip은 주간 집계된 평균온도 x 총강수량으로 새로 계산한다.
    agg["temp_x_precip"] = agg["평균온도"] * agg["총강수량"]
    return agg


def build_operational_block(eval_slice: pd.DataFrame, cols: list[str], temp_history: list[float]) -> np.ndarray:
    """실전(operational) 평가용 미래 exog 대체 블록.
    - 공휴일_W0/W-1/W+1: 미리 알 수 있는 정보라 실제 캘린더값 사용.
    - 평균온도: 직전 4주 실측 평균으로 대체.
    - 총강수량 / 강수량_호우_flag / covid_flag: 0으로 고정.
    - temp_x_precip: 대체된 값으로 재계산 -> 항상 0.
    알려진 그룹에 속하지 않는 컬럼이 들어오면 즉시 에러."""
    n = len(eval_slice)
    sub_temp = float(np.mean(temp_history[-4:])) if temp_history else np.nan
    sub_precip = 0.0

    data = {}
    for col in cols:
        if col in CALENDAR_COLS:
            data[col] = eval_slice[col].to_numpy()
        elif col == "평균온도":
            data[col] = np.full(n, sub_temp)
        elif col == "총강수량":
            data[col] = np.full(n, sub_precip)
        elif col == "강수량_호우_flag":
            data[col] = np.zeros(n)
        elif col == "covid_flag":
            data[col] = np.zeros(n)
        elif col == "temp_x_precip":
            data[col] = np.full(n, sub_temp * sub_precip)
        else:
            raise ValueError(f"실전 대체 규칙이 정의되지 않은 exog 컬럼: {col}")
    # exog는 항상 위치 기반 순수 numpy 배열로 주고받는다.
    return pd.DataFrame(data, columns=cols).to_numpy(dtype=float)


def fit_and_eval(train_vals: np.ndarray, eval_vals: np.ndarray, order: tuple, horizons: list[int],
                  train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str], center: str,
                  eval_split_labels: np.ndarray | None = None,
                  capture_detail_cols: list[str] | None = None) -> dict:
    """cols가 빈 리스트면 순수 ARIMA(exog 없음). order 고정, 계수는 train에서 1회만
    추정. rolling-origin 루프는 하나의 Kalman 상태(res)를 공유하며 매 origin마다
    실전/설명적 두 모드로 get_forecast()하고, append()는 항상 실측 exog로만 상태 갱신.

    eval_split_labels를 주면 각 예측 대상 시점의 원래 split(val/test)을 기록해
    out['by_split']에 split별 지표를 추가로 반환한다(루프 자체는 동일).

    capture_detail_cols를 주면 origin/target 주차, y_true/y_pred, error와 지정 컬럼의
    target_week 값을 out['detail']에 담아 반환한다(y_pred는 operational 값 사용)."""
    train_exog = train_df[cols] if cols else None
    res = SARIMAX(
        list(train_vals), exog=train_exog, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    train_res = res  # 계수 해석용 참조 — append()는 새 객체를 반환하므로 이후에도 안 바뀐다.

    max_h = max(horizons)
    n_eval = len(eval_vals)
    last_observed = train_vals[-1] if len(train_vals) else np.nan

    has_temp = "평균온도" in cols
    temp_history = list(train_df["평균온도"].to_numpy()) if has_temp else []

    collected = {
        "operational": {h: {"y_true": [], "y_pred": [], "naive_pred": [], "eval_split": []} for h in horizons},
        "explanatory": {h: {"y_true": [], "y_pred": [], "naive_pred": [], "eval_split": []} for h in horizons},
    }

    # origin_week: 그 시점까지 마지막으로 관측(append)된 주.
    detail_rows: list[dict] = []
    current_origin_week = train_df[WEEK_COL].iloc[-1] if len(train_df) else None

    for i in range(n_eval):
        steps = min(max_h, n_eval - i)

        if cols:
            # 설명적 모드: 실제 미래 exog 그대로 사용
            actual_block = eval_df.iloc[i:i + steps][cols].to_numpy(dtype=float)
            # 실전 모드: 온도/강수 계열은 대체 규칙, 캘린더는 실제값 그대로
            op_block = build_operational_block(eval_df.iloc[i:i + steps], cols, temp_history)
        else:
            actual_block = None
            op_block = None

        fc_explanatory = np.asarray(res.get_forecast(steps=steps, exog=actual_block).predicted_mean)
        fc_operational = np.asarray(res.get_forecast(steps=steps, exog=op_block).predicted_mean)

        for h in horizons:
            idx = i + h - 1
            if idx >= n_eval:
                continue
            # 예측 대상(idx)이 원래 어느 split이었는지 기록(out['by_split']에서 사용).
            split_label = eval_split_labels[idx] if eval_split_labels is not None else None
            for mode, fc in [("explanatory", fc_explanatory), ("operational", fc_operational)]:
                collected[mode][h]["y_true"].append(eval_vals[idx])
                collected[mode][h]["y_pred"].append(fc[h - 1])
                collected[mode][h]["naive_pred"].append(last_observed)
                collected[mode][h]["eval_split"].append(split_label)

            if capture_detail_cols is not None:
                target_row = eval_df.iloc[idx]
                y_pred_h = float(fc_operational[h - 1])
                y_true_h = float(eval_vals[idx])
                detail_row = {
                    "center": center,
                    "eval_split": split_label,
                    "origin_week": current_origin_week,
                    "target_week": target_row[WEEK_COL],
                    "horizon": f"h{h}",
                    "y_true": y_true_h,
                    "y_pred": y_pred_h,
                    "error": y_pred_h - y_true_h,
                    "absolute_error": abs(y_pred_h - y_true_h),
                }
                for c in capture_detail_cols:
                    detail_row[c] = target_row[c]
                detail_rows.append(detail_row)

        # append: 실전/설명적 구분과 무관하게 항상 그 주의 "실측" exog로만 상태 갱신.
        step_exog = eval_df.iloc[[i]][cols].to_numpy(dtype=float) if cols else None
        res = res.append([eval_vals[i]], exog=step_exog, refit=False)
        last_observed = eval_vals[i]
        if has_temp:
            temp_history.append(float(eval_df.iloc[i]["평균온도"]))
        if capture_detail_cols is not None:
            current_origin_week = eval_df.iloc[i][WEEK_COL]

    out = {"model_res": train_res}
    for mode in ["operational", "explanatory"]:
        rows = []
        for h in horizons:
            n = len(collected[mode][h]["y_true"])
            if n == 0:
                rows.append({"center": center, "horizon": f"h{h}", "n": 0, "RMSE": np.nan, "MAE": np.nan,
                             "WAPE": np.nan, "MASE": np.nan, "Bias(%)": np.nan, "MAPE": np.nan})
                continue
            m = compute_metrics(collected[mode][h]["y_true"], collected[mode][h]["y_pred"], collected[mode][h]["naive_pred"])
            mape = compute_mape(collected[mode][h]["y_true"], collected[mode][h]["y_pred"])
            rows.append({"center": center, "horizon": f"h{h}", "n": n,
                         **{k: round(v, 3) for k, v in m.items()}, "MAPE": round(mape, 3)})
        out[mode] = pd.DataFrame(rows)

    # eval_split_labels가 주어졌을 때만 val/test로 나눈 지표를 추가로 계산한다.
    if eval_split_labels is not None:
        by_split_rows = []
        for mode in ["operational", "explanatory"]:
            for h in horizons:
                labels = np.asarray(collected[mode][h]["eval_split"], dtype=object)
                y_true_all = np.asarray(collected[mode][h]["y_true"], dtype=float)
                y_pred_all = np.asarray(collected[mode][h]["y_pred"], dtype=float)
                naive_all = np.asarray(collected[mode][h]["naive_pred"], dtype=float)
                for split_name in pd.unique(labels):
                    mask = labels == split_name
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    m = compute_metrics(y_true_all[mask], y_pred_all[mask], naive_all[mask])
                    mape = compute_mape(y_true_all[mask], y_pred_all[mask])
                    by_split_rows.append({"center": center, "eval_split": split_name, "mode": mode,
                                           "horizon": f"h{h}", "n": n,
                                           **{k: round(v, 3) for k, v in m.items()}, "MAPE": round(mape, 3)})
        out["by_split"] = pd.DataFrame(by_split_rows)

    if capture_detail_cols is not None:
        out["detail"] = pd.DataFrame(detail_rows)
    return out


def to_rows(metrics_df: pd.DataFrame, variant: str, mode: str) -> list[dict]:
    out = metrics_df.copy()
    out.insert(1, "variant", variant)
    out.insert(2, "mode", mode)
    return out.to_dict("records")


def to_split_rows(by_split_df: pd.DataFrame | None, variant: str,
                   keep_mode: str | None = None, mode_label: str | None = None) -> list[dict]:
    """fit_and_eval의 out['by_split']에 variant 라벨을 붙여 펼친다.
    by_split_df가 None이면 빈 리스트를 반환한다. keep_mode/mode_label은 no_exog/
    calendar_only처럼 operational==explanatory인 경우 중복 모드를 한 행으로 접을 때 쓴다
    (full/weather_only는 keep_mode=None으로 둘 다 남긴다)."""
    if by_split_df is None or len(by_split_df) == 0:
        return []
    out = by_split_df.copy()
    if keep_mode is not None:
        out = out[out["mode"] == keep_mode].copy()
        if mode_label is not None:
            out["mode"] = mode_label
    out.insert(1, "variant", variant)
    return out.to_dict("records")


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE(%) — y_true==0인 행은 분모가 0이 되어 집계 대상에서 제외한다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape_bias(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """common.compute_metrics와 동일한 WAPE/Bias(%) 정의만 따로 뗀 버전."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan
    bias = float(np.sum(y_pred - y_true) / denom * 100) if denom > 0 else np.nan
    return wape, bias


def coefficient_rows(res, cols: list[str], train_df: pd.DataFrame, train_vals: np.ndarray, center: str) -> list[dict]:
    """exog 계수만 정리한다. 원 단위(level) 모델이므로 이진 변수 상대효과율은
    coef / train qty 평균 x 100으로 정의한다."""
    params, bse, pvalues = res.params, res.bse, res.pvalues
    ci = res.conf_int()
    ci.columns = ["ci_95_low", "ci_95_high"]
    train_qty_mean = float(train_vals.mean())

    rows = []
    for col in cols:
        if col not in params.index:
            print(f"  ⚠ [{center}] {col} 계수가 params에 없음 — exog 이름 매핑을 확인할 것")
            continue
        coef = float(params[col])
        row = {
            "center": center,
            "variable": col,
            "coef": round(coef, 5),
            "SE": round(float(bse[col]), 5),
            "p_value": round(float(pvalues[col]), 5),
            "ci_95_low": round(float(ci.loc[col, "ci_95_low"]), 5),
            "ci_95_high": round(float(ci.loc[col, "ci_95_high"]), 5),
        }
        is_binary = train_df[col].dropna().isin([0, 1]).all()
        if is_binary:
            row["train_event_count"] = int(train_df[col].sum())
            row["relative_effect_%"] = round(coef / train_qty_mean * 100, 3)
        else:
            row["train_event_count"] = None
            row["relative_effect_%"] = None
        row["note"] = "exploratory 해석 한정(B train 26주, 정식 추론 아님)" if center == "B" else ""
        rows.append(row)
    return rows


def _run_center_core(train_df: pd.DataFrame, eval_df: pd.DataFrame, center: str, order: tuple,
                      horizons: list[int], eval_split_labels: np.ndarray | None = None,
                      capture_detail_cols: list[str] | None = None) -> dict:
    """run_center(B 53주 walk-forward)/run_center_a_cv(A 4-Window CV)/
    run_final_holdout_test(A 2024 holdout)가 공유하는 학습·평가 본체. train_df/eval_df는
    호출부에서 이미 원하는 구간으로 잘라서 넘긴다."""
    train_vals = train_df[QTY_COL].to_numpy(dtype=float)
    eval_vals = eval_df[QTY_COL].to_numpy(dtype=float)

    selected = select_exog(train_df, EXOG_CANDIDATES[center], center)
    print(f"[{center}] select_exog 최종 선택 컬럼: {selected}")

    rows: list[dict] = []
    coef_rows: list[dict] = []
    by_split_rows: list[dict] = []

    print(f"[{center}] 베이스라인(exog 없음) 학습/평가")
    base_out = fit_and_eval(train_vals, eval_vals, order, horizons, train_df, eval_df, cols=[], center=center,
                             eval_split_labels=eval_split_labels)
    rows.extend(to_rows(base_out["explanatory"], "no_exog", "n/a"))
    # exog가 없어 operational==explanatory로 항상 동일 — 기존 통합 결과표와 동일하게 한 행만 남긴다.
    by_split_rows.extend(to_split_rows(base_out.get("by_split"), "no_exog", keep_mode="explanatory", mode_label="n/a"))

    print(f"[{center}] full(select_exog 전체 {len(selected)}개) 학습/평가")
    full_out = fit_and_eval(train_vals, eval_vals, order, horizons, train_df, eval_df, cols=selected, center=center,
                             eval_split_labels=eval_split_labels)
    rows.extend(to_rows(full_out["operational"], "full", "operational"))
    rows.extend(to_rows(full_out["explanatory"], "full", "explanatory"))
    by_split_rows.extend(to_split_rows(full_out.get("by_split"), "full"))
    if selected:
        coef_rows.extend(coefficient_rows(full_out["model_res"], selected, train_df, train_vals, center))

    cal_cols = [c for c in CALENDAR_COLS if c in selected]
    weather_cols = [c for c in WEATHER_COLS if c in selected]

    calendar_detail_df: pd.DataFrame | None = None

    if cal_cols:
        print(f"[{center}] ablation: calendar_only {cal_cols}")
        cal_out = fit_and_eval(train_vals, eval_vals, order, horizons, train_df, eval_df, cols=cal_cols, center=center,
                                eval_split_labels=eval_split_labels, capture_detail_cols=capture_detail_cols)
        # 캘린더 변수는 실전 대체 규칙 대상이 아니라 operational==explanatory로 항상 동일하다.
        rows.extend(to_rows(cal_out["operational"], "calendar_only", "operational(=explanatory)"))
        by_split_rows.extend(to_split_rows(cal_out.get("by_split"), "calendar_only",
                                            keep_mode="operational", mode_label="operational(=explanatory)"))
        calendar_detail_df = cal_out.get("detail")
    else:
        print(f"[{center}] ablation calendar_only 생략 — select_exog에서 캘린더 변수가 선택되지 않음")

    if weather_cols:
        print(f"[{center}] ablation: weather_only {weather_cols}")
        weather_out = fit_and_eval(train_vals, eval_vals, order, horizons, train_df, eval_df, cols=weather_cols, center=center,
                                    eval_split_labels=eval_split_labels)
        rows.extend(to_rows(weather_out["operational"], "weather_only", "operational"))
        rows.extend(to_rows(weather_out["explanatory"], "weather_only", "explanatory"))
        by_split_rows.extend(to_split_rows(weather_out.get("by_split"), "weather_only"))
    else:
        print(f"[{center}] ablation weather_only 생략 — select_exog에서 날씨 변수가 선택되지 않음")

    return {"rows": rows, "coef_rows": coef_rows, "by_split_rows": by_split_rows,
            "calendar_detail_df": calendar_detail_df}


def run_center(agg: pd.DataFrame, center: str, order: tuple, horizons: list[int], eval_splits: list[str]) -> dict:
    """센터의 기존 'split' 컬럼(train/val/test/pool) 기준 단일 연속 rolling-origin 평가.
    B센터 53주 walk-forward 전용 경로다. A센터는 run_center_a_cv/run_final_holdout_test로
    분리되어 이 함수를 쓰지 않는다."""
    print("=" * 80)
    print(f"[센터 {center}] order={order}(Day4 확정값 고정, 재탐색 없음)")

    sub = agg[agg[CENTER_COL] == center].reset_index(drop=True)
    train_df = sub[sub["split"] == "train"].reset_index(drop=True)
    eval_df = sub[sub["split"].isin(eval_splits)].reset_index(drop=True)
    print(f"  train {len(train_df)}주 / eval {len(eval_df)}주(splits={eval_splits})")

    return _run_center_core(train_df, eval_df, center, order, horizons)


def run_center_a_cv(agg: pd.DataFrame, order: tuple, horizons: list[int]) -> dict:
    """A센터 exog 조합(no_exog/calendar_only/weather_only/full) "선택" 전용 — cv_pooling/
    folds.py의 OPTION2_FOLDS(4-Window Expanding CV)를 재사용한다. Window 경계마다
    _run_center_core -> fit_and_eval에서 새로 계수를 적합하고, Window 내부
    rolling-origin은 append(refit=False)로 진행한다. agg를 Window별 날짜 범위로만
    슬라이스하므로 2024 데이터에는 이 함수 안에서 접근할 수 없다."""
    print("=" * 80)
    print(f"[센터 A] order={order}(Day4 확정값 고정) — 새 CV 정책: 2023 안 3개월 분기 "
          f"Expanding CV {len(A_CV_WINDOWS)}-Window (2024는 이 함수에서 전혀 접근하지 않음)")

    sub = agg[agg[CENTER_COL] == "A"].reset_index(drop=True)

    window_rows: list[dict] = []
    for f in A_CV_WINDOWS:
        print(f"  -- Window {f['fold']}: train {f['train_start']}~{f['train_end']} "
              f"-> val {f['val_start']}~{f['val_end']}")
        train_df = sub[(sub[WEEK_COL] >= pd.Timestamp(f["train_start"])) &
                       (sub[WEEK_COL] <= pd.Timestamp(f["train_end"]))].reset_index(drop=True)
        val_df = sub[(sub[WEEK_COL] >= pd.Timestamp(f["val_start"])) &
                     (sub[WEEK_COL] <= pd.Timestamp(f["val_end"]))].reset_index(drop=True)
        out = _run_center_core(train_df, val_df, "A", order, horizons)
        window_rows.extend({"window": f["fold"], **row} for row in out["rows"])

    window_df = pd.DataFrame(window_rows)
    cv_summary = (
        window_df.groupby(["variant", "mode", "horizon"])["WAPE"]
        .agg(WAPE_mean="mean", WAPE_std="std", n_windows="count")
        .reset_index()
        .sort_values(["horizon", "WAPE_mean"])
    )
    print()
    print("[센터 A] CV 요약(4-Window WAPE 평균/표준편차 — exog 조합 선택 기준)")
    print(cv_summary.to_string(index=False))

    return {"window_rows": window_rows, "cv_summary": cv_summary}


def run_final_holdout_test(agg: pd.DataFrame, order: tuple, horizons: list[int]) -> dict:
    """2024 전체(A_HOLDOUT_SPLITS=val+test)를 쓰는 유일한 지점 — 4-Window CV로 조합
    선택이 끝난 뒤 main()에서 별도 호출한다. val->test를 하나의 연속 루프로
    append(refit=False) 진행."""
    print("=" * 80)
    print(f"[센터 A] order={order}(Day4 확정값 고정) — 2024 전체 최종 holdout "
          f"(4-Window CV 조합 선택 완료 후 1회만 채점)")

    sub = agg[agg[CENTER_COL] == "A"].reset_index(drop=True)
    train_df = sub[sub["split"] == "train"].reset_index(drop=True)
    eval_df = sub[sub["split"].isin(A_HOLDOUT_SPLITS)].reset_index(drop=True)
    print(f"  train {len(train_df)}주 / holdout eval {len(eval_df)}주(splits={A_HOLDOUT_SPLITS})")

    return _run_center_core(train_df, eval_df, "A", order, horizons,
                             eval_split_labels=eval_df["split"].to_numpy(),
                             capture_detail_cols=CALENDAR_COLS)


def main():
    print("=" * 80)
    print("[Day5 ARIMAX] feature_table_final.parquet 로드 및 (center_id, week_st) 집계")
    cols_needed = [CENTER_COL, WEEK_COL, QTY_COL, "split"] + RAW_AGG_CANDIDATES
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=cols_needed)
    print(f"  원본 로드 완료: shape={df.shape}")

    agg = aggregate_to_center_week(df)
    print(f"  (center_id, week_st) 집계 완료 -> {len(agg):,}행")

    all_rows: list[dict] = []
    all_coef_rows: list[dict] = []
    all_by_split_rows: list[dict] = []
    calendar_detail_df: pd.DataFrame | None = None

    # B센터: 기존 53주 walk-forward 그대로 유지
    b_out = run_center(agg, "B", ORDER_FIXED["B"], B_HORIZONS, B_EVAL_SPLITS)
    all_rows.extend(b_out["rows"])
    all_coef_rows.extend(b_out["coef_rows"])

    # A센터 1단계: 4-Window CV(2023 안)로 exog 조합 선택
    a_cv_out = run_center_a_cv(agg, ORDER_FIXED["A"], A_HORIZONS)
    print()
    print("=" * 80)
    print("[센터 A] CV 기준 최저 평균 WAPE 조합 (horizon별)")
    for h in sorted(a_cv_out["cv_summary"]["horizon"].unique()):
        cv_h = a_cv_out["cv_summary"][a_cv_out["cv_summary"]["horizon"] == h]
        best = cv_h.loc[cv_h["WAPE_mean"].idxmin()]
        print(f"  [{h}] variant={best['variant']}, mode={best['mode']} "
              f"(CV WAPE mean={best['WAPE_mean']:.3f}%, std={best['WAPE_std']:.3f}%, "
              f"n_windows={int(best['n_windows'])})")

    # A센터 2단계: 조합 확정 후에만 명시적으로 2024 holdout을 1회 채점
    a_out = run_final_holdout_test(agg, ORDER_FIXED["A"], A_HORIZONS)
    all_rows.extend(a_out["rows"])
    all_coef_rows.extend(a_out["coef_rows"])
    all_by_split_rows.extend(a_out["by_split_rows"])
    calendar_detail_df = a_out["calendar_detail_df"]

    all_df = pd.DataFrame(all_rows)
    perf_df = all_df[all_df["variant"].isin(["no_exog", "full"])].reset_index(drop=True)
    ablation_df = all_df[all_df["variant"].isin(["calendar_only", "weather_only", "full"])].reset_index(drop=True)
    coef_df = pd.DataFrame(all_coef_rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    perf_path = RESULT_DIR / "performance_comparison.csv"
    ablation_path = RESULT_DIR / "ablation.csv"
    coef_path = RESULT_DIR / "coefficients.csv"
    perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")
    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("[성능 비교표] ARIMA(exog 없음) vs ARIMAX 실전(operational) vs ARIMAX 설명적(explanatory)")
    print(perf_df.to_string(index=False))
    print(f"  저장 완료 -> {perf_path}")

    print()
    print("=" * 80)
    print("[Ablation 표] 캘린더군만 / 날씨군만 / 전체 변수군(실제 select_exog 결과 기준)")
    print(ablation_df.to_string(index=False))
    print(f"  저장 완료 -> {ablation_path}")

    print()
    print("=" * 80)
    print("[계수 해석표] full 모델(select_exog 최종 선택 변수) 기준")
    print(coef_df.to_string(index=False) if len(coef_df) else "  (선택된 exog 변수 없음)")
    print(f"  저장 완료 -> {coef_path}")
    print("  ※ temp_x_precip은 Day3 상호작용 feature와 동일한 정의(평균온도x총강수량)를 "
          "재사용한 것일 뿐 별도 검증이 아님 — 보조 참고 문구, 메인 결론 아님")

    # --- A센터 4-Window CV(run_center_a_cv) 원본 산출물 저장 ---
    window_detail_df = pd.DataFrame(a_cv_out["window_rows"])
    window_detail_path = RESULT_DIR / "window_detail.csv"
    cv_summary_path = RESULT_DIR / "cv_summary.csv"
    window_detail_df.to_csv(window_detail_path, index=False, encoding="utf-8-sig")
    a_cv_out["cv_summary"].to_csv(cv_summary_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print(f"[A센터 4-Window CV 상세] window_rows {len(window_detail_df):,}행 저장 완료 -> {window_detail_path}")
    print(f"[A센터 4-Window CV 요약] cv_summary {len(a_cv_out['cv_summary']):,}행 저장 완료 -> {cv_summary_path}")

    # --- A센터 val/test 분리 산출 (기존 결과 파일과는 별도로 신규 저장) ---
    # exog 조합 선택은 이미 run_center_a_cv()에서 끝났다 — 여기 val/test는 그 확정
    # 조합을 run_final_holdout_test()가 채점한 2024 holdout 결과에서 재확인용으로
    # 분리해서 보여주기만 한다(추가 학습/재적합 없음).
    by_split_df = pd.DataFrame(all_by_split_rows)
    perf_by_split_df = by_split_df[by_split_df["variant"].isin(["no_exog", "full"])].reset_index(drop=True)
    ablation_by_split_df = by_split_df[by_split_df["variant"].isin(["calendar_only", "weather_only", "full"])].reset_index(drop=True)

    perf_by_split_path = RESULT_DIR / "performance_by_split.csv"
    ablation_by_split_path = RESULT_DIR / "ablation_by_split.csv"
    perf_by_split_df.to_csv(perf_by_split_path, index=False, encoding="utf-8-sig")
    ablation_by_split_df.to_csv(ablation_by_split_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("[A센터 val/test 분리 — 성능 비교표] Val=모델 선택용, Test=최종 검증용")
    print(perf_by_split_df.to_string(index=False))
    print(f"  저장 완료 -> {perf_by_split_path}")

    print()
    print("=" * 80)
    print("[A센터 val/test 분리 — Ablation 표] Val=모델 선택용, Test=최종 검증용")
    print(ablation_by_split_df.to_string(index=False))
    print(f"  저장 완료 -> {ablation_by_split_path}")

    print()
    print("=" * 80)
    print("[요약] A센터 CV(2023, 4-Window) 최저 평균 WAPE 조합 및 2024 holdout Test 성능 확인")
    combo_df = pd.concat([perf_by_split_df, ablation_by_split_df], ignore_index=True).drop_duplicates(
        subset=["eval_split", "variant", "mode", "horizon"]
    )
    for h in sorted(a_cv_out["cv_summary"]["horizon"].unique()):
        cv_h = a_cv_out["cv_summary"][a_cv_out["cv_summary"]["horizon"] == h]
        best = cv_h.loc[cv_h["WAPE_mean"].idxmin()]
        test_match = combo_df[
            (combo_df["eval_split"] == "test") & (combo_df["variant"] == best["variant"])
            & (combo_df["mode"] == best["mode"]) & (combo_df["horizon"] == h)
        ]
        print(f"  [{h}] CV 최저 평균 WAPE 조합: variant={best['variant']}, mode={best['mode']} "
              f"(CV WAPE mean={best['WAPE_mean']:.3f}%, std={best['WAPE_std']:.3f}%)")
        if len(test_match) == 0:
            print(f"        -> 동일 조합의 2024 holdout Test 결과 없음(해당 horizon이 test 구간에 없을 수 있음)")
        else:
            t = test_match.iloc[0]
            delta = t["WAPE"] - best["WAPE_mean"]
            print(f"        -> 2024 holdout Test WAPE={t['WAPE']:.3f}%, Bias={t['Bias(%)']:+.3f}% "
                  f"(CV 평균 대비 {delta:+.3f}%p, {'유지' if abs(delta) <= 5 else '괴리 큼'})")

    # --- calendar_only 모델 rolling-origin 예측 상세 (A센터만) ---
    if calendar_detail_df is None or len(calendar_detail_df) == 0:
        print()
        print("=" * 80)
        print("[A센터 calendar_only 예측 상세] 산출 대상 없음(calendar_only ablation이 생략됨)")
    else:
        detail_path = RESULT_DIR / "calendar_origin_predictions.csv"
        calendar_detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

        print()
        print("=" * 80)
        print(f"[A센터 calendar_only 예측 상세] {len(calendar_detail_df):,}행 저장 완료 -> {detail_path}")

        # h8 기준, val/test 각각 absolute_error 상위 10개
        h8_df = calendar_detail_df[calendar_detail_df["horizon"] == "h8"]
        print()
        print("[A센터 calendar_only h8] eval_split별 absolute_error 상위 10개")
        for split_name in ["val", "test"]:
            top10 = h8_df[h8_df["eval_split"] == split_name].sort_values("absolute_error", ascending=False).head(10)
            print(f"  -- {split_name} (n={len(h8_df[h8_df['eval_split'] == split_name])}) --")
            print(top10.to_string(index=False))

        # target_week 기준 공휴일(W0/W-1/W+1 중 하나라도 1) vs 비공휴일 WAPE/Bias 비교
        is_holiday = (calendar_detail_df[CALENDAR_COLS] == 1).any(axis=1)
        print()
        print("[A센터 calendar_only 전체 horizon] target_week 공휴일 포함 여부별 WAPE/Bias 비교(val/test 각각)")
        for split_name in ["val", "test"]:
            split_mask = calendar_detail_df["eval_split"] == split_name
            for label, group_mask in [("공휴일 포함(W0/W-1/W+1 중 1개 이상=1)", is_holiday), ("비공휴일(셋 다 0)", ~is_holiday)]:
                sub = calendar_detail_df[split_mask & group_mask]
                if len(sub) == 0:
                    print(f"  [{split_name}] {label}: 대상 없음")
                    continue
                w, b = wape_bias(sub["y_true"], sub["y_pred"])
                print(f"  [{split_name}] {label}: n={len(sub)}, WAPE={w:.3f}%, Bias={b:+.3f}%")


if __name__ == "__main__":
    main()
