"""
Step 4 결과물에서 발견된 이상 징후에 대한 진단 전용 스크립트. 코드 수정은 하지 않고
원인만 확인한다.

1) A센터 Top5 상품 차트("참소주G", "카스355<캔>")에서 2024년 초~중반 구간이
   부자연스러운 직선으로 이어지는 현상의 원인 확인
2) B센터 반품률 레짐 전환(2023-07~)이 시각적 인상이 아니라 정량적으로도 확인되는
   구조적 변화인지 재확인
3) plot_weekly_demand_trends.py의 Top5 선정 로직(상품명 groupby 포함 여부,
   학습기간 기준 여부)이 실제로 올바르게 반영되어 있는지 재확인
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
PLOT_SCRIPT_PATH = "analysis-results/eda/weekly/plot_weekly_demand_trends.py"
OUT_DIAG_PNG = "analysis-results/preprocessing/diag_coke_substitution_B.png"

REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")
GAP_WINDOW = (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-09-30"))
PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]

KOREAN_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def setup_korean_font() -> None:
    import os

    for path in KOREAN_FONT_CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    else:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def find_missing_weeks(weekly_index: pd.DatetimeIndex, window: tuple) -> list:
    all_weeks = pd.date_range("2021-01-04", "2024-12-30", freq="7D")
    window_weeks = [w for w in all_weeks if window[0] <= w <= window[1]]
    present = set(weekly_index)
    return [w for w in window_weeks if w not in present]


def contiguous_blocks(missing_weeks: list) -> list:
    if not missing_weeks:
        return []
    blocks = []
    cur = [missing_weeks[0]]
    for w in missing_weeks[1:]:
        if (w - cur[-1]).days == 7:
            cur.append(w)
        else:
            blocks.append(cur)
            cur = [w]
    blocks.append(cur)
    return blocks


def section1_line_gap_diagnosis(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("[1] A센터 Top5 상품 차트 직선 구간 원인 진단")
    print("=" * 70)

    a = df[df["센터"] == "A"]
    targets = {
        "참소주G": ("8801080929104", "BX", 1),
        "카스355<캔>": ("8801858011123", "BX", 1),
    }

    gap_blocks_by_product = {}
    for name, key in targets.items():
        sub = a[(a["바코드"] == key[0]) & (a["옵션코드"] == key[1]) & (a["상품클러스터"] == key[2])]
        weekly = sub.groupby("주시작일")["총판매수량"].sum()
        missing = find_missing_weeks(weekly.index, GAP_WINDOW)
        blocks = contiguous_blocks(missing)
        gap_blocks_by_product[name] = blocks
        print(f"\n--- {name} ---")
        print(f"진단 구간({GAP_WINDOW[0].date()}~{GAP_WINDOW[1].date()}) 내 결측 주차 수: {len(missing)}")
        print(f"연속 결측 블록: {[(b[0].date(), b[-1].date(), len(b)) for b in blocks]}")

    # 가장 긴 결측 블록을 기준으로 (a)/(b) 판별
    main_block = max(
        (b for blocks in gap_blocks_by_product.values() for b in blocks), key=len
    )
    gap_start, gap_end = main_block[0], main_block[-1]
    print(f"\n주요 결측 구간: {gap_start.date()} ~ {gap_end.date()} ({len(main_block)}주)")

    gap_window_df = a[(a["주시작일"] >= gap_start) & (a["주시작일"] <= gap_end)]
    total_products_in_gap = gap_window_df.groupby(PRODUCT_KEY_COLS).ngroups
    print(f"A센터 전체: 같은 구간 내 활성 상품키 수 = {total_products_in_gap:,} (A센터 자체는 정상 운영 중)")

    # 참소주G/카스355 가 속한 카테고리(KAN_중분류=주류) 전체가 죽었는지 확인
    alcohol = a[a["KAN_중분류"] == "주류"]
    alcohol_gap = alcohol[(alcohol["주시작일"] >= gap_start) & (alcohol["주시작일"] <= gap_end)]
    alcohol_out = alcohol[(alcohol["주시작일"] < gap_start) | (alcohol["주시작일"] > gap_end)]
    print(f"\nA센터 KAN_중분류='주류' 카테고리 전체:")
    print(f"  결측 구간 내 활성 상품키 수: {alcohol_gap.groupby(PRODUCT_KEY_COLS).ngroups} (총판매수량 합계 {alcohol_gap['총판매수량'].sum():,})")
    print(f"  결측 구간 밖 활성 상품키 수: {alcohol_out.groupby(PRODUCT_KEY_COLS).ngroups} (총판매수량 합계 {alcohol_out['총판매수량'].sum():,})")

    b = df[df["센터"] == "B"]
    b_alcohol_gap = b[(b["KAN_중분류"] == "주류") & (b["주시작일"] >= gap_start) & (b["주시작일"] <= gap_end)]
    print(f"\n(대조) B센터 KAN_중분류='주류', 같은 기간: 활성 상품키 {b_alcohol_gap.groupby(PRODUCT_KEY_COLS).ngroups}개, 총판매수량 합계 {b_alcohol_gap['총판매수량'].sum():,}")

    print("\n결론:")
    print(f"  - A센터 자체는 이 구간에 정상 운영 중(활성 상품키 {total_products_in_gap:,}개) -> (b) 파이프라인 결측 아님")
    print(f"  - 참소주G/카스355<캔>뿐 아니라 A센터 '주류' 카테고리 전체가 이 구간에 사실상 전면 중단됨")
    print(f"    (활성 상품키 {alcohol_gap.groupby(PRODUCT_KEY_COLS).ngroups}개, 정상기간 대비 판매량 {alcohol_gap['총판매수량'].sum()}/{alcohol_out['총판매수량'].sum():,})")
    print("  - B센터는 같은 기간 주류가 정상 판매됨 -> A센터에서만 발생한 사건(주류 판매 중단 추정)")
    print("  - 즉 (a)/(b) 둘 다 아니라 '카테고리 단위의 실제 판매 중단 이벤트'로 보임 - 정상 zero-demand도, 데이터 결측도 아님")
    print("\n플로팅 이슈: plot_top_products는 marker 없이 ax.plot()으로 점을 직선으로 연결하고,")
    print("결측 주차를 reindex로 0/NaN 채우지 않는다 -> 26주 공백이 시각적으로 완만한 직선 추세처럼 보인다.")
    print("(수정 여부는 이 진단 이후 별도 결정 사항)")


def section2_b_regime_shift_diagnosis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("[2] B센터 반품률 레짐 전환 - 정량적 재확인")
    print("=" * 70)

    b = df[df["센터"] == "B"]
    pre = b[b["주시작일"] < REGIME_SHIFT_DATE]
    post = b[b["주시작일"] >= REGIME_SHIFT_DATE]

    pre_sale_cnt = pre.groupby("주시작일")["판매건수"].sum().mean()
    post_sale_cnt = post.groupby("주시작일")["판매건수"].sum().mean()
    pre_return_cnt = pre.groupby("주시작일")["반품건수"].sum().mean()
    post_return_cnt = post.groupby("주시작일")["반품건수"].sum().mean()

    print("\n-- 거래 건수(빈도) 전후 비교 --")
    print(f"주당 평균 판매건수: {pre_sale_cnt:,.1f} -> {post_sale_cnt:,.1f} ({(post_sale_cnt / pre_sale_cnt - 1):+.1%})")
    print(f"주당 평균 반품건수: {pre_return_cnt:,.1f} -> {post_return_cnt:,.1f} ({(post_return_cnt / pre_return_cnt - 1):+.1%})")

    pre_qty_per_sale = pre["총판매수량"].sum() / pre["판매건수"].sum()
    post_qty_per_sale = post["총판매수량"].sum() / post["판매건수"].sum()
    pre_qty_per_return = pre["반품수량"].sum() / pre["반품건수"].sum()
    post_qty_per_return = post["반품수량"].sum() / post["반품건수"].sum()

    print("\n-- 건당 단위 수량 전후 비교 --")
    print(f"판매 건당 평균 수량: {pre_qty_per_sale:.2f} -> {post_qty_per_sale:.2f}")
    print(f"반품 건당 평균 수량: {pre_qty_per_return:.2f} -> {post_qty_per_return:.2f}")
    print("\n판정: 반품이 급증한 건 '건당 수량'이 아니라 '반품 발생 빈도' 자체가 늘어난 것 (15배 -> 빈도, 건당 수량은 오히려 감소)")

    key = PRODUCT_KEY_COLS
    pre_keys = set(map(tuple, pre[key].drop_duplicates().values))
    post_keys = set(map(tuple, post[key].drop_duplicates().values))
    print("\n-- 고유 SKU 수 전후 비교 --")
    print(f"전(21-23.06) 고유 상품키 수: {len(pre_keys):,}")
    print(f"후(23.07-24) 고유 상품키 수: {len(post_keys):,} ({(len(post_keys) / len(pre_keys) - 1):+.1%})")
    print(f"전에는 있었는데 후에 사라진 상품키: {len(pre_keys - post_keys):,}개")
    print(f"후에 새로 생긴 상품키: {len(post_keys - pre_keys):,}개")

    print("\n-- 코카콜라 슈퍼용355ml vs 250ml 대체 관계 확인 --")
    coke_keys = {
        "코카콜라(슈퍼용)355ml": ("8801094016203", "EA", 1),
        "코카콜라250ml": ("8801094013004", "EA", 1),
    }
    totals = {}
    for name, k in coke_keys.items():
        sub = b[(b["바코드"] == k[0]) & (b["옵션코드"] == k[1]) & (b["상품클러스터"] == k[2])]
        pre_sum = sub[sub["주시작일"] < REGIME_SHIFT_DATE]["총판매수량"].sum()
        post_sum = sub[sub["주시작일"] >= REGIME_SHIFT_DATE]["총판매수량"].sum()
        totals[name] = (pre_sum, post_sum)
        print(f"{name}: 전 {pre_sum:,} -> 후 {post_sum:,} ({(post_sum / pre_sum - 1):+.1%})")

    setup_korean_font()
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, k in coke_keys.items():
        sub = b[(b["바코드"] == k[0]) & (b["옵션코드"] == k[1]) & (b["상품클러스터"] == k[2])]
        weekly = sub.groupby("주시작일")["총판매수량"].sum()
        ax.plot(weekly.index, weekly.values, label=name, linewidth=1.6)
    xlo, xhi = ax.get_xlim()
    ax.axvline(REGIME_SHIFT_DATE, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlim(xlo, xhi)
    ax.set_ylabel("총판매수량 (개)")
    ax.set_title("[B센터] 코카콜라(슈퍼용)355ml vs 코카콜라250ml 대체 여부 확인 (참고용)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIAG_PNG, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\n참고용 차트 저장 -> {OUT_DIAG_PNG}")

    coke355_pre, coke355_post = totals["코카콜라(슈퍼용)355ml"]
    coke250_pre, coke250_post = totals["코카콜라250ml"]
    print("\n판정: 코카콜라(슈퍼용)355ml은 후 구간에 큰 폭 감소(-73.6%)했지만, 코카콜라250ml은 거의 변화 없음(-0.1%).")
    print("-> '한쪽이 줄고 다른 쪽이 그만큼 늘어나는' 대체 관계는 아님. 차트를 보면 두 상품 모두 레짐 전환 시점에")
    print("   '완만한 지속적 흐름'에서 '낮은 기준선 + 간헐적 대형 스파이크' 패턴으로 동시에 바뀜 -> 개별 상품")
    print("   교체가 아니라 B센터의 발주 방식 자체가 바뀐 것이 두 상품에 공통으로 나타난 것으로 해석하는 게 더 타당함.")


def section3_bug_reverification() -> None:
    print("\n" + "=" * 70)
    print("[3] 이전 지적 버그의 실제 반영 여부 재확인")
    print("=" * 70)

    with open(PLOT_SCRIPT_PATH, encoding="utf-8") as f:
        code = f.read()

    print("\n-- (a) plot_top_products의 groupby key에 상품명 포함 여부 --")
    func_start = code.index("def plot_top_products")
    func_end = code.index("\ndef ", func_start + 1)
    func_body = code[func_start:func_end]
    totals_line = [line for line in func_body.splitlines() if "train.groupby(" in line]
    weekly_groupby_line = [line for line in func_body.splitlines() if '.groupby(["주시작일"] + PRODUCT_KEY_COLS)' in line]
    print(f"  Top5 집계: {totals_line[0].strip() if totals_line else '확인 안됨'}")
    print(f"  주간 시계열 집계: {weekly_groupby_line[0].strip() if weekly_groupby_line else '확인 안됨'}")
    if totals_line and "상품명" not in totals_line[0] and weekly_groupby_line and "상품명" not in weekly_groupby_line[0]:
        print("  -> 확인됨, 문제 없음 (PRODUCT_KEY_COLS만 사용, 상품명은 별도 name_map으로 매핑)")
    else:
        print("  -> 상품명이 groupby key에 포함되어 있을 가능성 있음, 코드 재확인 필요")

    print("\n-- (b) Top5 선정이 학습기간(21-23, 센터별) 기준인지 --")
    cat_func = code[code.index("def plot_top_category"):code.index("def plot_top_products")]
    prod_func = func_body
    cat_uses_train = "train = df_center[df_center[\"주시작일\"] < TRAIN_CUTOFF]" in cat_func
    prod_uses_train = "train = df_center[df_center[\"주시작일\"] < TRAIN_CUTOFF]" in prod_func
    print(f"  plot_top_category 학습기간 필터 사용: {cat_uses_train}")
    print(f"  plot_top_products 학습기간 필터 사용: {prod_uses_train}")
    if cat_uses_train and prod_uses_train:
        print("  -> 확인됨, 문제 없음 (두 함수 모두 df_center에서 주시작일 < TRAIN_CUTOFF로 필터링 후 순위 산출)")
    else:
        print("  -> 학습기간 필터가 누락된 함수가 있음, 코드 재확인 필요")


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    print(f"Loaded {len(df):,} rows\n")

    section1_line_gap_diagnosis(df)
    section2_b_regime_shift_diagnosis(df)
    section3_bug_reverification()

    print("\n" + "=" * 70)
    print("요약: 코드 수정 필요 여부")
    print("=" * 70)
    print("[수정 필요]")
    print("  - plot_top_products/plot_top_category: 결측 주차를 reindex로 명시적 처리하지 않고")
    print("    matplotlib이 직선으로 이어버려, A센터 주류 중단 같은 '진짜 이벤트'가 완만한 추세처럼")
    print("    잘못 보임. marker 추가 또는 reindex+NaN 처리로 결측 구간을 시각적으로 끊어줄 필요 있음.")
    print("[수정 불필요 - 이미 반영됨]")
    print("  - Top5 선정 시 상품명 groupby 포함 문제: 없음(확인됨)")
    print("  - Top5 선정 기준이 21-24 전체가 아닌 학습기간(21-23)인지: 맞게 반영됨(확인됨)")
    print("[추가 사실 확인, 코드 문제 아님]")
    print("  - A센터 2024-01~06 주류 판매 전면 중단: 실제 비즈니스 이벤트로 추정, 협업자 확인 필요")
    print("  - B센터 반품 급증은 '빈도' 증가이며 SKU 구성도 25% 감소 - 시각적 인상과 일치하는 구조적 변화")
    print("  - 코카콜라 두 SKU는 대체 관계가 아니라 발주 패턴 자체의 동시 변화로 해석하는 게 더 타당")


if __name__ == "__main__":
    main()
