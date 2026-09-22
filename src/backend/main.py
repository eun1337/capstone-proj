import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, tableau, logistics, dashboard, model_analysis
from services import dashboard_data

# 대시보드 첫 화면(센터 전환 포함)이 실제로 요청하는 기본 조합 — Dashboard.jsx의
# DEFAULT_OPERATIONAL_DATE('2024-09-30')/기본 historyWeeks(4)/기본 topUnit('EA')와
# 정확히 맞춰뒀다. 이 조합과 정확히 일치하는 요청만 lru_cache로 즉시 응답되고, 발표 중
# 다른 날짜/카테고리로 벗어나면 그 조합은 처음 한 번만 정상 속도로 느리다.
_DEMO_CENTERS = ("A", "B")
_DEMO_DATE = "2024-09-30"
_DEMO_PREV_DATE = "2024-09-29"  # daily/summary가 전일 대비 계산용으로 항상 같이 부르는 날짜
_DEMO_WEEK = "2024-09-30"
_DEMO_OPTION_CODE = "EA"
_DEMO_HISTORY_WEEKS = 4


def _warm_endpoints_for_center(center: str) -> None:
    """9개 hot endpoint를 프론트가 첫 화면에서 실제로 부르는 것과 동일한 인자로 직접
    호출해 각 함수의 @lru_cache까지 채운다. 파일 로딩만 캐시하는 것과 달리, groupby/merge
    같은 계산 결과 자체가 이미 준비돼 있어야 발표 중 재클릭이 즉시 반환된다.

    ⚠️ FastAPI 라우팅을 거치지 않고 함수를 직접 호출하므로, `Optional[str] = Query(None)`
    같은 파라미터는 안 넘기면 파이썬 기본값인 Query(...) 객체 그대로 들어간다(None이 아니다).
    endpoint 안에서는 `if category_large:` 식으로 참/거짓만 보는데 Query 객체는 항상
    참이라, 실제로는 없는 카테고리로 필터링돼 결과가 통째로 비어버린다(get_daily_summary는
    404까지 던진다). 그래서 아래는 옵션 파라미터도 전부 명시적으로 값을 채워 호출한다."""
    dashboard.get_categories(center=center)
    dashboard.get_daily_summary(
        center=center, date_=_DEMO_DATE,
        category_large=None, category_middle=None, category_small=None, sku_id=None,
    )
    dashboard.get_daily_summary(
        center=center, date_=_DEMO_PREV_DATE,
        category_large=None, category_middle=None, category_small=None, sku_id=None,
    )
    dashboard.get_daily_category_sales(
        center=center, date_=_DEMO_DATE,
        category_large=None, category_middle=None, category_small=None,
    )
    dashboard.get_daily_region_sales(
        center=center, date_=_DEMO_DATE, sido=None, sigungu=None,
        category_large=None, category_middle=None, category_small=None,
    )
    dashboard.get_forecast_products(
        center=center, week_st=_DEMO_WEEK,
        category_large=None, category_middle=None, category_small=None,
        option_code=_DEMO_OPTION_CODE, top=None,
    )
    dashboard.get_insights_inventory_shortage(
        center=center, week_st=_DEMO_WEEK,
        category_large=None, category_middle=None, category_small=None,
        option_code=_DEMO_OPTION_CODE,
    )
    dashboard.get_daily_sales_surge(
        center=center, date_=_DEMO_DATE,
        category_large=None, category_middle=None, category_small=None,
        option_code=_DEMO_OPTION_CODE,
    )
    dashboard.get_daily_returns(
        center=center, date_=_DEMO_DATE,
        category_large=None, category_middle=None, category_small=None,
    )
    dashboard.get_demand_trend(
        center=center, option_code=_DEMO_OPTION_CODE,
        category_large=None, category_middle=None, category_small=None,
        week_st=_DEMO_WEEK, history_weeks=_DEMO_HISTORY_WEEKS,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # dashboard_data의 load_* 함수들은 각각 @lru_cache로 프로세스당 1회만 읽지만, 그 "처음
    # 1회"가 여러 endpoint 요청이 동시에 몰리는 순간(예: 프론트에서 센터를 바꾸면 카드마다
    # 별도 API를 한 번에 호출) 발생하면 lru_cache는 동시 캐시미스를 막아주지 않아 여러
    # 스레드가 같은 대용량 parquet을 동시에 각자 다시 읽고 변환하는 캐시 스탬피드가 생긴다.
    # (실측: daily/summary는 캐시가 이미 따뜻하면 ~1초지만, 콜드 상태에서 대시보드 진입 시
    # 몰리는 9개 endpoint를 한꺼번에 쏘면 같은 요청이 ~29초까지 늘어짐 — 프론트에서 보고된
    # "Failed to fetch"의 실제 원인이었다.) 서버 기동 시 미리 읽어 캐시를 채워 두면, 이후
    # 무엇이 몰려도 이미 캐시된 DataFrame만 읽으므로 재발하지 않는다.
    #
    # 13개 parquet은 서로 독립적이라 처음엔 asyncio.to_thread로 병렬 로딩을 시도했지만,
    # 이 머신(16코어 NVMe)에서 실측해보니 오히려 더 느렸다(순차 ~52초 vs 병렬 ~69초).
    # load_sales_transactions_raw()가 astype/to_datetime/문자열 concat 같은 무거운 pandas
    # 연산을 포함하는데, 이런 연산은 GIL을 완전히 놓아주지 않아 여러 스레드가 동시에 돌면
    # GIL 경합·메모리 할당자 경합이 병렬 이득보다 커진 것으로 보인다. 그래서 순차 로딩으로
    # 되돌렸다 — "병렬화하면 무조건 빠르다"고 가정하지 말고, 이 프로젝트에서는 이 방식이
    # 실측상 더 낫다.
    t0 = time.perf_counter()
    loaders = [
        dashboard_data.load_product_master,
        dashboard_data.load_forecast_sku_keys,
        dashboard_data.load_weekly_demand,
        dashboard_data.load_forecast_2024,
        dashboard_data.load_weekly_transactions,
        dashboard_data.load_inventory_weekly,
        dashboard_data.load_sales_transactions_raw,
        dashboard_data.load_daily_demand,
        dashboard_data.load_daily_transactions,
        dashboard_data.load_inventory_daily,
        dashboard_data.load_daily_summary_agg,
        dashboard_data.load_daily_region_agg,
        dashboard_data.load_daily_transactions_sparse,
        dashboard_data.load_daily_sku_grid,
        dashboard_data.load_kan_category_order,
    ]
    for fn in loaders:
        await asyncio.to_thread(fn)
    t1 = time.perf_counter()
    print(f"[warmup] parquet 로더 {len(loaders)}개 순차 로딩 완료: {t1 - t0:.1f}s", flush=True)

    # 위 로더들이 전부 캐시된 뒤에야 의미가 있으므로(같은 DataFrame을 재사용) 반드시 이후에
    # 실행한다. 센터별로 9개씩, 역시 서로 독립적이라 병렬로 호출한다.
    await asyncio.gather(*(asyncio.to_thread(_warm_endpoints_for_center, c) for c in _DEMO_CENTERS))
    t2 = time.perf_counter()
    print(f"[warmup] 센터 {len(_DEMO_CENTERS)}개 x hot endpoint 9개 사전 계산 완료: {t2 - t1:.1f}s "
          f"(총 {t2 - t0:.1f}s)", flush=True)
    # 이 프로세스가 오늘 수정된 최신 코드로 뜬 게 맞는지, 예열이 실제로 다 끝났는지를 로그
    # 한 줄로 명확히 구분하기 위한 마커 — 코드를 고친 뒤에도 예전에 띄워둔 프로세스가 계속
    # 떠 있으면(파이썬은 파일을 다시 읽지 않는다) 이 줄 자체가 안 찍히므로, 화면에서 계속
    # 느리다면 이 줄이 안 보이는지부터 확인하면 된다("--reload 없이 실행 = 코드 고치면
    # 반드시 재기동 필요"라는 뜻이 바로 이거다).
    print("[Warmup Complete] A/B센터 데모 캐시 준비 완료 — 지금부터 요청을 받아도 됩니다.", flush=True)
    yield


app = FastAPI(title="물류 수요 예측 API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api", tags=["auth"])
app.include_router(tableau.router,   prefix="/api", tags=["tableau"])
app.include_router(logistics.router, prefix="/api", tags=["logistics"])
app.include_router(dashboard.router,  prefix="/api/dashboard", tags=["dashboard"])
app.include_router(model_analysis.router, prefix="/api/model-analysis", tags=["model-analysis"])


@app.get("/")
def root():
    return {"message": "물류 수요 예측 API 정상 동작 중"}
