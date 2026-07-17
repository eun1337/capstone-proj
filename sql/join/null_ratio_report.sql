-- =====================================================================
-- null_ratio_report.sql
--
-- 목적: 01_index_tuning.sql로 재생성된 main_joined_all 기준, 외부 데이터가
--       매칭되지 않아 발생하는 컬럼별 NULL 비율을 화면에 보고.
--       (sql/join/validate_null_ratio.sql 검증1~4와 동일한 관점, phase1
--        산출물 기준으로 재실행)
-- =====================================================================

-- [리포트 1] 전체 기준 컬럼별 NULL 비율
SELECT
    COUNT(*)                                                  AS 전체행수,
    SUM(시도 IS NULL)                                         AS 시도_NULL,
    ROUND(100 * SUM(시도 IS NULL) / COUNT(*), 2)              AS 시도_NULL_pct,
    SUM(평균온도 IS NULL)                                     AS weather_NULL,
    ROUND(100 * SUM(평균온도 IS NULL) / COUNT(*), 2)          AS weather_NULL_pct,
    SUM(covid_영향여부 IS NULL)                                AS covid_NULL,
    ROUND(100 * SUM(covid_영향여부 IS NULL) / COUNT(*), 2)     AS covid_NULL_pct,
    SUM(cpi IS NULL)                                          AS cpi_NULL,
    SUM(경상지수 IS NULL)                                     AS retail_current_NULL,
    ROUND(100 * SUM(경상지수 IS NULL) / COUNT(*), 2)          AS retail_current_NULL_pct,
    SUM(불변지수 IS NULL)                                     AS retail_constant_NULL,
    SUM(공휴일 IS NULL)                                       AS holiday_NULL_주의  -- 대부분 NULL이 정상(평일)
FROM main_joined_all;
-- 해석 가이드:
--   - cpi는 항상 NULL 0%여야 함(전국 단일 값, 시도 무관, 전 기간 커버) — 0이 아니면 이상.
--   - covid_영향여부도 항상 NULL 0%여야 함(전 기간 커버) — 0이 아니면 이상.
--   - weather/retail_*의 NULL은 대부분 시도_NULL(우편번호 매칭 실패)에서
--     비롯됨 -> 시도_NULL_pct와 비슷한 수준이면 정상(인천/세종 등 매핑 미커버 지역).
--   - 공휴일은 컬럼 특성상 NULL이 대부분(연중 25~29일만 값 존재)이므로 다른
--     컬럼과 같은 잣대로 보지 말 것.


-- [리포트 2] 연도별 NULL 비율 (covid/weather 기간 커버리지 확인)
SELECT
    년,
    COUNT(*)                                              AS 행수,
    ROUND(100 * SUM(평균온도 IS NULL) / COUNT(*), 2)      AS weather_NULL_pct,
    ROUND(100 * SUM(covid_영향여부 IS NULL) / COUNT(*), 2) AS covid_NULL_pct
FROM main_joined_all
GROUP BY 년
ORDER BY 년;


-- [리포트 3] 시도별 NULL 비율 (인천/세종 등 커버리지 갭 확인)
SELECT
    COALESCE(시도, '(매핑 실패-NULL)') AS 시도,
    COUNT(*)                                          AS 행수,
    ROUND(100 * SUM(평균온도 IS NULL) / COUNT(*), 2)  AS weather_NULL_pct
FROM main_joined_all
GROUP BY 시도
ORDER BY 행수 DESC;
-- '(매핑 실패-NULL)' 행이 있으면 postal_code_region에 없는 우편번호가
-- 메인 데이터에 존재한다는 뜻 -> 리포트4에서 구체적인 우편번호 확인.


-- [리포트 4] 지역 매핑에 실패한(=postal_code_region에 없는) 우편번호 TOP 20
SELECT
    m.우편번호,
    COUNT(*) AS 거래건수
FROM main_with_region m
WHERE m.시도 IS NULL
  AND m.우편번호 IS NOT NULL
  AND m.우편번호 <> ''
GROUP BY m.우편번호
ORDER BY 거래건수 DESC
LIMIT 20;
