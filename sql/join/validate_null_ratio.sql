-- =====================================================================
-- 03_validate_null_ratio.sql
--
-- 목적: 외부 데이터가 매칭되지 않아 발생하는 NULL 비율을 컬럼별/연도별/
--       시도별로 점검. 결측이 "버그"인지 "구조적으로 정상"인지
--       (예: 인천/세종, covid 2024년 이후 범위 밖 등) 구분하기 위함.
-- =====================================================================

-- [검증 1] 전체 기준 컬럼별 NULL 비율
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
    SUM(공휴일 IS NULL)                                       AS holiday_NULL_pct_주의  -- 대부분 NULL이 정상(평일)
FROM main_joined_all;
-- 해석 가이드:
--   - cpi는 항상 NULL 0%여야 함(전국 단일 값, 시도 무관, 전 기간 커버) — 0이 아니면 이상.
--   - covid_영향여부도 항상 NULL 0%여야 함(2019~2024 전 구간 커버) — 0이 아니면 이상.
--   - weather/retail_*의 NULL은 대부분 시도_NULL(우편번호 매칭 실패)에서
--     비롯됨 -> 시도_NULL_pct와 나머지 항목들의 pct가 비슷한 수준이면 정상.
--   - 공휴일은 컬럼 특성상 NULL이 대부분(연중 25~29일만 값이 있음)이므로
--     이 컬럼만 NULL 비율이 높은 건 정상, 다른 컬럼과 같은 잣대로 보지 말 것.


-- [검증 2] 연도별 NULL 비율 (covid 2024년 커버리지, weather/economic 기간 확인용)
SELECT
    년,
    COUNT(*)                                              AS 행수,
    ROUND(100 * SUM(평균온도 IS NULL) / COUNT(*), 2)      AS weather_NULL_pct,
    ROUND(100 * SUM(covid_영향여부 IS NULL) / COUNT(*), 2) AS covid_NULL_pct
FROM main_joined_all
GROUP BY 년
ORDER BY 년;


-- [검증 3] 시도별 NULL 비율 (인천/세종 등 지역 커버리지 갭 확인용)
SELECT
    COALESCE(시도, '(매핑 실패-NULL)') AS 시도,
    COUNT(*)                                          AS 행수,
    ROUND(100 * SUM(평균온도 IS NULL) / COUNT(*), 2)  AS weather_NULL_pct
FROM main_joined_all
GROUP BY 시도
ORDER BY 행수 DESC;
-- '(매핑 실패-NULL)' 행이 있다면 postal_code_region에 없는 우편번호가
-- 메인 데이터에 존재한다는 뜻 -> 검증 4에서 구체적인 우편번호를 확인할 것.


-- [검증 4] 지역 매핑에 실패한(=postal_code_region에 없는) 우편번호 TOP 20
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


-- [검증 5] weather 시(구) 세분화 fallback이 실제로 잘 동작하는지 확인
--          (01_master_join.sql 주석 참고: 창원시/포항시/천안시는 weather엔
--          시 단위로만 있고 postal_code_region엔 구 단위로 세분화되어 있어
--          fallback 없이는 대부분 NULL이 나던 케이스). PK 없이도 정확하도록
--          main_joined_all과 재조인하지 않고, LEFT JOIN을 직접 재현해서 확인.
SELECT
    m.시도, m.시군구,
    COUNT(*)                                     AS 거래건수,
    SUM(w.평균온도 IS NULL)                       AS weather_미매칭건수,
    ROUND(100 * SUM(w.평균온도 IS NULL) / COUNT(*), 1) AS 미매칭_pct
FROM main_with_region m
LEFT JOIN weather_daily w
    ON  w.시도 = m.시도
    AND (
            w.시군구 = m.시군구
         OR w.시군구 IS NULL
         OR m.시군구 LIKE CONCAT(w.시군구, ' %')
        )
    AND w.년 = m.년 AND w.월 = m.월 AND w.일 = m.일
WHERE m.시군구 IN (
    '창원시 마산합포구', '창원시 마산회원구', '창원시 성산구',
    '창원시 의창구', '창원시 진해구',
    '포항시 남구', '포항시 북구',
    '천안시 서북구'
)
GROUP BY m.시도, m.시군구
ORDER BY 미매칭_pct DESC;
-- 기대 결과: weather_미매칭건수 = 0 (fallback이 정상 동작하면 전부 매칭됨).
-- 0이 아니면 fallback 조건(LIKE 패턴)이 실제 문자열과 어긋난 것이니
-- weather_daily/postal_code_region의 시군구 표기를 다시 대조할 것.
