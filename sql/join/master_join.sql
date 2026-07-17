-- master_join.sql
--
-- 목적: 메인 거래 데이터(main_with_region)를 기준으로 6종 외부 데이터를
--       전부 LEFT JOIN한 최종 결과를 실제 테이블(main_joined_all)로
--       구체화한다. 메인 데이터는 LEFT JOIN의 기준(driving table)이므로
--       행 손실이 없다.
--
-- 뷰가 아니라 테이블인 이유:
--       처음엔 뷰(vw_main_joined_all)로 만들었는데, EXPLAIN상으로는
--       조인 전부 인덱스를 정상적으로 타는데도(모든 테이블 type=ref/
--       eq_ref/ref_or_null, rows 1~2개) 실제 COUNT(*)가 5분 넘게 걸리며
--       타임아웃 남. 실행계획은 정상이라 쿼리 설계 문제가 아니라 하드웨어/
--       InnoDB 버퍼풀 크기 등 환경 차이(컴퓨터마다 사양이 다름)로 보임.
--       매번 조인을 다시 계산하는 뷰 대신, 한 번 계산해서 실제 테이블로
--       저장해두면 이후의 모든 조회/검증은 컴퓨터 사양과 무관하게 단순
--       테이블 스캔이라 항상 빠르다. 대신 원본 데이터가 바뀌면(재정제 등)
--       이 스크립트를 다시 실행해서 테이블을 갱신해야 한다.
--
-- 조인 키 요약
--   - weather                      : (시도, 시군구, 년, 월, 일)
--   - retail_sales_*               : (시도, 년, 월)  — 시군구 컬럼 자체가 없음
--   - cpi_monthly_national          : (년, 월)  — 전국 단일 값, 지역 무관
--   - public_holidays / covid_impact_daily : 지역 무관, 날짜만으로 매칭
--
-- weather 시군구 매칭 3단계 (실제 데이터 대조 검증함)
--   1) 정확 매칭        : w.시군구 = m.시군구
--   2) 광역시 broadcast : w.시군구 IS NULL (서울/부산/대구/울산이 시도 전체
--                          대표값 1건으로만 존재하는 경우)
--   3) 시(구) 세분화     : m.시군구 LIKE CONCAT(w.시군구, ' %')
--                          -- weather 수집 시점엔 "창원시/포항시/천안시"
--                          -- 단일 지점으로만 취급했는데, 실제
--                          -- postal_code_region_mapping에는 그 아래
--                          -- 구 단위(마산합포구/남구/서북구 등, 총 8종
--                          -- 시군구·887건 중 649건)까지 세분화되어 있어
--                          -- 이 fallback이 없으면 창원/포항/천안 소재
--                          -- 거래 대부분이 NULL로 빠짐. 실 데이터로 확인한
--                          -- 실제 이슈이며, weather엔 시(구) 단위 데이터가
--                          -- 없으므로 상위 시 단위 값을 그대로 적용한다.
--
-- 팬아웃(행 수 증가) 안전성:
--   - weather는 (시도,시군구,년,월,일) 기준 중복행 0건(clean 스크립트 실행
--     로그로 확인됨).
--   - 위 3단계 중 한 시점에 동시에 두 조건 이상이 참이 되는 경우가 없음을
--     실제 데이터로 대조 확인함 (창원/포항/천안은 정확매칭 후보가 아예 없고,
--     서울/부산/대구/울산은 시(구) 세분화 후보가 없음) → 항상 최대 1건 매칭.
--   - retail_sales_*/public_holidays 모두 조인 키 기준 중복행 0건(각 clean
--     스크립트 검증 로그로 확인됨).
--   - cpi_monthly_national/covid_impact_daily는 각각 (년,월)/(년,월,일)
--     UNIQUE 성격의 그레인(1개월 1행 / 1일 1행)이므로 항상 최대 1건 매칭.
--   실제 검증은 sql/join/validate_row_counts.sql 로 반드시 재확인할 것.
-- =====================================================================

DROP TABLE IF EXISTS main_joined_all;

CREATE TABLE main_joined_all AS
SELECT
    m.센터, m.유형, m.거래일, m.년, m.월, m.일,
    m.우편번호, m.시도, m.시군구,
    m.바코드, m.상품명, m.옵션코드, m.KAN_CODE, m.KAN_대분류, m.KAN_중분류, m.KAN_소분류,
    m.수량, m.금액,

    -- weather
    w.평균온도,
    w.총강수량,

    -- holiday (해당 날짜가 설날/추석 연휴가 아니면 NULL = 평일/기타 공휴일)
    h.공휴일,

    -- covid (2019-01-01~2024-12-31 전 구간 존재 → 메인 데이터 기간엔 NULL 없음)
    c.영향여부 AS covid_영향여부,

    -- cpi (전국 단일 값, 지역 무관 — 메인 데이터 기간 전 구간 커버라 NULL 없음)
    cpi.소비자물가지수 AS cpi,

    -- retail sales index
    rsc.경상지수,
    rsk.불변지수

FROM main_with_region m

LEFT JOIN weather_daily w
    ON  w.시도 = m.시도
    AND (
            w.시군구 = m.시군구
         OR w.시군구 IS NULL
         OR m.시군구 LIKE CONCAT(w.시군구, ' %')
        )
    AND w.년 = m.년 AND w.월 = m.월 AND w.일 = m.일

LEFT JOIN public_holidays h
    ON h.년 = m.년 AND h.월 = m.월 AND h.일 = m.일

LEFT JOIN covid_impact_daily c
    ON c.년 = m.년 AND c.월 = m.월 AND c.일 = m.일

LEFT JOIN cpi_monthly_national cpi
    ON cpi.년 = m.년 AND cpi.월 = m.월

LEFT JOIN retail_sales_current rsc
    ON rsc.시도 = m.시도 AND rsc.년 = m.년 AND rsc.월 = m.월

LEFT JOIN retail_sales_constant rsk
    ON rsk.시도 = m.시도 AND rsk.년 = m.년 AND rsk.월 = m.월;

-- 분석/검증 단계에서 WHERE·GROUP BY로 자주 걸릴 컬럼 인덱스 (선택이지만 권장)
ALTER TABLE main_joined_all ADD INDEX idx_date (년, 월, 일);
-- CREATE TABLE ... AS SELECT로 만들면 시도가 VARCHAR가 아니라 TEXT로 생성되어
-- 인덱스에 키 길이 지정이 필요해짐 -> 인덱스 걸기 전에 타입부터 명시 (NULL
-- 허용: postal_code_region에 없는 우편번호는 LEFT JOIN 결과 시도가 NULL이 됨)
ALTER TABLE main_joined_all MODIFY 시도 VARCHAR(20) NULL;
ALTER TABLE main_joined_all ADD INDEX idx_sido (시도);
