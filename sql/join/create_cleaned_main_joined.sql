-- =====================================================================
-- create_cleaned_main_joined_v1.sql
--
-- 목적: main_joined_all에서 수요예측 분석 대상인 '매출(Sales, 출고)' 거래만
--       남겨 최종 정제 테이블 cleaned_main_joined_v1을 생성.
--
-- 실측 결과 요약(null_ratio_report.sql 재실행 + 우편번호 유형 분석 기준):
--   - 시도_NULL/weather_NULL/retail_*_NULL = 150,756~150,796건(3.65%) 전부
--     지역 매핑 실패(postal_code_region_mapping에 없는 156개 우편번호)에서
--     기인.
--   - 이 156개 우편번호(150,756건)의 거래 유형을 확인한 결과 100% '매입'
--     거래 -> 이 프로젝트의 분석 그레인은 '매출'(출고, 물류센터->소매점)
--     데이터이므로, 매입 행을 애초에 제외하면 이 결측이 전부 함께 빠짐.
--   - 매출 거래는 우편번호 매핑 실패가 0건 -> 매출만 필터링하면 지역 의존
--     외부데이터(weather/retail) 결측이 구조적으로 0%가 됨. 이전 버전에
--     있던 전국평균 대체(imputation) 로직이 더 이상 필요 없어짐.
--   - cpi/covid는 애초에 전 기간 커버라 결측 0%(필터와 무관).
--
-- 적용 규칙:
--   1) 유형 = '매출'인 행만 필터링
--   2) 공휴일: 설날/추석이면 1, 아니면(NULL) 0인 이진 플래그로 인코딩
--      covid_영향여부: COALESCE(..., 0)
--   3) row_id: 트랜잭션 원본에 PK가 없어(문서 docs/join_key_candidates.md
--      참고), 파생변수를
--      행 단위로 다시 merge할 기준이 필요해서 추가한 surrogate key.
--      DROP/재생성마다 값이 새로 매겨지므로, 재생성 이후엔 이전에 내보낸
--      파일들과 row_id가 더 이상 대응되지 않는다는 점 주의.
-- =====================================================================

-- 이전 버전에서 만든 전국평균 보조 테이블은 더 이상 필요 없어 정리
DROP TABLE IF EXISTS phase1_national_weather_avg;
DROP TABLE IF EXISTS phase1_national_retail_current_avg;
DROP TABLE IF EXISTS phase1_national_retail_constant_avg;

DROP TABLE IF EXISTS cleaned_main_joined_v1;

CREATE TABLE cleaned_main_joined_v1 AS
SELECT
    j.센터, j.유형, j.거래일, j.년, j.월, j.일,
    j.우편번호, j.시도, j.시군구,
    j.바코드, j.상품명, j.옵션코드, j.KAN_CODE, j.KAN_대분류, j.KAN_중분류, j.KAN_소분류,
    j.수량, j.금액,

    j.평균온도,
    j.총강수량,

    CASE WHEN j.공휴일 IS NOT NULL THEN 1 ELSE 0 END AS 공휴일,
    COALESCE(j.covid_영향여부, 0)                    AS covid_영향여부,

    j.cpi,

    j.경상지수,
    j.불변지수

FROM main_joined_all j
WHERE j.유형 = '매출';

ALTER TABLE cleaned_main_joined_v1 MODIFY 시도 VARCHAR(20) NULL;
ALTER TABLE cleaned_main_joined_v1 ADD INDEX idx_date (년, 월, 일);
ALTER TABLE cleaned_main_joined_v1 ADD INDEX idx_sido (시도);

-- 행 단위 merge 기준으로
-- 쓸 surrogate key (원본 트랜잭션엔 PK가 없어서 추가)
ALTER TABLE cleaned_main_joined_v1
  ADD COLUMN row_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY FIRST;


-- ---------------------------------------------------------------------
-- 검증1: 행 수 + 컬럼별 잔존 NULL 확인 (전부 0이어야 정상)
-- ---------------------------------------------------------------------
SELECT
    COUNT(*)                                    AS 전체행수,
    SUM(시도 IS NULL)                           AS 시도_잔존NULL,
    SUM(평균온도 IS NULL)                       AS 평균온도_잔존NULL,
    SUM(총강수량 IS NULL)                       AS 총강수량_잔존NULL,
    SUM(cpi IS NULL)                            AS cpi_잔존NULL,
    SUM(경상지수 IS NULL)                       AS 경상지수_잔존NULL,
    SUM(불변지수 IS NULL)                       AS 불변지수_잔존NULL,
    SUM(공휴일 IS NULL)                         AS 공휴일_잔존NULL,
    SUM(covid_영향여부 IS NULL)                 AS covid_잔존NULL
FROM cleaned_main_joined_v1;

-- 검증2: main_joined_all의 매출 행 수와 최종 테이블 행 수가 정확히
--        일치하는지 확인 (필터링 과정에서 행 유실/중복이 없어야 함)
SELECT
    (SELECT COUNT(*) FROM main_joined_all WHERE 유형 = '매출') AS 원본_매출행수,
    (SELECT COUNT(*) FROM cleaned_main_joined_v1)              AS 최종_행수;

-- 검증3: 공휴일 1/0 분포 확인 (1의 개수가 원본 설날+추석 건수와 일치해야 함)
SELECT 공휴일, COUNT(*) AS 건수
FROM cleaned_main_joined_v1
GROUP BY 공휴일;
