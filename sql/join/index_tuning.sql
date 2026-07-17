-- =====================================================================
-- index_tuning.sql
--
-- 목적: 외부 데이터 조인 병목 해결
--   1) 조인 키 복합 인덱스 추가
--      (weather_daily/cpi_monthly_national 등은 load_external_to_mysql.py
--       적재 시 이미 대부분 생성됨 -> INFORMATION_SCHEMA로 존재 여부 확인 후
--       동적 SQL(PREPARE/EXECUTE)로 생성해 중복 생성 에러 없이 재실행 가능하게
--       작성. CREATE INDEX ... IF NOT EXISTS는 MySQL 8.0.29+ 전용이라 그보다
--       낮은 버전에서 1064 문법 오류가 나서 이 방식으로 대체함)
--   2) weather 조인의 OR/LIKE fallback이 main_with_region(4,127,312행) 전체를
--      대상으로 매 행마다 평가되는 게 병목이므로, postal_code_region 고유
--      (시도,시군구) 조합(887건) 단위에서 미리 1회만 계산해두는 "지역 키
--      리졸버" 테이블을 만들고, 본 조인은 리졸버가 만든 키로 순수
--      등가조인(=)만 하도록 재작성.
--
-- 실행 전제: sql/grain/*.sql, sql/join/master_join.sql이 이미 실행되어
--           main_with_region, weather_daily 등이 존재.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0) 사전 정리: TEXT로 적재된 시도/시군구를 VARCHAR로 변환
--    postal_code_region은 load_external_to_mysql.py의 ALTER문 목록에
--    시도/시군구 MODIFY가 빠져 있어 TEXT로 남아있고(우편번호만 CHAR(5)
--    처리됨), main_with_region은 그 postal_code_region을 LEFT JOIN해서
--    CREATE TABLE ... AS SELECT로 만든 테이블이라 TEXT를 그대로 물려받음.
--    TEXT/BLOB 컬럼은 키 길이 지정 없이 인덱스를 걸 수 없어(에러 1170)
--    인덱스 생성 전에 먼저 VARCHAR로 고정해야 함
--    (master_join.sql이 main_joined_all에 대해 이미 쓰던 것과 동일 패턴).
-- ---------------------------------------------------------------------
ALTER TABLE postal_code_region MODIFY 시도   VARCHAR(20) NOT NULL;
ALTER TABLE postal_code_region MODIFY 시군구 VARCHAR(20) NOT NULL;

ALTER TABLE main_with_region   MODIFY 시도   VARCHAR(20) NULL;
ALTER TABLE main_with_region   MODIFY 시군구 VARCHAR(20) NULL;


-- ---------------------------------------------------------------------
-- 1) 조인 키 복합 인덱스
--    INFORMATION_SCHEMA로 존재 여부를 확인한 뒤에만 생성하는 헬퍼
--    프로시저를 통해 이미 인덱스가 있어도 에러 없이 재실행 가능하게 처리.
-- ---------------------------------------------------------------------
DROP PROCEDURE IF EXISTS phase1_add_index_if_not_exists;

DELIMITER $$
CREATE PROCEDURE phase1_add_index_if_not_exists(
    IN p_table VARCHAR(64),
    IN p_index VARCHAR(64),
    IN p_cols  VARCHAR(255)
)
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.STATISTICS
        WHERE table_schema = DATABASE()
          AND table_name = p_table
          AND index_name = p_index
    ) THEN
        SET @phase1_sql = CONCAT('CREATE INDEX ', p_index, ' ON ', p_table, ' (', p_cols, ')');
        PREPARE phase1_stmt FROM @phase1_sql;
        EXECUTE phase1_stmt;
        DEALLOCATE PREPARE phase1_stmt;
    END IF;
END$$
DELIMITER ;

CALL phase1_add_index_if_not_exists('weather_daily',         'idx_region_date',  '시도, 시군구, 년, 월, 일');
CALL phase1_add_index_if_not_exists('weather_daily',         'idx_date_first',   '시도, 년, 월, 일, 시군구');
CALL phase1_add_index_if_not_exists('cpi_monthly_national',  'idx_month',        '년, 월');
CALL phase1_add_index_if_not_exists('retail_sales_current',  'idx_region_month', '시도, 년, 월');
CALL phase1_add_index_if_not_exists('retail_sales_constant', 'idx_region_month', '시도, 년, 월');
CALL phase1_add_index_if_not_exists('public_holidays',       'idx_date',         '년, 월, 일');
CALL phase1_add_index_if_not_exists('covid_impact_daily',    'idx_date',         '년, 월, 일');
CALL phase1_add_index_if_not_exists('postal_code_region',    'idx_sido_sigungu', '시도, 시군구');

-- main_with_region: 리졸버 테이블 및 날짜 필터용 (resolve_region.sql에 이미
-- 주석으로 제안돼 있던 인덱스와 동일)
CALL phase1_add_index_if_not_exists('main_with_region',      'idx_region_date',  '시도, 시군구, 년, 월, 일');

DROP PROCEDURE IF EXISTS phase1_add_index_if_not_exists;


-- ---------------------------------------------------------------------
-- 2) 지역 키 리졸버 테이블
--    postal_code_region의 고유 (시도,시군구) 조합(887건) 각각에 대해,
--    weather_daily 쪽에서 실제로 매칭될 시군구 값을 미리 계산해서 저장.
--    main_joined_all 재생성 시 OR/LIKE 대신 이 값과 등가조인(<=>)만 하면 됨.
-- ---------------------------------------------------------------------

-- weather: 정확매칭 -> 광역시 broadcast(NULL) -> 시(구) 세분화 prefix, 3단계
-- (master_join.sql의 실제 조인조건: w.시군구=m.시군구 OR w.시군구 IS NULL
--  OR m.시군구 LIKE CONCAT(w.시군구,' %') 를 그대로 반영)
DROP TABLE IF EXISTS region_weather_key;
CREATE TABLE region_weather_key AS
SELECT
    pr.시도,
    pr.시군구,
    CASE
        WHEN ex.시군구 IS NOT NULL THEN ex.시군구   -- 1) 정확매칭
        WHEN bd.시도   IS NOT NULL THEN NULL         -- 2) 광역시 broadcast
        ELSE pf.시군구                                -- 3) 시(구) 세분화 fallback
    END AS weather_시군구_key
FROM (SELECT DISTINCT 시도, 시군구 FROM postal_code_region) pr
LEFT JOIN (SELECT DISTINCT 시도, 시군구 FROM weather_daily WHERE 시군구 IS NOT NULL) ex
    ON ex.시도 = pr.시도 AND ex.시군구 = pr.시군구
LEFT JOIN (SELECT DISTINCT 시도 FROM weather_daily WHERE 시군구 IS NULL) bd
    ON bd.시도 = pr.시도 AND ex.시군구 IS NULL
LEFT JOIN (SELECT DISTINCT 시도, 시군구 FROM weather_daily WHERE 시군구 IS NOT NULL) pf
    ON pf.시도 = pr.시도
   AND pr.시군구 LIKE CONCAT(pf.시군구, ' %')
   AND ex.시군구 IS NULL;

-- CASE로 만든 파생 컬럼도 TEXT로 잡힐 수 있어 PK 걸기 전에 동일하게 고정
ALTER TABLE region_weather_key MODIFY 시도             VARCHAR(20) NOT NULL;
ALTER TABLE region_weather_key MODIFY 시군구           VARCHAR(20) NOT NULL;
ALTER TABLE region_weather_key MODIFY weather_시군구_key VARCHAR(20) NULL;
ALTER TABLE region_weather_key ADD PRIMARY KEY (시도, 시군구);


-- ---------------------------------------------------------------------
-- 3) main_joined_all 재생성 — weather 조인만 리졸버 기반 등가조인으로
--    교체, 나머지는 sql/join/master_join.sql과 동일. 결과셋은 기존 로직과
--    논리적으로 동일하므로 sql/join/validate_row_counts.sql,
--    validate_null_ratio.sql로 재검증 권장.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS main_joined_all;

CREATE TABLE main_joined_all AS
SELECT
    m.센터, m.유형, m.거래일, m.년, m.월, m.일,
    m.우편번호, m.시도, m.시군구,
    m.바코드, m.상품명, m.KAN_대분류, m.KAN_중분류, m.KAN_소분류,
    m.수량, m.금액,

    w.평균온도,
    w.총강수량,

    h.공휴일,

    c.영향여부 AS covid_영향여부,

    cpi.소비자물가지수 AS cpi,

    rsc.경상지수,
    rsk.불변지수

FROM main_with_region m

LEFT JOIN region_weather_key rwk
    ON rwk.시도 = m.시도 AND rwk.시군구 = m.시군구

LEFT JOIN weather_daily w
    ON  w.시도 = m.시도
    AND w.시군구 <=> rwk.weather_시군구_key
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

-- CREATE TABLE ... AS SELECT는 시도를 TEXT로 만들어 인덱스에 키 길이 지정이
-- 필요해짐 -> 인덱스 걸기 전에 타입부터 명시 (master_join.sql과 동일 패턴)
ALTER TABLE main_joined_all MODIFY 시도 VARCHAR(20) NULL;
ALTER TABLE main_joined_all ADD INDEX idx_date (년, 월, 일);
ALTER TABLE main_joined_all ADD INDEX idx_sido (시도);
