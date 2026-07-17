-- =====================================================================
-- resolve_region.sql
--
-- 목적: vw_main_transactions(6개 테이블 UNION ALL 뷰)의 우편번호를
--       postal_code_region을 통해 (시도, 시군구)로 변환한 결과를
--       실제 테이블(main_with_region)로 구체화(materialize)한다.
--       이후 모든 지역 기반 JOIN은 이 테이블을 기준으로 한다.
--
-- 뷰가 아니라 테이블인 이유(중요):
--       처음엔 이것도 뷰(vw_main_with_region)로 만들었는데, 그 위에
--       01_master_join.sql에서 외부데이터 8개를 추가로 LEFT JOIN하니
--       COUNT(*)가 5분 넘게 걸리며 타임아웃 남. vw_main_with_region 자체는
--       빠른데(COUNT만 하면 4,127,312행 바로 나옴), UNION ALL을 포함한
--       뷰 위에 뷰를 또 얹어서 조인하면 MySQL이 이를 인덱스 없는 임시
--       테이블로 만들어버려 후속 조인 최적화가 완전히 틀어지는 경우가 흔함
--       (특히 작은 외부 테이블 기준으로 400만 행짜리 임시결과를 거꾸로
--       스캔하는 실행계획을 짜는 경우). CREATE TABLE ... AS SELECT로 한 번
--       실제 테이블로 구체화해두면 이 문제가 사라진다 — 대신 원본 6개
--       테이블이나 postal_code_region이 바뀌면 이 스크립트를 다시 실행해서
--       테이블을 갱신해야 한다(뷰처럼 자동 반영되지 않음).
--
-- 안전성: postal_code_region.우편번호는 PRIMARY KEY(UNIQUE)이므로
--         이 LEFT JOIN은 N:1 관계다 — 여러 거래가 같은 우편번호를 가질 수는
--         있어도, 우편번호 하나당 매칭되는 지역 행은 정확히 0 또는 1개이므로
--         행 수가 늘어나지 않음. (검증: sql/join/02_validate_row_counts.sql).
--
-- 주의: postal_code_region_mapping은 매장이 실제로 존재하는 15개 시도
--       (887건)만 커버한다. 인천/세종 우편번호이거나 매핑 테이블에 없는
--       신규/변경 우편번호는 시도/시군구가 NULL로 남는다.
-- =====================================================================

DROP TABLE IF EXISTS main_with_region;

CREATE TABLE main_with_region AS
SELECT
    t.*,
    r.시도,
    r.시군구
FROM vw_main_transactions t
LEFT JOIN postal_code_region r
    ON t.우편번호 = r.우편번호;

-- 이후 01_master_join.sql에서 이 테이블을 기준(driving table)으로 외부
-- 데이터 8개를 LEFT JOIN하므로, 이 테이블 자체엔 별도 인덱스가 없어도
-- 무방하다(항상 풀스캔되는 쪽이라 인덱스가 조인 성능에 영향 없음).
-- 다만 나중에 이 테이블에 WHERE로 특정 시도/기간만 조회하는 용도로도
-- 쓴다면 아래 인덱스를 추가해도 된다(선택):
-- ALTER TABLE main_with_region ADD INDEX idx_region_date (시도, 시군구, 년, 월, 일);
-- ALTER TABLE main_with_region ADD INDEX idx_date (년, 월, 일);
