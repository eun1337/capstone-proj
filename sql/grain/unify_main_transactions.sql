-- =====================================================================
-- unify_main_transactions.sql
--
-- 목적: A/B센터 x 매입/매출 6개 원본 테이블을 하나의 그레인(1행 = 1거래 라인, 일 단위)으로 통일한 뷰 생성. 테이블마다 컬럼명이 다름
--         - 날짜: 매입 `일자` vs 매출 `판매일`
--         - 우편번호: 매입 `공급업체 우편번호` vs 매출 `매출처 우편번호`
--         - 수량/금액: 매입 `수량`/`판매금액` vs 매출 `판매수량`/`공급가액`을 표준 컬럼명(거래일/우편번호/수량/금액)으로 정렬한다.
--
-- 그레인: 원본 그대로 유지 (집계/보간 없음). 이미 일 단위 트랜잭션이므로 날짜를 일(day) 단위로 맞추는 작업이 곧 이 뷰의 역할
--
-- 방어적 캐스팅 (load_to_mysql.py 점검 결과 반영):
--   load_to_mysql.py의 pd.read_csv()에 dtype/parse_dates 지정이 없어,
--   일자/판매일이 DATE가 아닌 TEXT로, 우편번호가 앞자리 0이 소실된 채
--   숫자형으로 적재됐을 가능성이 있음. 실제 컬럼 타입을 몰라도 안전하게
--   동작하도록 아래처럼 명시적으로 캐스팅
--     - 날짜   : CAST(... AS DATE)  — 이미 DATE/DATETIME이면 no-op,
--                'YYYY-MM-DD' 형태 TEXT여도 정상 변환됨.
--     - 우편번호: LPAD(CAST(... AS CHAR), 5, '0') — 이미 5자리 CHAR면 no-op,
--                정수형이라 앞자리 0이 소실된 상태여도 5자리로 복원됨.
--   실제 컬럼 타입 확인용: DESCRIBE a_sales_2021_2023; (일자/우편번호 Type 확인)
--
-- 주의: 메인 데이터엔 트랜잭션 고유 PK가 없다(문서 docs/join_key_candidates.md
--       1-1 참고). 이 뷰는 원본 행을 그대로 통과시키므로, 원본에 중복/이상
--       행이 있었다면 이 뷰에도 동일하게 남아있다(여기서 별도 정제는 하지 않음
--       — drop_invalid_*.py 등 기존 전처리 단계에서 이미 처리된 것으로 간주).
--
-- 우편번호 관련 유의: 매입의 `공급업체 우편번호`와 매출의 `매출처 우편번호`는
--       서로 다른 주체(공급처 vs 고객사)의 우편번호이므로, 조인 후 "지역"의
--       의미가 매입/매출에서 서로 다르다는 점을 분석 시 반드시 인지할 것.
-- =====================================================================

CREATE OR REPLACE VIEW vw_main_transactions AS
SELECT
    'A'   AS 센터,
    '매입' AS 유형,
    CAST(일자 AS DATE)                             AS 거래일,
    YEAR(CAST(일자 AS DATE))                       AS 년,
    MONTH(CAST(일자 AS DATE))                      AS 월,
    DAY(CAST(일자 AS DATE))                        AS 일,
    LPAD(CAST(`공급업체 우편번호` AS CHAR), 5, '0') AS 우편번호,
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    수량                  AS 수량,
    판매금액              AS 금액
FROM a_purchase

UNION ALL

SELECT
    'B'   AS 센터,
    '매입' AS 유형,
    CAST(일자 AS DATE),
    YEAR(CAST(일자 AS DATE)), MONTH(CAST(일자 AS DATE)), DAY(CAST(일자 AS DATE)),
    LPAD(CAST(`공급업체 우편번호` AS CHAR), 5, '0'),
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    수량,
    판매금액
FROM b_purchase

UNION ALL

SELECT
    'A'   AS 센터,
    '매출' AS 유형,
    CAST(판매일 AS DATE),
    YEAR(CAST(판매일 AS DATE)), MONTH(CAST(판매일 AS DATE)), DAY(CAST(판매일 AS DATE)),
    LPAD(CAST(`매출처 우편번호` AS CHAR), 5, '0'),
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    판매수량              AS 수량,
    공급가액              AS 금액
FROM a_sales_2021_2023

UNION ALL

SELECT
    'A'   AS 센터,
    '매출' AS 유형,
    CAST(판매일 AS DATE),
    YEAR(CAST(판매일 AS DATE)), MONTH(CAST(판매일 AS DATE)), DAY(CAST(판매일 AS DATE)),
    LPAD(CAST(`매출처 우편번호` AS CHAR), 5, '0'),
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    판매수량,
    공급가액
FROM a_sales_2024

UNION ALL

SELECT
    'B'   AS 센터,
    '매출' AS 유형,
    CAST(판매일 AS DATE),
    YEAR(CAST(판매일 AS DATE)), MONTH(CAST(판매일 AS DATE)), DAY(CAST(판매일 AS DATE)),
    LPAD(CAST(`매출처 우편번호` AS CHAR), 5, '0'),
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    판매수량,
    공급가액
FROM b_sales_2021_2023

UNION ALL

SELECT
    'B'   AS 센터,
    '매출' AS 유형,
    CAST(판매일 AS DATE),
    YEAR(CAST(판매일 AS DATE)), MONTH(CAST(판매일 AS DATE)), DAY(CAST(판매일 AS DATE)),
    LPAD(CAST(`매출처 우편번호` AS CHAR), 5, '0'),
    바코드, 상품명, KAN_대분류, KAN_중분류, KAN_소분류,
    판매수량,
    공급가액
FROM b_sales_2024;
