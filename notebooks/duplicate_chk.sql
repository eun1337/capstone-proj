-- 중복 행 확인

-- 매입 테이블 (일자 + 상품코드 + 수량 기준)
SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 일자, 상품코드, 수량) AS 유니크행수
FROM a_center_purchase;

SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 일자, 상품코드, 수량) AS 유니크행수
FROM b_center_purchase;

-- 매출 테이블 (판매일 + 매출처코드 + 바코드 + 판매수량 기준)
SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 판매일, 매출처코드, 바코드, 판매수량) AS 유니크행수
FROM a_center_sales_2021_2023;

SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 판매일, 매출처코드, `상품 바코드(대한상의)`, 판매수량) AS 유니크행수
FROM a_center_sales_2024;

SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 판매일, 매출처코드, 바코드, 판매수량) AS 유니크행수
FROM b_center_sales_2021_2023;

SELECT COUNT(*) AS 전체행수,
       COUNT(DISTINCT 판매일, 매출처코드, 바코드, 판매수량) AS 유니크행수
FROM b_center_sales_2024;


-- 중복 행 상세 확인(1)

-- a_center_sales_2024 중복 예시
SELECT 판매일, 매출처코드, `상품 바코드(대한상의)`, 판매수량, COUNT(*) AS 중복횟수
FROM a_center_sales_2024
GROUP BY 판매일, 매출처코드, `상품 바코드(대한상의)`, 판매수량
HAVING COUNT(*) > 1
ORDER BY 중복횟수 DESC
LIMIT 10;

-- b_center_purchase 중복 예시
SELECT 일자, 상품코드, 수량, COUNT(*) AS 중복횟수
FROM b_center_purchase
GROUP BY 일자, 상품코드, 수량
HAVING COUNT(*) > 1
ORDER BY 중복횟수 DESC
LIMIT 10;


-- 중복 행 상세 확인(2)

-- a_center_sales_2024 중복 행 전체 컬럼 확인
SELECT *
FROM a_center_sales_2024
WHERE 판매일 = '2024-07-31'
AND 매출처코드 = 2315
AND `상품 바코드(대한상의)` = 8801040000000
AND 판매수량 = 2
LIMIT 5;

-- b_center_purchase 중복 행 전체 컬럼 확인
SELECT *
FROM b_center_purchase
WHERE 일자 = '2024-05-07'
AND 상품코드 = 2880934893682
AND 수량 = 5
LIMIT 5;