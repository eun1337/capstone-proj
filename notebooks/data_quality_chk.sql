-- 매입 작업유형 종류
SELECT 'a_center_purchase' AS 테이블명, 작업유형, COUNT(*) AS 건수
FROM a_center_purchase GROUP BY 작업유형
UNION ALL
SELECT 'b_center_purchase', 작업유형, COUNT(*)
FROM b_center_purchase GROUP BY 작업유형
ORDER BY 테이블명, 건수 DESC;

-- 매출 구분 종류
SELECT 'a_center_sales_2021_2023' AS 테이블명, 구분, COUNT(*) AS 건수
FROM a_center_sales_2021_2023 GROUP BY 구분
UNION ALL
SELECT 'a_center_sales_2024', 구분, COUNT(*)
FROM a_center_sales_2024 GROUP BY 구분
UNION ALL
SELECT 'b_center_sales_2021_2023', 구분, COUNT(*)
FROM b_center_sales_2021_2023 GROUP BY 구분
UNION ALL
SELECT 'b_center_sales_2024', 구분, COUNT(*)
FROM b_center_sales_2024 GROUP BY 구분
ORDER BY 테이블명, 건수 DESC;

-- 매입 음수 확인
SELECT 'a_center_purchase 수량 음수' AS 항목, COUNT(*) AS 건수 FROM a_center_purchase WHERE 수량 < 0
UNION ALL
SELECT 'a_center_purchase 판매금액 음수', COUNT(*) FROM a_center_purchase WHERE 판매금액 < 0
UNION ALL
SELECT 'b_center_purchase 수량 음수', COUNT(*) FROM b_center_purchase WHERE 수량 < 0
UNION ALL
SELECT 'b_center_purchase 판매금액 음수', COUNT(*) FROM b_center_purchase WHERE 판매금액 < 0;

-- 매출 음수 확인
SELECT 'a_sales_2021_2023 판매수량 음수' AS 항목, COUNT(*) AS 건수 FROM a_center_sales_2021_2023 WHERE 판매수량 < 0
UNION ALL
SELECT 'a_sales_2021_2023 공급금액 음수', COUNT(*) FROM a_center_sales_2021_2023 WHERE 공급금액 < 0
UNION ALL
SELECT 'a_sales_2024 판매수량 음수', COUNT(*) FROM a_center_sales_2024 WHERE 판매수량 < 0
UNION ALL
SELECT 'a_sales_2024 공급가액 음수', COUNT(*) FROM a_center_sales_2024 WHERE 공급가액 < 0
UNION ALL
SELECT 'b_sales_2021_2023 판매수량 음수', COUNT(*) FROM b_center_sales_2021_2023 WHERE 판매수량 < 0
UNION ALL
SELECT 'b_sales_2021_2023 공급금액 음수', COUNT(*) FROM b_center_sales_2021_2023 WHERE 공급금액 < 0
UNION ALL
SELECT 'b_sales_2024 판매수량 음수', COUNT(*) FROM b_center_sales_2024 WHERE 판매수량 < 0;

-- a_center_purchase 반출 확인
SELECT 작업유형,
    CASE WHEN 판매금액 > 0 THEN '양수'
         WHEN 판매금액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 수량 > 0 THEN '양수'
         WHEN 수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM a_center_purchase
GROUP BY 작업유형, 금액부호, 수량부호
ORDER BY 작업유형, 건수 DESC;

-- b_center_purchase 반출
SELECT 작업유형,
    CASE WHEN 판매금액 > 0 THEN '양수'
         WHEN 판매금액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 수량 > 0 THEN '양수'
         WHEN 수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM b_center_purchase
GROUP BY 작업유형, 금액부호, 수량부호
ORDER BY 작업유형, 건수 DESC;

-- a_center_sales_2021_2023 반품
SELECT 구분,
    CASE WHEN 공급금액 > 0 THEN '양수'
         WHEN 공급금액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 판매수량 > 0 THEN '양수'
         WHEN 판매수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM a_center_sales_2021_2023
GROUP BY 구분, 금액부호, 수량부호
ORDER BY 구분, 건수 DESC;

-- a_center_sales_2024 반품
SELECT 구분,
    CASE WHEN 공급가액 > 0 THEN '양수'
         WHEN 공급가액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 판매수량 > 0 THEN '양수'
         WHEN 판매수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM a_center_sales_2024
GROUP BY 구분, 금액부호, 수량부호
ORDER BY 구분, 건수 DESC;

-- b_center_sales_2021_2023 반품
SELECT 구분,
    CASE WHEN 공급금액 > 0 THEN '양수'
         WHEN 공급금액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 판매수량 > 0 THEN '양수'
         WHEN 판매수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM b_center_sales_2021_2023
GROUP BY 구분, 금액부호, 수량부호
ORDER BY 구분, 건수 DESC;

-- b_center_sales_2024 반품
SELECT 구분,
    CASE WHEN 공급금액 > 0 THEN '양수'
         WHEN 공급금액 < 0 THEN '음수'
         ELSE '0' END AS 금액부호,
    CASE WHEN 판매수량 > 0 THEN '양수'
         WHEN 판매수량 < 0 THEN '음수'
         ELSE '0' END AS 수량부호,
    COUNT(*) AS 건수
FROM b_center_sales_2024
GROUP BY 구분, 금액부호, 수량부호
ORDER BY 구분, 건수 DESC;

-- a_center_sales_2024 반품_수량 양수 (4,580건)
SELECT *
FROM a_center_sales_2024
WHERE 구분 = '반품'
AND 판매수량 > 0;

-- a_center_purchase 입고_금액 0 (4,941건)
SELECT *
FROM a_center_purchase
WHERE 작업유형 = '입고'
AND 판매금액 = 0;

-- b_center_purchase 입고_음수 (645건)
SELECT *
FROM b_center_purchase
WHERE 작업유형 = '입고'
AND 판매금액 < 0;

-- b_center_sales_2021_2023 매출_음수 (719건)
SELECT *
FROM b_center_sales_2021_2023
WHERE 구분 = '매출'
AND 공급금액 < 0;

-- b_center_sales_2024 매출_음수 (1,742건)
SELECT *
FROM b_center_sales_2024
WHERE 구분 = '매출'
AND 공급금액 < 0;