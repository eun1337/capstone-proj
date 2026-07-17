-- a_center_sales_2021_2023 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM a_center_sales_2021_2023
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;

-- a_center_sales_2024 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM a_center_sales_2024
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;

-- b_center_sales_2021_2023 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM b_center_sales_2021_2023
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;

-- b_center_sales_2024 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM b_center_sales_2024
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;

-- a_center_purchase 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM a_center_purchase
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;

-- b_center_purchase 바코드 오류
SELECT 바코드, COUNT(DISTINCT 상품명) AS 상품명_수
FROM b_center_purchase
GROUP BY 바코드
HAVING COUNT(DISTINCT 상품명) > 1
ORDER BY 상품명_수 DESC;