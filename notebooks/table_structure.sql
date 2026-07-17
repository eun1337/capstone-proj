-- 행수 확인
USE capstone_db;
DESCRIBE a_center_purchase;
DESCRIBE a_center_sales_2021_2023;
DESCRIBE a_center_sales_2024;
DESCRIBE b_center_purchase;
DESCRIBE b_center_sales_2021_2023;
DESCRIBE b_center_sales_2024;

SELECT 
    'a_center_purchase' AS 테이블명, COUNT(*) AS 행수 FROM a_center_purchase
UNION ALL SELECT 'a_center_sales_2021_2023', COUNT(*) FROM a_center_sales_2021_2023
UNION ALL SELECT 'a_center_sales_2024', COUNT(*) FROM a_center_sales_2024
UNION ALL SELECT 'b_center_purchase', COUNT(*) FROM b_center_purchase
UNION ALL SELECT 'b_center_sales_2021_2023', COUNT(*) FROM b_center_sales_2021_2023
UNION ALL SELECT 'b_center_sales_2024', COUNT(*) FROM b_center_sales_2024;


-- 컬럼명 및 타입 확인
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'capstone_db'
ORDER BY TABLE_NAME, ORDINAL_POSITION;


-- 날짜 범위 확인

-- (매입 날짜 범위)
SELECT 'a_center_purchase' AS 테이블명, MIN(일자) AS 시작일, MAX(일자) AS 종료일 FROM a_center_purchase
UNION ALL
SELECT 'b_center_purchase', MIN(일자), MAX(일자) FROM b_center_purchase;

-- (매출 날짜 범위)
SELECT 'a_center_sales_2021_2023' AS 테이블명, MIN(판매일) AS 시작일, MAX(판매일) AS 종료일 FROM a_center_sales_2021_2023
UNION ALL
SELECT 'a_center_sales_2024', MIN(판매일), MAX(판매일) FROM a_center_sales_2024
UNION ALL
SELECT 'b_center_sales_2021_2023', MIN(판매일), MAX(판매일) FROM b_center_sales_2021_2023
UNION ALL
SELECT 'b_center_sales_2024', MIN(판매일), MAX(판매일) FROM b_center_sales_2024;