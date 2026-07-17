-- a_center_purchase 결측치
SELECT
    SUM(CASE WHEN 작업유형 IS NULL OR 작업유형 = '' THEN 1 ELSE 0 END) AS 작업유형_결측,
    SUM(CASE WHEN 일자 IS NULL OR 일자 = '' THEN 1 ELSE 0 END) AS 일자_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN `매출처 우편번호` IS NULL THEN 1 ELSE 0 END) AS 매출처우편번호_결측,
    SUM(CASE WHEN `공급업체 코드` IS NULL THEN 1 ELSE 0 END) AS 공급업체코드_결측,
    SUM(CASE WHEN `공급업체 우편번호` IS NULL THEN 1 ELSE 0 END) AS 공급업체우편번호_결측,
    SUM(CASE WHEN `입고 형태` IS NULL OR `입고 형태` = '' THEN 1 ELSE 0 END) AS 입고형태_결측,
    SUM(CASE WHEN 상품코드 IS NULL THEN 1 ELSE 0 END) AS 상품코드_결측,
    SUM(CASE WHEN 바코드 IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN `옵션 코드` IS NULL OR `옵션 코드` = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN 수량 IS NULL THEN 1 ELSE 0 END) AS 수량_결측,
    SUM(CASE WHEN 판매금액 IS NULL THEN 1 ELSE 0 END) AS 판매금액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM a_center_purchase;

-- b_center_purchase 결측치
SELECT
    SUM(CASE WHEN 작업유형 IS NULL OR 작업유형 = '' THEN 1 ELSE 0 END) AS 작업유형_결측,
    SUM(CASE WHEN 일자 IS NULL OR 일자 = '' THEN 1 ELSE 0 END) AS 일자_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN `매출처 우편번호` IS NULL THEN 1 ELSE 0 END) AS 매출처우편번호_결측,
    SUM(CASE WHEN `공급업체 코드` IS NULL THEN 1 ELSE 0 END) AS 공급업체코드_결측,
    SUM(CASE WHEN `공급업체 우편번호` IS NULL THEN 1 ELSE 0 END) AS 공급업체우편번호_결측,
    SUM(CASE WHEN `입고 형태` IS NULL OR `입고 형태` = '' THEN 1 ELSE 0 END) AS 입고형태_결측,
    SUM(CASE WHEN 상품코드 IS NULL THEN 1 ELSE 0 END) AS 상품코드_결측,
    SUM(CASE WHEN 바코드 IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN `옵션 코드` IS NULL OR `옵션 코드` = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN 수량 IS NULL THEN 1 ELSE 0 END) AS 수량_결측,
    SUM(CASE WHEN 판매금액 IS NULL THEN 1 ELSE 0 END) AS 판매금액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM b_center_purchase;

-- a_center_sales_2021_2023 결측치
SELECT
    SUM(CASE WHEN 판매일 IS NULL OR 판매일 = '' THEN 1 ELSE 0 END) AS 판매일_결측,
    SUM(CASE WHEN 구분 IS NULL OR 구분 = '' THEN 1 ELSE 0 END) AS 구분_결측,
    SUM(CASE WHEN `매출처 우편번호` IS NULL THEN 1 ELSE 0 END) AS 매출처우편번호_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN 판매수량 IS NULL THEN 1 ELSE 0 END) AS 판매수량_결측,
    SUM(CASE WHEN `옵션 코드` IS NULL OR `옵션 코드` = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN 바코드 IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN 공급금액 IS NULL THEN 1 ELSE 0 END) AS 공급금액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM a_center_sales_2021_2023;

-- a_center_sales_2024 결측치
SELECT
    SUM(CASE WHEN 판매일 IS NULL OR 판매일 = '' THEN 1 ELSE 0 END) AS 판매일_결측,
    SUM(CASE WHEN 구분 IS NULL OR 구분 = '' THEN 1 ELSE 0 END) AS 구분_결측,
    SUM(CASE WHEN 우편번호 IS NULL THEN 1 ELSE 0 END) AS 우편번호_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN 판매수량 IS NULL THEN 1 ELSE 0 END) AS 판매수량_결측,
    SUM(CASE WHEN 옵션코드 IS NULL OR 옵션코드 = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN `상품 바코드(대한상의)` IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN 공급가액 IS NULL THEN 1 ELSE 0 END) AS 공급가액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM a_center_sales_2024;

-- b_center_sales_2021_2023 결측치
SELECT
    SUM(CASE WHEN 판매일 IS NULL OR 판매일 = '' THEN 1 ELSE 0 END) AS 판매일_결측,
    SUM(CASE WHEN 구분 IS NULL OR 구분 = '' THEN 1 ELSE 0 END) AS 구분_결측,
    SUM(CASE WHEN `매출처 우편번호` IS NULL THEN 1 ELSE 0 END) AS 매출처우편번호_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN 판매수량 IS NULL THEN 1 ELSE 0 END) AS 판매수량_결측,
    SUM(CASE WHEN `옵션 코드` IS NULL OR `옵션 코드` = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN 바코드 IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN 공급금액 IS NULL THEN 1 ELSE 0 END) AS 공급금액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM b_center_sales_2021_2023;

-- b_center_sales_2024 결측치
SELECT
    SUM(CASE WHEN 판매일 IS NULL OR 판매일 = '' THEN 1 ELSE 0 END) AS 판매일_결측,
    SUM(CASE WHEN 구분 IS NULL OR 구분 = '' THEN 1 ELSE 0 END) AS 구분_결측,
    SUM(CASE WHEN `매출처 우편번호` IS NULL THEN 1 ELSE 0 END) AS 매출처우편번호_결측,
    SUM(CASE WHEN 매출처코드 IS NULL THEN 1 ELSE 0 END) AS 매출처코드_결측,
    SUM(CASE WHEN 판매수량 IS NULL THEN 1 ELSE 0 END) AS 판매수량_결측,
    SUM(CASE WHEN `옵션 코드` IS NULL OR `옵션 코드` = '' THEN 1 ELSE 0 END) AS 옵션코드_결측,
    SUM(CASE WHEN 바코드 IS NULL THEN 1 ELSE 0 END) AS 바코드_결측,
    SUM(CASE WHEN 상품명 IS NULL OR 상품명 = '' THEN 1 ELSE 0 END) AS 상품명_결측,
    SUM(CASE WHEN 공급금액 IS NULL THEN 1 ELSE 0 END) AS 공급금액_결측,
    SUM(CASE WHEN 대분류 IS NULL OR 대분류 = '' THEN 1 ELSE 0 END) AS 대분류_결측,
    SUM(CASE WHEN 중분류 IS NULL OR 중분류 = '' THEN 1 ELSE 0 END) AS 중분류_결측,
    SUM(CASE WHEN 소분류 IS NULL OR 소분류 = '' THEN 1 ELSE 0 END) AS 소분류_결측
FROM b_center_sales_2024;