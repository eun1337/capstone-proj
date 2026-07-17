-- =====================================================================
-- validate_row_counts.sql
--
-- 목적: 그레인 통일 + LEFT JOIN 과정에서 팬아웃 여부 검증. 모든 단계의 행 수가 동일해야 정상
-- =====================================================================

-- [검증 1] 원본 6개 테이블 합계 vs 각 뷰 단계별 행 수 — 전부 같아야 함
SELECT '1-1 원본 6개 테이블 합계' AS 검증항목,
       (SELECT COUNT(*) FROM a_purchase)
     + (SELECT COUNT(*) FROM b_purchase)
     + (SELECT COUNT(*) FROM a_sales_2021_2023)
     + (SELECT COUNT(*) FROM a_sales_2024)
     + (SELECT COUNT(*) FROM b_sales_2021_2023)
     + (SELECT COUNT(*) FROM b_sales_2024) AS 행수
UNION ALL
SELECT '1-2 vw_main_transactions', COUNT(*) FROM vw_main_transactions
UNION ALL
SELECT '1-3 main_with_region', COUNT(*) FROM main_with_region
UNION ALL
SELECT '1-4 main_joined_all', COUNT(*) FROM main_joined_all;
-- 기대 결과: 4행의 '행수'가 전부 동일한 값이어야 함.
-- 값이 다르면 > 1-2 다르면 UNION ALL 컬럼 순서/개수 오류,
--            1-3 다르면 postal_code_region 우편번호 중복,
--            1-4 다르면 외부 데이터 조인 키 중복(팬아웃) 의심.


-- [검증 2] 각 LEFT JOIN을 개별적으로 main_with_region에 붙였을 때
--          단독으로도 행 수가 늘어나지 않는지 개별 확인 (문제 조인 격리용)
SELECT 'A. weather 단독 조인' AS 조인대상, COUNT(*) AS 행수
FROM main_with_region m
LEFT JOIN weather_daily w
    ON  w.시도 = m.시도
    AND (w.시군구 = m.시군구 OR w.시군구 IS NULL OR m.시군구 LIKE CONCAT(w.시군구, ' %'))
    AND w.년 = m.년 AND w.월 = m.월 AND w.일 = m.일

UNION ALL

SELECT 'B. public_holidays 단독 조인', COUNT(*)
FROM main_with_region m
LEFT JOIN public_holidays h ON h.년 = m.년 AND h.월 = m.월 AND h.일 = m.일

UNION ALL

SELECT 'C. covid_impact_daily 단독 조인', COUNT(*)
FROM main_with_region m
LEFT JOIN covid_impact_daily c ON c.년 = m.년 AND c.월 = m.월 AND c.일 = m.일

UNION ALL

SELECT 'D. cpi_monthly_national 단독 조인', COUNT(*)
FROM main_with_region m
LEFT JOIN cpi_monthly_national cpi ON cpi.년 = m.년 AND cpi.월 = m.월

UNION ALL

SELECT 'E. retail_sales_current 단독 조인', COUNT(*)
FROM main_with_region m
LEFT JOIN retail_sales_current rsc ON rsc.시도 = m.시도 AND rsc.년 = m.년 AND rsc.월 = m.월

UNION ALL

SELECT 'F. retail_sales_constant 단독 조인', COUNT(*)
FROM main_with_region m
LEFT JOIN retail_sales_constant rsk ON rsk.시도 = m.시도 AND rsk.년 = m.년 AND rsk.월 = m.월

UNION ALL

SELECT '기준값 (main_with_region)', COUNT(*) FROM main_with_region;
-- 기대 결과: A~F 모두 '기준값'과 동일해야 함. 하나라도 크면 그 조인의
-- 매칭 조건이 N:1이 아니라 N:M이 되어 팬아웃이 발생했다는 뜻이므로, 해당 외부 테이블에서 조인 키 기준 중복 행이 있는지 아래처럼 확인할 것.
--   예) SELECT 시도,시군구,년,월,일, COUNT(*) FROM weather_daily
--       GROUP BY 시도,시군구,년,월,일 HAVING COUNT(*) > 1;
