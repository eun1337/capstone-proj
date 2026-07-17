-- =====================================================================
-- unmatched_postal_codes_full.sql
--
-- 목적: postal_code_region_mapping에 없어서 시도/시군구가 NULL로 남은
--       우편번호를 전부(TOP20 제한 없이) 출력. 매핑 테이블을 100% 커버로
--       확장하기 위해, 실제로 채워야 할 우편번호 전체 목록을 뽑는 용도.
-- =====================================================================

-- [전체 1] 매칭 실패 우편번호 전체 목록 (건수 많은 순)
SELECT
    m.우편번호,
    COUNT(*) AS 거래건수
FROM main_with_region m
WHERE m.시도 IS NULL
  AND m.우편번호 IS NOT NULL
  AND m.우편번호 <> ''
GROUP BY m.우편번호
ORDER BY 거래건수 DESC;

-- [전체 2] 참고용 — 매입(공급업체 우편번호) vs 매출(매출처 우편번호) 구분해서 보고 싶을 때
-- (원본 컬럼이 서로 다른 주체의 우편번호라, 어느 쪽에서 많이 실패했는지 보면
--  매핑 데이터를 어디서부터 채워야 할지 우선순위 잡기 쉬움)
SELECT
    m.우편번호,
    m.유형,
    COUNT(*) AS 거래건수
FROM main_with_region m
WHERE m.시도 IS NULL
  AND m.우편번호 IS NOT NULL
  AND m.우편번호 <> ''
GROUP BY m.우편번호, m.유형
ORDER BY m.우편번호, 거래건수 DESC;
