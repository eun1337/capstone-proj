_source_sheet 컬럼 삭제
--    (멀티시트 합본 파일에만 존재하는 내부 컬럼)
ALTER TABLE raw_a_sales_2021_2023 DROP COLUMN _source_sheet;
ALTER TABLE raw_b_sales_2021_2023 DROP COLUMN _source_sheet;


