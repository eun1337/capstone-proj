"""
config.py

설명: 전처리 스크립트 공통 경로 설정 파일입니다.
  협업 시 이 파일의 BASE_DIR 경로만 본인 환경에 맞게 수정하세요.

[실행 순서]

1. convert_to_parquet.py   - 원본 엑셀 → parquet 변환
2. rename_col.py       - 컬럼명 통일
3. drop_invalid_barcode.py - 바코드 결측치(null, 0) 행 삭제
4. convert_date_col.py - 날짜 컬럼 타입 변환 및 통일
5. convert_zipcode_dtype.py - 우편번호 문자열 변환 (앞자리 0 보존)
6. drop_purchase_retail_col.py - 매입 데이터 매출처 컬럼 삭제
7. drop_optioncode_col.py  - 옵션 열 삭제
8. drop_invalid_purchase_row.py - 매입 부호 이상 데이터 삭제
9. drop_invalid_sales_row.py    - 매출 부호 이상 데이터 삭제
10. drop_invalid_zipcode_barcode.py   - 우편번호(1000 미만) / 바코드(비정상 자릿수) 행 삭제

"""

import os

# !경로 설정 (본인 환경에 맞게 수정)!

# 원본 엑셀 파일이 있는 data/ 폴더 경로
# 예) Windows: r"C:\Users\사용자명\Desktop\data"
# 예) Mac/Linux: "/Users/사용자명/Desktop/data"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

# parquet 저장 폴더 (자동 생성)
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)