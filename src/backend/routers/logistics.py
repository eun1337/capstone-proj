from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class ProductItem(BaseModel):
    id: int
    barcode: str
    product_name: str
    category: str
    center: str
    forecast_qty: int
    current_stock: int
    status: str


MOCK_PRODUCTS: List[ProductItem] = [
    ProductItem(id=1,  barcode="8801234567890", product_name="제주 감귤 2kg",   category="과일", center="A센터", forecast_qty=320, current_stock=150, status="부족"),
    ProductItem(id=2,  barcode="8809876543210", product_name="경기 쌀 10kg",    category="곡물", center="A센터", forecast_qty=500, current_stock=620, status="정상"),
    ProductItem(id=3,  barcode="8801122334455", product_name="강원 감자 3kg",   category="채소", center="B센터", forecast_qty=210, current_stock=80,  status="부족"),
    ProductItem(id=4,  barcode="8806677889900", product_name="충남 딸기 1kg",   category="과일", center="B센터", forecast_qty=150, current_stock=200, status="정상"),
    ProductItem(id=5,  barcode="8800011223344", product_name="전남 고구마 5kg", category="채소", center="A센터", forecast_qty=380, current_stock=310, status="주의"),
    ProductItem(id=6,  barcode="8805566778899", product_name="경북 사과 5kg",   category="과일", center="A센터", forecast_qty=420, current_stock=180, status="부족"),
    ProductItem(id=7,  barcode="8802233445566", product_name="충북 오이 2kg",   category="채소", center="B센터", forecast_qty=95,  current_stock=120, status="정상"),
    ProductItem(id=8,  barcode="8807788990011", product_name="제주 당근 2kg",   category="채소", center="B센터", forecast_qty=270, current_stock=260, status="정상"),
    ProductItem(id=9,  barcode="8803344556677", product_name="제주 한라봉 2kg", category="과일", center="A센터", forecast_qty=185, current_stock=90,  status="주의"),
    ProductItem(id=10, barcode="8804455667788", product_name="전북 보리 5kg",   category="곡물", center="B센터", forecast_qty=300, current_stock=340, status="정상"),
    ProductItem(id=11, barcode="8806655443322", product_name="경기 배추 3kg",   category="채소", center="A센터", forecast_qty=450, current_stock=200, status="부족"),
    ProductItem(id=12, barcode="8808899001122", product_name="충남 포도 2kg",   category="과일", center="B센터", forecast_qty=130, current_stock=155, status="정상"),
]


class SummaryData(BaseModel):
    total_items: int
    shortage_items: int
    caution_items: int
    normal_items: int
    total_forecast_qty: int
    total_stock_qty: int
    top_forecast: List[dict]


@router.get("/products", response_model=List[ProductItem])
def get_products(
    center: Optional[str] = Query(None, description="A센터 / B센터"),
    category: Optional[str] = Query(None, description="과일 / 채소 / 곡물"),
    status: Optional[str] = Query(None, description="정상 / 주의 / 부족"),
):
    items = MOCK_PRODUCTS
    if center:
        items = [p for p in items if p.center == center]
    if category:
        items = [p for p in items if p.category == category]
    if status:
        items = [p for p in items if p.status == status]
    return items


@router.get("/summary", response_model=SummaryData)
def get_summary(center: Optional[str] = Query(None)):
    items = MOCK_PRODUCTS
    if center:
        items = [p for p in items if p.center == center]

    top = sorted(items, key=lambda x: x.forecast_qty, reverse=True)[:5]
    return SummaryData(
        total_items=len(items),
        shortage_items=sum(1 for p in items if p.status == "부족"),
        caution_items=sum(1 for p in items if p.status == "주의"),
        normal_items=sum(1 for p in items if p.status == "정상"),
        total_forecast_qty=sum(p.forecast_qty for p in items),
        total_stock_qty=sum(p.current_stock for p in items),
        top_forecast=[{"name": p.product_name, "qty": p.forecast_qty, "center": p.center} for p in top],
    )
