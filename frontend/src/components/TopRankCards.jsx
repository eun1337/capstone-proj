import { useState } from 'react';
import InsightCard from './InsightCard.jsx';
import InsightModal from './InsightModal.jsx';
import RegionMap from './RegionMap.jsx';
import './TopRankCards.css';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtNum = (n) => Math.round(n).toLocaleString();
const fmtNum1 = (n) => n.toFixed(1);
const fmtPct = (v) => (v === null || v === undefined ? '-' : `${v.toFixed(1)}%`);

// 대시보드 하단(우측 2개 + 아래 4개) TOP5 카드 6종 + 각 "전체보기" modal.
// 상단에서 이미 단위(EA/BX/CS)가 확정돼 있으므로, 여기서는 행마다 단위 배지/단위 접미사를
// 반복하지 않는다(단위 표시는 modal 표에서만 별도 컬럼으로). 비율(비중/충족률/증가율)은
// 요청대로 메인 카드에서는 빼고 modal 표에만 남긴다.

/* ── In-cell bar row ── 메인 카드(TOP5) ranking list 전용. 별도 차트 없이 행 배경에
   1위 기준 비율(%) 막대를 깔아, 텍스트(좌: 이름, 우: 값)는 그 위에 그대로 보이게 한다.
   tone: 'blue'(매출/지역) | 'amber'|'red'(고정 톤 — 지금은 재고부족이 accentColor로
   대체해서 안 씀). accentColor가 주어지면 tone 클래스 대신 인라인 그라데이션/텍스트
   색을 그대로 쓴다(순위별 빨강→노랑 연속 그라데이션처럼 클래스 몇 개로 못 나눌 때). */
function BarRow({ label, subLabel, valueText, pct, tone = 'blue', accentColor }) {
  const width = Math.max(4, Math.min(100, pct));
  const fillStyle = accentColor
    ? { width: `${width}%`, backgroundImage: `linear-gradient(90deg, ${accentColor.bgStrong}, ${accentColor.bgSoft})` }
    : { width: `${width}%` };
  return (
    <div className={`trc-bar-row ${accentColor ? '' : `trc-bar-${tone}`}`}>
      <div className="trc-bar-fill" style={fillStyle} />
      <span className="trc-bar-label" title={subLabel ? `${label} · ${subLabel}` : label}>
        <span className="trc-bar-label-main">{label}</span>
        {subLabel && <span className="trc-bar-sub">{subLabel}</span>}
      </span>
      <span className="trc-bar-val" style={accentColor ? { color: accentColor.text } : undefined}>{valueText}</span>
    </div>
  );
}

// ── 순위별 빨강→노랑(주황) 연속 그라데이션 ──────────────────────────────
// 결품 위험 TOP5는 이미 shortage_qty 내림차순(1위 = 가장 위험)이므로, 순위(idx)를 그대로
// "위험도"로 써서 1위=진한 빨강 → 5위=앰버/노랑으로 자연스럽게 이어지는 색을 만든다.
function lerpRGB(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

function shortageRankColor(idx, total) {
  const t = total > 1 ? idx / (total - 1) : 0; // 0 = 1위(최상단, 가장 위험) → 1 = 마지막
  const [r, g, b] = lerpRGB([220, 38, 38], [245, 158, 11], t);   // red-600 → amber-500
  const [tr, tg, tb] = lerpRGB([153, 27, 27], [180, 83, 9], t);  // 텍스트용 더 진한 변형
  return {
    bgStrong: `rgba(${r}, ${g}, ${b}, 0.22)`,
    bgSoft: `rgba(${r}, ${g}, ${b}, 0.04)`,
    text: `rgb(${tr}, ${tg}, ${tb})`,
  };
}

/* ── 가로 막대 차트 ── DETAIL 모달 상단에 TOP N 데이터를 시각화
   items: [{ label: string, value: number }]
   unit : 표시 단위 문자열 (선택)
   color: 막대 색상 (기본 #3b82f6) */
function HorizontalBarChart({ items, unit = '', color = '#3b82f6', formatValue = (v) => Math.round(v).toLocaleString() }) {
  if (!items || items.length === 0) return null;
  const maxVal = Math.max(...items.map((i) => i.value), 1);
  return (
    <div style={{ marginBottom: 18 }}>
      {items.map((item, idx) => (
        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
          <span
            style={{ flex: '0 0 140px', fontSize: 11.5, color: '#374151', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textAlign: 'right' }}
            title={item.label}
          >
            {item.label}
          </span>
          <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 4, height: 16, position: 'relative', overflow: 'hidden' }}>
            <div
              style={{
                width: `${(item.value / maxVal) * 100}%`,
                height: '100%',
                background: color,
                borderRadius: 4,
                transition: 'width 0.3s ease',
              }}
            />
          </div>
          <span style={{ flex: '0 0 80px', fontSize: 11.5, fontWeight: 700, color: '#1e293b', textAlign: 'right' }}>
            {formatValue(item.value)}{unit ? ` ${unit}` : ''}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   1. 대분류/중분류/소분류별 매출 상위 → 소분류까지 선택되면 상품별 매출 상위로 전환
   (같은 슬롯, 좌측 카테고리 트리 드릴다운 깊이에 따라 4단계로 바뀐다)
   ══════════════════════════════════════════════════════════════ */
// selectedPath(트리에서 선택한 [대분류, 중분류, 소분류] 경로)의 길이 = 드릴다운 깊이.
// depth 0(전체) → 대분류별, 1(대분류 선택) → 중분류별, 2(대분류+중분류) → 소분류별,
// 3(소분류까지 확정, 더 내려갈 하위 카테고리 없음) → 상품(SKU)별로 전환한다.
// 백엔드(get_daily_category_sales)의 group_col 로직과 정확히 1:1로 대응된다.
const CATEGORY_LEVEL_TITLES = ['대분류별 매출 상위', '중분류별 매출 상위', '소분류별 매출 상위'];
const CATEGORY_LEVEL_LABELS = ['대분류', '중분류', '소분류'];

export function categoryDrilldownInfo(selectedPath) {
  const depth = selectedPath?.length ?? 0;
  const isProductLevel = depth >= 3;
  return {
    isProductLevel,
    title: isProductLevel ? '상품별 매출 상위' : CATEGORY_LEVEL_TITLES[depth],
    levelLabel: isProductLevel ? '상품' : CATEGORY_LEVEL_LABELS[depth],
  };
}

export function CategoryOrProductTop5Card({ data, loading, error, onOpenDetail, selectedPath = [] }) {
  const { isProductLevel, title } = categoryDrilldownInfo(selectedPath);
  const items = isProductLevel ? data?.product_top5 : data?.top5;
  const isEmpty = !data || !items || items.length === 0 || data.total_sales_amount === 0;
  const maxVal = items && items.length ? Math.max(...items.map((r) => r.sales_amount)) : 1;

  return (
    <InsightCard
      title={title}
      loading={loading}
      error={error}
      isEmpty={isEmpty}
      emptyMessage="해당 조회일에 판매 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items?.map((row) => (
          <BarRow
            key={isProductLevel ? row.sku_id : row.category}
            label={isProductLevel ? row.product_name : row.category}
            valueText={fmtWon(row.sales_amount)}
            pct={(row.sales_amount / maxVal) * 100}
            tone="blue"
          />
        ))}
      </div>
    </InsightCard>
  );
}

export function CategoryOrProductDetail({ data, selectedPath = [], onSelectCategory }) {
  const { isProductLevel, levelLabel } = categoryDrilldownInfo(selectedPath);
  const [search, setSearch] = useState('');
  const allRows = isProductLevel ? data.product_ranking : data.ranking;
  const q = search.trim().toLowerCase();
  const rows = (isProductLevel && q)
    ? allRows.filter((row) => row.product_name.toLowerCase().includes(q) || row.barcode.includes(q))
    : allRows;
  const chartItems = isProductLevel
    ? allRows.slice(0, 5).map((r) => ({ label: r.product_name, value: r.sales_amount }))
    : allRows.slice(0, 5).map((r) => ({ label: r.category, value: r.sales_amount }));
  return (
    <>
      {isProductLevel && (
        <input
          type="text"
          className="trc-search"
          placeholder="상품명 / 바코드 검색..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      )}
      <HorizontalBarChart items={chartItems} color="#3b82f6" formatValue={fmtWon} />
      <table className="trc-detail-table">
        <thead>
          {isProductLevel ? (
            <tr><th className="left">상품명</th><th>단위</th><th>판매금액</th></tr>
          ) : (
            <tr><th className="left">{levelLabel}</th><th>판매금액</th></tr>
          )}
        </thead>
        <tbody>
          {rows.map((row) => (
            isProductLevel ? (
              <tr key={row.sku_id}>
                <td className="left">{row.product_name}</td>
                <td>{row.option_code}</td>
                <td className="num">{fmtWon(row.sales_amount)}</td>
              </tr>
            ) : (
              <tr
                key={row.category}
                className="trc-clickable-row"
                onClick={() => onSelectCategory(row.category)}
              >
                <td className="left">{row.category}</td>
                <td className="num">{fmtWon(row.sales_amount)}</td>
              </tr>
            )
          ))}
          {isProductLevel && rows.length === 0 && (
            <tr><td colSpan={3} className="trc-empty-row">조건에 맞는 상품이 없습니다.</td></tr>
          )}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   2. 판매지역별 매출 TOP5
   ══════════════════════════════════════════════════════════════ */
export function RegionTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data?.top5 || [];
  const maxVal = items.length ? Math.max(...items.map((r) => r.sales_amount)) : 1;
  return (
    <InsightCard
      title="권역별 매출 현황"
      loading={loading}
      error={error}
      isEmpty={!data || items.length === 0}
      emptyMessage="해당 조회일에 판매 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((r) => (
          <BarRow
            key={`${r.sido}-${r.sigungu}`}
            label={r.sigungu}
            subLabel={r.sido}
            valueText={fmtWon(r.sales_amount)}
            pct={(r.sales_amount / maxVal) * 100}
            tone="blue"
          />
        ))}
      </div>
    </InsightCard>
  );
}

// 모달 상단 "표로 보기 / 지도로 보기" 토글 — 권역별 매출 현황 전용, InsightModal의 headerExtra로 얹는다.
export function RegionViewToggle({ value, onChange }) {
  return (
    <div className="trc-sort-toggle">
      <button type="button" className={`trc-sort-btn ${value === 'table' ? 'active' : ''}`} onClick={() => onChange('table')}>
        표로 보기
      </button>
      <button type="button" className={`trc-sort-btn ${value === 'map' ? 'active' : ''}`} onClick={() => onChange('map')}>
        지도로 보기
      </button>
    </div>
  );
}

// viewMode('table'|'map')는 부모(DashboardMain)가 RegionViewToggle과 함께 관리하는 controlled prop.
export function RegionDetail({ data, viewMode = 'table' }) {
  if (viewMode === 'map') {
    return <RegionMap data={data.ranking} />;
  }
  const chartItems = data.ranking.slice(0, 5).map((r) => ({ label: r.sigungu, value: r.sales_amount }));
  return (
    <>
      <HorizontalBarChart items={chartItems} color="#0ea5e9" formatValue={fmtWon} />
      <table className="trc-detail-table">
        <thead>
          <tr><th className="left">시도</th><th className="left">시군구</th><th>판매금액</th></tr>
        </thead>
        <tbody>
          {data.ranking.map((r) => (
            <tr key={`${r.sido}-${r.sigungu}`}>
              <td className="left">{r.sido}</td>
              <td className="left">{r.sigungu}</td>
              <td className="num">{fmtWon(r.sales_amount)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   3. AI 예측 상품 TOP5
   ══════════════════════════════════════════════════════════════ */
export function AiTop5Card({ data, loading, error, onOpenDetail }) {
  const items = (data || []).slice(0, 5);
  return (
    <InsightCard
      title="예측 출고량 상위 TOP 5"
      loading={loading}
      error={error}
      isEmpty={items.length === 0}
      emptyMessage="해당 기준주에 예측 가능한 상품이 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((p, idx) => (
          <div key={p.sku_id} className="trc-row trc-row-ranked">
            <span className="trc-rank-badge trc-rank-badge-purple">{idx + 1}</span>
            <span className="trc-name" title={p.product_name}>{p.product_name}</span>
            <span className="trc-val trc-val-mono">{fmtNum1(p.h1_pred)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function AiTop5Detail({ data }) {
  const chartItems = data.slice(0, 5).map((p) => ({ label: p.product_name, value: p.h1_pred ?? 0 }));
  return (
    <>
      <HorizontalBarChart items={chartItems} color="#8b5cf6" formatValue={(v) => v.toFixed(1)} />
      <table className="trc-detail-table trc-detail-table-wide">
        <thead>
          <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>기준 판매수량</th><th>1주 예상수요</th><th>2주 예상수요</th><th>4주 예상수요</th></tr>
        </thead>
        <tbody>
          {data.map((p) => (
            <tr key={p.sku_id}>
              <td className="left">{p.product_name}</td>
              <td className="left">{p.barcode}</td>
              <td>{p.option_code}</td>
              <td className="num">{p.recent_sales_qty === null ? '-' : fmtNum(p.recent_sales_qty)}</td>
              <td className="num">{p.h1_pred === null ? '-' : fmtNum1(p.h1_pred)}</td>
              <td className="num">{p.h2_pred === null ? '-' : fmtNum1(p.h2_pred)}</td>
              <td className="num">{p.h4_pred === null ? '-' : fmtNum1(p.h4_pred)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   4. 1주 예상수요 대비 재고 부족 TOP5
   ══════════════════════════════════════════════════════════════ */
export function ShortageTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data?.top5 || [];
  const maxVal = items.length ? Math.max(...items.map((s) => s.shortage_qty)) : 1;
  return (
    <InsightCard
      title="결품 위험 상품 TOP 5"
      loading={loading}
      error={error}
      isEmpty={!data || items.length === 0}
      emptyMessage="해당 기준주에 비교 가능한 재고/예측 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((s, idx) => (
          <BarRow
            key={s.sku_id}
            label={s.product_name}
            valueText={fmtNum(s.shortage_qty)}
            pct={(s.shortage_qty / maxVal) * 100}
            // 이미 shortage_qty 내림차순 정렬이므로 순위 그대로가 위험도다 — 1위(맨 위) = 진한
            // 빨강, 아래로 갈수록 앰버/노랑으로 옅어지는 연속 그라데이션.
            accentColor={shortageRankColor(idx, items.length)}
          />
        ))}
      </div>
    </InsightCard>
  );
}

export function ShortageDetail({ data }) {
  const chartItems = data.ranking.slice(0, 5).map((s) => ({ label: s.product_name, value: s.shortage_qty }));
  return (
    <>
      <HorizontalBarChart items={chartItems} color="#ef4444" formatValue={(v) => v.toFixed(1)} />
      <table className="trc-detail-table">
        <thead>
          <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>추정재고</th><th>1주 예상수요</th><th>부족수량</th></tr>
        </thead>
        <tbody>
          {data.ranking.map((s) => (
            <tr key={s.sku_id}>
              <td className="left">{s.product_name}</td>
              <td className="left">{s.barcode}</td>
              <td>{s.option_code}</td>
              <td className="num">{fmtNum(s.estimated_inventory)}</td>
              <td className="num">{fmtNum1(s.h1_pred)}</td>
              <td className="num">{fmtNum1(s.shortage_qty)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   5. 판매 증가 TOP5
   ══════════════════════════════════════════════════════════════ */
// 판매 급증 TOP5는 백엔드가 항상 "증가수량" 내림차순 전체 랭킹(data.ranking)을 내려주므로,
// "증가율순"은 그 전체 랭킹에서 클라이언트가 다시 골라 뽑는다(새 backend 없이 처리 가능).
// - 비교 기준 판매수량이 0이라 증가율을 계산할 수 없는 상품(increase_pct === null)은 제외.
// - 1개→2개처럼 소량 노이즈가 상단을 도배하지 않도록, 현재 판매수량이 이 값 미만이면 제외한다.
const SURGE_MIN_QTY_FOR_PCT = 5;

function topSurgeByPct(ranking, limit = 5) {
  return (ranking || [])
    .filter((r) => r.increase_pct !== null && r.increase_pct !== undefined && r.current_sales_qty >= SURGE_MIN_QTY_FOR_PCT)
    .sort((a, b) => b.increase_pct - a.increase_pct)
    .slice(0, limit);
}

function SurgeSortToggle({ value, onChange }) {
  return (
    <div className="trc-sort-toggle">
      <button type="button" className={`trc-sort-btn ${value === 'qty' ? 'active' : ''}`} onClick={() => onChange('qty')}>
        증가수량순
      </button>
      <button type="button" className={`trc-sort-btn ${value === 'pct' ? 'active' : ''}`} onClick={() => onChange('pct')}>
        증가율순
      </button>
    </div>
  );
}

export function SurgeTop5Card({ data, loading, error, onOpenDetail, sortMode = 'qty', onChangeSortMode }) {
  const isPct = sortMode === 'pct';
  const items = isPct ? topSurgeByPct(data?.ranking) : (data?.top5 || []);
  return (
    <InsightCard
      title="판매 급증 상품 TOP 5"
      headerExtra={<SurgeSortToggle value={sortMode} onChange={onChangeSortMode} />}
      loading={loading}
      error={error}
      isEmpty={!data || items.length === 0}
      emptyMessage={
        isPct
          ? `유효 표본(현재 판매수량 ${SURGE_MIN_QTY_FOR_PCT}개 이상)의 증가율 데이터가 없습니다.`
          : '전주 동일요일 대비 판매가 증가한 상품이 없습니다.'
      }
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((s) => (
          <div key={s.sku_id} className="trc-row">
            <span className="trc-name" title={s.product_name}>{s.product_name}</span>
            <span className="trc-badge-up">
              {isPct ? `▲ ${s.increase_pct.toFixed(1)}%` : `▲ ${fmtNum(s.increase_qty)}`}
            </span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

// initialSortMode: 메인 카드에서 고른 정렬('qty'|'pct')을 그대로 받아 모달 기본 정렬에 반영한다.
// 정렬 조작은 (헤더 클릭이 아니라) 메인 카드와 동일한 세그먼트 토글로 통일해 일관성을 준다.
export function SurgeDetail({ data, initialSortMode = 'qty' }) {
  const [sortMode, setSortMode] = useState(initialSortMode);

  const rows = [...data.ranking].sort((a, b) => {
    if (sortMode === 'pct') {
      // 증가율 null(비교 기준 판매수량 0)은 항상 맨 아래로 보낸다.
      if (a.increase_pct === null && b.increase_pct === null) return 0;
      if (a.increase_pct === null) return 1;
      if (b.increase_pct === null) return -1;
      return b.increase_pct - a.increase_pct;
    }
    return b.increase_qty - a.increase_qty;
  });

  return (
    <>
      <div className="trc-modal-sort-row">
        <SurgeSortToggle value={sortMode} onChange={setSortMode} />
      </div>
      <table className="trc-detail-table">
        <thead>
          <tr>
            <th className="left">상품명</th><th className="left">바코드</th><th>단위</th>
            <th>비교 기준 판매수량</th><th>현재 판매수량</th>
            <th>증가수량</th>
            <th>증가율</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.sku_id}>
              <td className="left">{s.product_name}</td>
              <td className="left">{s.barcode}</td>
              <td>{s.option_code}</td>
              <td className="num">{fmtNum(s.prev_sales_qty)}</td>
              <td className="num">{fmtNum(s.current_sales_qty)}</td>
              <td className="num">+{fmtNum(s.increase_qty)}</td>
              <td className="num">{s.increase_pct === null ? '-' : `${s.increase_pct.toFixed(1)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   6. 오늘 반품수량 TOP5 — daily/returns(ranking)를 반품수량 기준으로 재정렬해 쓴다
      (금액 정렬 endpoint를 그대로 재사용, 새 backend 없이 클라이언트에서만 정렬 변경).
   ══════════════════════════════════════════════════════════════ */
export function ReturnQtyTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data || [];
  return (
    <InsightCard
      title="반품 발생 상위 TOP 5"
      loading={loading}
      error={error}
      isEmpty={items.length === 0}
      emptyMessage="해당 조회일에 반품 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.slice(0, 5).map((r) => (
          <div key={r.sku_id} className="trc-row">
            <span className="trc-name" title={r.product_name}>{r.product_name}</span>
            <span className="trc-chip">{fmtNum(r.return_qty)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function ReturnQtyDetail({ data }) {
  return (
    <table className="trc-detail-table">
      <thead>
        <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>반품수량</th><th>반품금액</th></tr>
      </thead>
      <tbody>
        {data.map((r) => (
          <tr key={r.sku_id}>
            <td className="left">{r.product_name}</td>
            <td className="left">{r.barcode}</td>
            <td>{r.option_code}</td>
            <td className="num">{fmtNum(r.return_qty)}</td>
            <td className="num">{fmtWon(r.return_amount)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export { InsightModal };
