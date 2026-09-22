import { useMemo, useRef } from 'react';
import * as echarts from 'echarts';
import EChart from './charts/EChart.jsx';
import koreaGeo from '../data/korea-sigungu-simplified.geo.json';
import './RegionMap.css';

// data/korea-sigungu-simplified.geo.json: southkorea/southkorea-maps(kostat/2018/json/
// skorea-municipalities-2018-geo.json, KOSTAT 2018 시군구 전국 250개 원본 18MB)를
// mapshaper로 Douglas-Peucker 단순화(150m 허용오차, 위상 보존 — 인접 지역끼리 공유하는
// 경계선을 하나로 취급해 단순화 후에도 틈/겹침이 생기지 않는다)해 약 1MB로 줄인 로컬
// 사본이다. 전국 250개 시군구를 전부 담고 있어 매출 유무와 무관하게 항상 전체 윤곽이
// 그려진다. feature.properties.name은 "{시도} {시군구}"(예: "경상북도 포항시 남구")로
// 만들어 뒀고, RegionSalesItem의 sido/sigungu를 그대로 같은 형식으로 합치면 정확히
// 매칭된다 — 시도명은 2023~2024년 개편 이후 표기(강원특별자치도/전북특별자치도 등)로
// 통일했다(data/external/regional/postal_code_region_mapping_clean.csv의 표기와 대조 확인함).
const MAP_NAME = 'kr-sigungu-sales';
let mapRegistered = false;
function ensureMapRegistered() {
  if (!mapRegistered) {
    echarts.registerMap(MAP_NAME, koreaGeo);
    mapRegistered = true;
  }
}

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;

// 지도의 초기 배율/중심 — echarts 기본값(zoom=1)은 전국 250개 폴리곤 bbox 전체를 캔버스에
// 꽉 채우므로, 실제 매출 데이터가 몰려 있는 영남권(경북·경남·대구·부산·울산, 64개 활성
// 지역 중 47개)이 화면 한가운데 작게 찍힌다. 이 영남권 47개 지역의 bbox 중심을 좌표로,
// 그 bbox가 화면을 적당히 채우는 배율을 실측해 고정값으로 넣었다 — zoom/reset 버튼이
// 항상 이 값으로 돌아온다. (제주/서울/강원 등 나머지 소수 활성 지역은 초기 화면 밖에
// 있지만 팬/줌으로 바로 접근 가능하다.)
const INITIAL_ZOOM = 2.1;
const INITIAL_CENTER = [128.58, 35.84];

const ZOOM_FACTOR = 1.25;
const ZOOM_MIN = 1;
const ZOOM_MAX = 12;

// data: RegionSalesItem[] — [{ sido, sigungu, sales_amount, share }]
export default function RegionMap({ data }) {
  ensureMapRegistered();
  const echartsRef = useRef(null);

  const { seriesData, maxVal } = useMemo(() => {
    const rows = (data || []).map((r) => ({
      name: `${r.sido} ${r.sigungu}`,
      value: r.sales_amount,
      share: r.share,
    }));
    const maxVal = rows.length ? Math.max(...rows.map((r) => r.value)) : 1;
    return { seriesData: rows, maxVal };
  }, [data]);

  const option = useMemo(() => ({
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        if (p.value === undefined || p.value === null || Number.isNaN(p.value)) {
          return `<strong>${p.name}</strong><br/>매출 데이터 없음`;
        }
        const share = p.data && p.data.share;
        const shareLine = (share !== undefined && share !== null)
          ? `<br/>전체 대비 비중 ${(share * 100).toFixed(1)}%`
          : '';
        return `<strong>${p.name}</strong><br/>매출액 ${fmtWon(p.value)}${shareLine}`;
      },
    },
    visualMap: {
      type: 'continuous',
      min: 0,
      max: maxVal || 1,
      calculable: false,
      orient: 'horizontal',
      right: 4,
      bottom: 2,
      itemWidth: 10,
      itemHeight: 80,
      text: [fmtWon(maxVal), '₩0'],
      textStyle: { fontSize: 9, color: '#64748b' },
      inRange: { color: ['#EFF6FF', '#93C5FD', '#3B82F6', '#1D4ED8', '#1E3A8A'] },
    },
    series: [
      {
        type: 'map',
        map: MAP_NAME,
        roam: true,
        zoom: INITIAL_ZOOM,
        center: INITIAL_CENTER,
        scaleLimit: { min: ZOOM_MIN, max: ZOOM_MAX },
        aspectScale: 1.05,
        label: { show: false },
        itemStyle: {
          // 매출 데이터 없는(이번 조회 범위에 0인) 지역 — 아주 연한 회색 배경 + 은은한
          // 슬레이트 경계선. 매출 있는 지역은 visualMap의 블루 그라데이션이 이 areaColor를
          // 덮어써서 실제로는 적용되지 않는다.
          areaColor: '#F1F5F9',
          borderColor: '#CBD5E1',
          borderWidth: 0.8,
        },
        emphasis: {
          label: { show: false },
          itemStyle: { borderColor: '#1E3A8A', borderWidth: 1.5 },
        },
        data: seriesData,
      },
    ],
  }), [seriesData, maxVal]);

  function getInstance() {
    return echartsRef.current?.getEchartsInstance?.() ?? null;
  }

  function zoomBy(factor, originX, originY) {
    const inst = getInstance();
    if (!inst) return;
    const width = inst.getWidth();
    const height = inst.getHeight();
    inst.dispatchAction({
      type: 'geoRoam',
      componentType: 'series',
      seriesIndex: 0,
      zoom: factor,
      originX: originX ?? width / 2,
      originY: originY ?? height / 2,
    });
  }

  function handleZoomIn() { zoomBy(ZOOM_FACTOR); }
  function handleZoomOut() { zoomBy(1 / ZOOM_FACTOR); }

  function handleZoomReset() {
    const inst = getInstance();
    if (!inst) return;
    // 사용자가 휠/드래그로 옮겨 놓은 zoom/center를 초기값으로 되돌린다. option prop 쪽은
    // notMerge라 여기서 건드리지 않고, 인스턴스에 직접 부분 병합(setOption merge)한다.
    inst.setOption({ series: [{ zoom: INITIAL_ZOOM, center: INITIAL_CENTER }] });
  }

  // 더블클릭한 화면 좌표를 그대로 확대 중심(originX/Y)으로 써서 "그 위치를 확대"한다.
  function handleDblClick(params) {
    const originX = params?.event?.offsetX;
    const originY = params?.event?.offsetY;
    zoomBy(ZOOM_FACTOR, originX, originY);
  }

  return (
    <div className="rmap-wrap">
      <EChart
        ref={echartsRef}
        option={option}
        height={420}
        onEvents={{ dblclick: handleDblClick }}
      />
      <div className="rmap-zoom-ctrl">
        <button type="button" className="rmap-zoom-btn" onClick={handleZoomIn} title="확대" aria-label="지도 확대">+</button>
        <button type="button" className="rmap-zoom-btn" onClick={handleZoomOut} title="축소" aria-label="지도 축소">&minus;</button>
        <button type="button" className="rmap-zoom-btn rmap-zoom-reset" onClick={handleZoomReset} title="화면 맞춤" aria-label="지도 배율 초기화">⟲</button>
      </div>
    </div>
  );
}
