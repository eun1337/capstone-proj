import { useMemo } from 'react';
import * as echarts from 'echarts';
import EChart from './charts/EChart.jsx';
import koreaGeo from '../data/korea-sigungu-simplified.geo.json';

// data/korea-sigungu-simplified.geo.json: southkorea/southkorea-maps(KOSTAT 2018, 시군구)
// GeoJSON(18MB)에서 실제 매출 데이터에 등장하는 64개 시군구만 골라내고(전국 250개 중),
// Douglas-Peucker로 좌표를 단순화(150m 허용오차)해 약 244KB로 줄인 로컬 사본이다.
// feature.properties.name은 "{시도} {시군구}"(예: "경상북도 포항시 남구")로 만들어 뒀고,
// RegionSalesItem의 sido/sigungu를 그대로 같은 형식으로 합치면 정확히 매칭된다.
const MAP_NAME = 'kr-sigungu-sales';
let mapRegistered = false;
function ensureMapRegistered() {
  if (!mapRegistered) {
    echarts.registerMap(MAP_NAME, koreaGeo);
    mapRegistered = true;
  }
}

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;

// data: RegionSalesItem[] — [{ sido, sigungu, sales_amount, share }]
export default function RegionMap({ data }) {
  ensureMapRegistered();

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
          return `<strong>${p.name}</strong><br/>해당 조회 범위에 매출 데이터가 없습니다`;
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
        roam: false,
        aspectScale: 1.05,
        label: { show: false },
        itemStyle: {
          areaColor: '#F1F5F9', // 매출 데이터 없는 지역(회색)
          borderColor: '#fff',
          borderWidth: 0.6,
        },
        emphasis: {
          label: { show: false },
          itemStyle: { borderColor: '#1e293b', borderWidth: 1.5 },
        },
        data: seriesData,
      },
    ],
  }), [seriesData, maxVal]);

  return <EChart option={option} height={420} />;
}
