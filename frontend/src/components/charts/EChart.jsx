import { forwardRef } from 'react';
import ReactECharts from 'echarts-for-react';

// ref는 echarts-for-react의 인스턴스로 그대로 전달된다 — 호출부는
// ref.current.getEchartsInstance()로 실제 echarts 인스턴스를 얻어 dispatchAction 등
// (예: RegionMap의 zoom/pan 컨트롤 버튼)을 직접 호출할 수 있다.
const EChart = forwardRef(function EChart({ option, height = 260, onEvents, fill = false, renderer = 'svg' }, ref) {
  const style = fill
    ? { height: '100%', width: '100%', minHeight: 0, flex: 1 }
    : { height, width: '100%' };
  return (
    <ReactECharts
      ref={ref}
      option={option}
      style={style}
      opts={{ renderer }}
      notMerge
      onEvents={onEvents}
    />
  );
});

export default EChart;
