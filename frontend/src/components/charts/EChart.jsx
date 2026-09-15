import ReactECharts from 'echarts-for-react';

export default function EChart({ option, height = 260, onEvents, fill = false, renderer = 'svg' }) {
  const style = fill
    ? { height: '100%', width: '100%', minHeight: 0, flex: 1 }
    : { height, width: '100%' };
  return (
    <ReactECharts
      option={option}
      style={style}
      opts={{ renderer }}
      notMerge
      onEvents={onEvents}
    />
  );
}
