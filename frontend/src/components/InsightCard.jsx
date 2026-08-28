import './InsightCard.css';

// 운영 인사이트 5개 카드가 공유하는 compact 카드 shell.
// 카드 전체(상세보기 버튼 제외 영역 포함) 클릭 또는 버튼 클릭으로 모달을 연다.
// headerExtra: 수량 기반 ranking 카드(판매증가/재고부족)의 단위 필터 chip처럼, loading/error/
// empty 상태와 무관하게 항상 보여야 하는 보조 컨트롤. 카드 클릭(모달 열기)과 분리하기 위해
// 별도 stopPropagation 래퍼로 감싼다.
export default function InsightCard({ title, headerExtra, loading, error, isEmpty, emptyMessage, onOpenDetail, children }) {
  return (
    <div className="ins-card" onClick={onOpenDetail} role="button" tabIndex={0}>
      <div className="ins-card-hd">
        <h4>{title}</h4>
      </div>

      {/* 단위 필터 chip이 없는 카드(카테고리/지역/반품)도 이 슬롯을 항상 같은 높이로
          차지해, 5개 카드의 본문(ranking list) 시작 위치가 전부 같은 선상에 오게 한다. */}
      <div className="ins-card-header-extra" onClick={(e) => e.stopPropagation()}>
        {headerExtra}
      </div>

      {/* title → content(flex:1) → footer 구조 — 내용 길이와 무관하게 상세보기 버튼이
          항상 카드 하단 동일 위치에 오도록, 남는 세로 공간을 이 content 영역이 흡수한다. */}
      <div className="ins-card-content">
        {loading && <div className="ins-hint">불러오는 중...</div>}

        {!loading && error && (
          <div className="ins-hint ins-hint-error">
            <p>데이터를 불러오지 못했습니다.</p>
            <p className="ins-hint-sub">{error}</p>
          </div>
        )}

        {!loading && !error && isEmpty && (
          <div className="ins-hint">{emptyMessage || '표시할 데이터가 없습니다.'}</div>
        )}

        {!loading && !error && !isEmpty && (
          <div className="ins-card-body">{children}</div>
        )}
      </div>

      {!loading && !error && !isEmpty && (
        <button className="ins-detail-btn" onClick={(e) => { e.stopPropagation(); onOpenDetail(); }}>
          상세보기 →
        </button>
      )}
    </div>
  );
}
