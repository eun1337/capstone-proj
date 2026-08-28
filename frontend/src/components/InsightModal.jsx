import { useEffect } from 'react';
import './InsightModal.css';

// 운영 인사이트 5개 카드가 공유하는 modal shell. 내용(children)만 카드별로 다르다.
// dateLabel/basisWeek: 대부분 조회일(일간)이지만 재고부족만 기준주(주간)이므로 호출부에서 지정한다.
export default function InsightModal({ title, center, dateLabel = '기준주', basisWeek, wide, onClose, children }) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  return (
    <div className="ins-modal-backdrop" onClick={onClose}>
      <div className={`ins-modal ${wide ? 'ins-modal-wide' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="ins-modal-hd">
          <div>
            <h3>{title}</h3>
            <p>{center}센터 · {dateLabel} {basisWeek}</p>
          </div>
          <button className="ins-modal-close" onClick={onClose} aria-label="닫기">✕</button>
        </div>
        <div className="ins-modal-body">{children}</div>
      </div>
    </div>
  );
}
