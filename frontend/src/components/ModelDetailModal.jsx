import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import '../pages/analysis/analysis.css';

// 공통 "선택 모델 상세" 팝업 쉘. 모델별 분기는 전부 호출 측이 만든 sections로 넘어온다.
// sections: [{ id, heading, defaultOpen, body: ReactNode }]
export default function ModelDetailModal({ title, typeLabel, statusBadge, onClose, sections }) {
  const [activeSection, setActiveSection] = useState(sections[0]?.id);
  useEffect(() => {
    function onKeyDown(e) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  return createPortal(
    <div className="mdl-backdrop" onClick={(e) => { e.stopPropagation(); onClose(); }}>
      <div className="mdl-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="mdl-hd">
          <div className="mdl-hd-title">
            <h2>{title}</h2>
            {typeLabel && <span className="mdl-type-label">{typeLabel}</span>}
            {statusBadge && <span className={`mdl-badge mdl-badge-${statusBadge.tone}`}>{statusBadge.text}</span>}
          </div>
          <button type="button" className="mdl-close" onClick={onClose} aria-label="닫기">✕</button>
        </div>
        <nav className="mdl-tabs" aria-label="모델 상세 항목">
          {sections.map((section) => <button type="button" key={section.id} className={activeSection === section.id ? 'active' : ''} onClick={() => setActiveSection(section.id)}>{section.heading}</button>)}
        </nav>
        <div className="mdl-body">
          {sections.find((section) => section.id === activeSection)?.body}
        </div>
      </div>
    </div>,
    document.body,
  );
}
