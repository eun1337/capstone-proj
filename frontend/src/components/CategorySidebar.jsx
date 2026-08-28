// 대시보드가 사용하는 KAN 표준분류 accordion 사이드바 — 항상 고정.
// 대/중/소분류 label에 검색어가 포함된 노드 + 그 조상 경로만 남기고 나머지는 제거한다.
// backend API 추가 없이 이미 받아온 /categories 응답을 프론트에서만 필터링한다.
function filterCategoryTree(tree, query) {
  const q = query.trim().toLowerCase();
  if (!q) return tree;

  function walk(nodes) {
    const out = [];
    for (const node of nodes) {
      const selfMatch = node.label.toLowerCase().includes(q);
      const filteredChildren = node.children ? walk(node.children) : undefined;
      if (selfMatch || (filteredChildren && filteredChildren.length > 0)) {
        out.push({ ...node, children: filteredChildren });
      }
    }
    return out;
  }
  return walk(tree);
}

// ── KAN 카테고리 트리 노드 (실제 /categories 응답 구조: {label, sku_count, children}) ──
function CategoryTreeNode({ node, path, depth, selectedPath, onSelect, openMap, onToggle, searchActive }) {
  const key       = path.join('>');
  const hasKids   = node.children?.length > 0;
  const isOpen    = searchActive || !!openMap[key];
  const isSelected = selectedPath.join('>') === key;

  return (
    <div>
      <button
        className={`cat-btn depth-${depth} ${isSelected ? 'cat-active' : ''}`}
        onClick={() => { if (hasKids && !searchActive) onToggle(key); onSelect(path); }}
      >
        <span className="cat-label-wrap">
          <span>{node.label}</span>
          <span className="cat-count">{node.sku_count.toLocaleString()}</span>
        </span>
        {hasKids && !searchActive && <span className={`cat-chevron ${isOpen ? 'open' : ''}`}>›</span>}
      </button>
      {hasKids && isOpen && (
        <div>
          {node.children.map((c) => (
            <CategoryTreeNode
              key={c.label} node={c} path={[...path, c.label]} depth={depth + 1}
              selectedPath={selectedPath} onSelect={onSelect}
              openMap={openMap} onToggle={onToggle} searchActive={searchActive}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function CategorySidebar({
  categoryTree, categoriesLoading, categoriesError,
  selectedPath, onSelectCategory,
  openMap, onToggleCategory,
  categorySearch, onCategorySearchChange,
  selectedSku, onOpenProductPicker, onClearSku,
}) {
  const displayedCategoryTree = filterCategoryTree(categoryTree, categorySearch);
  const searchActive = !!categorySearch.trim();
  const totalSkuCount = categoryTree.reduce((sum, n) => sum + n.sku_count, 0);

  return (
    <aside className="sidebar">
      <div className="sidebar-hd">
        <span>KAN 표준 분류</span>
      </div>
      <div className="sidebar-search">
        <input
          type="text"
          placeholder="카테고리 검색..."
          value={categorySearch}
          onChange={(e) => onCategorySearchChange(e.target.value)}
        />
      </div>
      <div className="cat-tree">
        {categoriesLoading && <div className="cat-hint">불러오는 중...</div>}
        {!categoriesLoading && categoriesError && (
          <div className="cat-hint cat-hint-error">{categoriesError}</div>
        )}
        {!categoriesLoading && !categoriesError && !searchActive && (
          <button
            className={`cat-btn depth-0 ${selectedPath.length === 0 ? 'cat-active' : ''}`}
            onClick={() => onSelectCategory([])}
          >
            <span className="cat-label-wrap">
              <span>전체</span>
              <span className="cat-count">{totalSkuCount.toLocaleString()}</span>
            </span>
          </button>
        )}
        {!categoriesLoading && !categoriesError && displayedCategoryTree.length === 0 && (
          <div className="cat-hint">검색 결과가 없습니다.</div>
        )}
        {!categoriesLoading && !categoriesError && displayedCategoryTree.map((node) => (
          <CategoryTreeNode
            key={node.label} node={node} path={[node.label]} depth={0}
            selectedPath={selectedPath} onSelect={onSelectCategory}
            openMap={openMap} onToggle={onToggleCategory}
            searchActive={searchActive}
          />
        ))}
      </div>

      <div className="sidebar-selected-product">
        <div className="sidebar-selected-hd">선택된 상품</div>
        {!selectedSku ? (
          <div className="sidebar-selected-empty">
            <div className="sidebar-selected-empty-icon">📦</div>
            <p>선택된 상품이 없습니다.</p>
            <button className="sidebar-find-btn" onClick={onOpenProductPicker}>상품 찾기 ›</button>
          </div>
        ) : (
          <div className="sidebar-selected-filled">
            <p className="sidebar-selected-name" title={selectedSku.product_name}>{selectedSku.product_name}</p>
            <p className="sidebar-selected-sub">
              {selectedSku.barcode} <span className="sidebar-selected-unit">{selectedSku.option_code}</span>
            </p>
            <div className="sidebar-selected-actions">
              <button className="sidebar-find-btn sidebar-find-btn-outline" onClick={onOpenProductPicker}>변경</button>
              <button className="sidebar-find-btn sidebar-find-btn-outline" onClick={onClearSku}>선택 해제</button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
