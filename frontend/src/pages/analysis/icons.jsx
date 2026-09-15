const BASE_PROPS = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

export function IconHome({ className }) {
  return (
    <svg className={className} {...BASE_PROPS}>
      <path d="M3 11.5 12 4l9 7.5" />
      <path d="M5.5 10v9h13v-9" />
    </svg>
  );
}

export function IconBarChart({ className }) {
  return (
    <svg className={className} {...BASE_PROPS}>
      <line x1="5" y1="20" x2="5" y2="12" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="19" y1="20" x2="19" y2="15" />
    </svg>
  );
}

export function IconLayers({ className }) {
  return (
    <svg className={className} {...BASE_PROPS}>
      <polygon points="12 3 21 8 12 13 3 8 12 3" />
      <polyline points="3 16 12 21 21 16" />
      <polyline points="3 12 12 17 21 12" />
    </svg>
  );
}

export function IconAward({ className }) {
  return (
    <svg className={className} {...BASE_PROPS}>
      <circle cx="12" cy="8" r="6" />
      <path d="M8.2 13.5 7 21l5-2.6 5 2.6-1.2-7.5" />
    </svg>
  );
}

export function IconMessage({ className }) {
  return (
    <svg className={className} {...BASE_PROPS}>
      <path d="M4 5h16v11H8l-4 4V5z" />
    </svg>
  );
}
