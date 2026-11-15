import type { ReactNode } from 'react'
import { BY_ID, GROUP_ROLLUP } from './data/species'
import { href, type Route } from './router'
import type { Detection, Group, Post, Quality } from './types'

/* ---------- icons (24px, stroke = currentColor) ---------- */
const Svg = ({ children, className = 'size-6' }: { children: ReactNode; className?: string }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
    {children}
  </svg>
)
export const Icon = {
  feed: (c?: string) => <Svg className={c}><path d="M4 10.5 12 4l8 6.5V20h-5v-6H9v6H4z" /></Svg>,
  species: (c?: string) => <Svg className={c}><rect x="4" y="4" width="7" height="7" rx="1.5" /><rect x="13" y="4" width="7" height="7" rx="1.5" /><rect x="4" y="13" width="7" height="7" rx="1.5" /><rect x="13" y="13" width="7" height="7" rx="1.5" /></Svg>,
  rankings: (c?: string) => <Svg className={c}><path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0zM7 6H4v1a3 3 0 0 0 3 3M17 6h3v1a3 3 0 0 1-3 3" /></Svg>,
  identify: (c?: string) => <Svg className={c}><path d="M4 9V5h4M16 5h4v4M20 15v4h-4M8 19H4v-4" /><circle cx="12" cy="12" r="3" /></Svg>,
  heart: (c?: string, filled?: boolean) => <Svg className={c}><path fill={filled ? 'currentColor' : 'none'} d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z" /></Svg>,
  comment: (c?: string) => <Svg className={c}><path d="M5 18.5V6.5A1.5 1.5 0 0 1 6.5 5h11A1.5 1.5 0 0 1 19 6.5v8a1.5 1.5 0 0 1-1.5 1.5H9z" /></Svg>,
  back: (c?: string) => <Svg className={c}><path d="M15 5l-7 7 7 7" /></Svg>,
  pin: (c?: string) => <Svg className={c}><path d="M12 21s6-5.6 6-11a6 6 0 0 0-12 0c0 5.4 6 11 6 11z" /><circle cx="12" cy="10" r="2" /></Svg>,
}

/* ---------- navigation: top bar on desktop, tab bar on phones ---------- */
const TABS = [
  { key: 'feed', label: 'Home', to: href.feed, icon: Icon.feed },
  { key: 'species', label: 'Species', to: href.species, icon: Icon.species },
  { key: 'rankings', label: 'Top rankings', to: href.rankings, icon: Icon.rankings },
] as const

function activeTab(r: Route) {
  if (r.name === 'species' || r.name === 'speciesDetail') return 'species'
  if (r.name === 'rankings') return 'rankings'
  if (r.name === 'identify') return 'identify'
  return 'feed'
}

export function TopNav({ route }: { route: Route }) {
  const cur = activeTab(route)
  return (
    <header className="sticky top-0 z-[1000] hidden border-b border-line bg-fog/90 backdrop-blur md:block">
      <nav className="mx-auto flex h-16 max-w-6xl items-center gap-10 px-8" aria-label="Main">
        <a href={href.feed} className="flex items-center gap-2 font-display text-xl font-extrabold tracking-tight">
          <img src="/favicon.svg" alt="" className="size-7" />
          Campus Wildlife
        </a>
        <ul className="flex gap-1">
          {TABS.map((t) => (
            <li key={t.key}>
              <a
                href={t.to}
                aria-current={cur === t.key ? 'page' : undefined}
                className="rounded-full px-4 py-2 text-[15px] text-canopy-soft hover:text-canopy aria-[current=page]:bg-canopy aria-[current=page]:text-fog"
              >
                {t.label}
              </a>
            </li>
          ))}
        </ul>
        <a href={href.identify} className="ml-auto flex items-center gap-2 rounded-full bg-redwood px-5 py-2.5 font-bold text-white hover:bg-redwood-dark">
          {Icon.identify('size-5')} Identify an animal
        </a>
      </nav>
    </header>
  )
}

export function TabBar({ route }: { route: Route }) {
  const cur = activeTab(route)
  const item = (key: string, label: string, to: string, icon: ReactNode) => (
    <a href={to} aria-current={cur === key ? 'page' : undefined} className="flex flex-1 flex-col items-center gap-0.5 py-2 text-[11px] text-canopy-soft aria-[current=page]:text-canopy aria-[current=page]:font-bold">
      {icon}
      {label}
    </a>
  )
  return (
    <nav className="fixed inset-x-0 bottom-0 z-[1000] border-t border-line bg-fog/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden" aria-label="Main">
      <div className="flex items-end">
        {item('feed', 'Home', href.feed, Icon.feed())}
        {item('species', 'Species', href.species, Icon.species())}
        <a href={href.identify} aria-current={cur === 'identify' ? 'page' : undefined} className="-mt-5 flex flex-1 flex-col items-center gap-0.5 pb-2 text-[11px] font-bold text-redwood">
          <span className="grid size-14 place-items-center rounded-full bg-redwood text-white shadow-lg ring-4 ring-fog">{Icon.identify('size-7')}</span>
          Identify
        </a>
        {item('rankings', 'Ranked', href.rankings, Icon.rankings())}
      </div>
    </nav>
  )
}

/* ---------- photo with the detector's brackets ---------- */
export function DetectionPhoto({
  post,
  showTags = true,
  animate = false,
  className = '',
  selected,
  onSelect,
}: {
  post: Pick<Post, 'image' | 'width' | 'height' | 'prediction'>
  showTags?: boolean
  animate?: boolean
  className?: string
  selected?: number
  onSelect?: (i: number) => void
}) {
  const { width, height } = post
  return (
    <div className={`relative overflow-hidden bg-canopy ${className}`}>
      <img src={post.image} alt={altText(post.prediction.detections)} className="block h-auto w-full" loading="lazy" />
      {post.prediction.detections.map((d, i) => {
        const [x1, y1, x2, y2] = d.box
        const style = { left: `${(x1 / width) * 100}%`, top: `${(y1 / height) * 100}%`, width: `${((x2 - x1) / width) * 100}%`, height: `${((y2 - y1) / height) * 100}%` }
        const dim = selected !== undefined && selected !== i
        const Tag = onSelect ? 'button' : 'div'
        return (
          <Tag
            key={i}
            type={onSelect ? 'button' : undefined}
            onClick={onSelect ? () => onSelect(i) : undefined}
            className={`bracket absolute ${animate ? 'lock-on' : ''} ${dim ? 'opacity-40' : ''}`}
            data-status={d.status}
            style={{ ...style, animationDelay: animate ? `${i * 120}ms` : undefined }}
            aria-label={onSelect ? `Show animal ${i + 1}: ${labelOf(d)}` : undefined}
          >
            {showTags && (
              <span className="absolute -top-px left-0 -translate-y-full whitespace-nowrap rounded-t-md bg-slug px-2 py-0.5 text-[12px] font-bold text-canopy">
                {labelOf(d)}
              </span>
            )}
          </Tag>
        )
      })}
    </div>
  )
}

export function labelOf(d: Detection): string {
  if (d.status === 'identified' && d.species) return d.species.common_name
  if (d.status === 'uncertain' && d.coarse_group_fallback) return GROUP_ROLLUP[d.coarse_group_fallback as Group] ?? 'Species uncertain'
  if (d.status === 'uncertain') return 'Species uncertain'
  if (d.status === 'unsupported') return 'Not one of the 20 species'
  return 'No animal found'
}

function altText(ds: Detection[]) {
  if (!ds.length) return 'Photo with no animal detected'
  return `Photo of ${ds.map(labelOf).join(', ')}`
}

/* ---------- result details ---------- */
export function TopThree({ d }: { d: Detection }) {
  return (
    <ol className="space-y-2.5">
      {d.top_3.map((c, i) => {
        const name = c.common_name ?? BY_ID[c.class_id]?.common_name ?? c.class_id
        const pct = Math.round(c.probability * 100)
        return (
          <li key={c.class_id}>
            <div className="mb-1 flex justify-between text-[15px]">
              <span className={i === 0 ? 'font-bold' : 'text-canopy-soft'}>{name}</span>
              <span className="tabular-nums text-canopy-soft">{pct}%</span>
            </div>
            <div className="h-2 rounded-full bg-line">
              <div className={`h-2 rounded-full ${i === 0 ? 'bg-canopy' : 'bg-lichen'}`} style={{ width: `${Math.max(pct, 2)}%` }} />
            </div>
          </li>
        )
      })}
    </ol>
  )
}

const Q_LABELS: [keyof Quality, string][] = [
  ['sharpness', 'Sharpness'],
  ['visibility', 'Animal visible'],
  ['exposure', 'Exposure'],
  ['resolution', 'Resolution'],
  ['composition', 'Composition'],
]

export function QualityBars({ q }: { q: Quality }) {
  return (
    <dl className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-2 text-[14px]">
      {Q_LABELS.map(([k, label]) => (
        <div key={k} className="contents">
          <dt className="text-canopy-soft">{label}</dt>
          <dd className="h-1.5 rounded-full bg-line">
            <div className="h-1.5 rounded-full bg-canopy-soft" style={{ width: `${q[k]}%` }} />
          </dd>
          <dd className="w-7 text-right tabular-nums">{q[k]}</dd>
        </div>
      ))}
    </dl>
  )
}

export function RatingBadge({ rating, size = 'md' }: { rating: number | null; size?: 'md' | 'lg' }) {
  if (rating === null) return null
  const big = size === 'lg'
  return (
    <span className={`inline-flex items-baseline gap-1 rounded-full bg-slug font-display font-extrabold text-canopy ${big ? 'px-4 py-1.5 text-2xl' : 'px-2.5 py-0.5 text-[15px]'}`} title="Photo rating: 65% photo quality, 35% identification confidence">
      {rating}
      <span className={`font-sans font-normal ${big ? 'text-sm' : 'text-[11px]'}`}>/100</span>
    </span>
  )
}

export function SampleNote({ className = '' }: { className?: string }) {
  return (
    <p className={`text-[13px] text-canopy-soft ${className}`}>
      Demo post. The photo and its species come from iNaturalist; the boxes, confidence and rating are sample values until the model API is connected.
    </p>
  )
}

export function PageTitle({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-6">
      <h1 className="font-display text-[34px] leading-tight font-extrabold tracking-tight md:text-[44px]">{children}</h1>
      {sub && <p className="mt-1 max-w-[60ch] text-canopy-soft">{sub}</p>}
    </div>
  )
}

export function timeAgo(date: string) {
  if (!date) return ''
  const d = new Date(date)
  const days = Math.floor((Date.now() - d.getTime()) / 86400000)
  if (days < 1) return 'today'
  if (days < 2) return 'yesterday'
  if (days < 30) return `${days} days ago`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}
