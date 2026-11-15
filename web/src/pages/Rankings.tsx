import { useMemo, useState } from 'react'
import { labelOf, PageTitle, RatingBadge } from '../components'
import { GROUP_LABEL } from '../data/species'
import { href } from '../router'
import { primaryOf, useStore } from '../store'
import type { Group } from '../types'

type Filter = 'all' | Group

export function Rankings() {
  const { posts } = useStore()
  const [filter, setFilter] = useState<Filter>('all')
  const ranked = useMemo(
    () =>
      posts
        .map((p) => ({ p, d: primaryOf(p) }))
        .filter(({ d }) => d?.rating != null && (filter === 'all' || d.species?.group === filter))
        .sort((a, b) => b.d!.rating! - a.d!.rating!)
        .slice(0, 25),
    [posts, filter],
  )
  const filters: Filter[] = ['all', 'large_mammal', 'small_mammal', 'bird']

  return (
    <div className="mx-auto max-w-3xl">
      <PageTitle sub="Ranked by photo rating: 65% photo quality and 35% how confidently the model identified the animal. Uncertain IDs aren’t ranked.">Top rankings</PageTitle>
      <div className="mb-6 flex flex-wrap gap-2" role="radiogroup" aria-label="Filter by group">
        {filters.map((f) => (
          <button key={f} role="radio" aria-checked={filter === f} onClick={() => setFilter(f)} className="rounded-full border border-line px-4 py-1.5 text-[14px] text-canopy-soft aria-checked:border-canopy aria-checked:bg-canopy aria-checked:text-fog">
            {f === 'all' ? 'Everything' : GROUP_LABEL[f]}
          </button>
        ))}
      </div>
      <ol className="space-y-3">
        {ranked.map(({ p, d }, i) => (
          <li key={p.id}>
            <a href={href.post(p.id)} className="flex items-center gap-4 rounded-2xl bg-white p-3 pr-5 hover:ring-2 hover:ring-lichen">
              <span className={`w-10 shrink-0 text-center font-display font-extrabold tabular-nums ${i < 3 ? 'text-4xl text-redwood' : 'text-2xl text-lichen'}`}>{i + 1}</span>
              <img src={p.image} alt="" loading="lazy" className={`shrink-0 rounded-xl object-cover ${i === 0 ? 'size-28' : 'size-20'}`} />
              <div className="min-w-0 flex-1">
                <p className="truncate font-display text-lg font-bold">{labelOf(d!)}</p>
                <p className="truncate text-[14px] text-canopy-soft">{p.author}</p>
              </div>
              <RatingBadge rating={d!.rating} />
            </a>
          </li>
        ))}
      </ol>
      {posts.some((p) => p.sample) && <p className="mt-6 text-[13px] text-canopy-soft">Ratings on demo posts are sample values until the model API is connected.</p>}
    </div>
  )
}
