import { createContext, useContext, useEffect, useMemo, useReducer, type ReactNode } from 'react'
import seed from './data/seed.json'
import { sampleDetection, seeded, wrap } from './mock'
import type { Comment, Post } from './types'

interface SeedPhoto {
  id: string
  class_id: string
  image: string
  attribution: string
  source: string
  observed_on: string
  lat: number
  lng: number
  place: string
  observer: string
}

const NOMINAL = 1000 // seed boxes are expressed on a 1000×1000 canvas and drawn as percentages

function seedPosts(): Post[] {
  return (seed as SeedPhoto[]).map((s) => {
    const rand = seeded(s.id)
    const author = s.observer
    return {
      id: s.id,
      image: s.image,
      width: NOMINAL,
      height: NOMINAL,
      author,
      attribution: s.attribution,
      source: s.source,
      observed_on: s.observed_on,
      place: s.place,
      lat: s.lat,
      lng: s.lng,
      prediction: wrap(sampleDetection(s.class_id, NOMINAL, NOMINAL, s.id, true)),
      likes: Math.floor(rand() * 60),
      liked: false,
      comments: [],
      sample: true,
    }
  })
}

type Action =
  | { type: 'add'; post: Post }
  | { type: 'like'; id: string }
  | { type: 'comment'; id: string; comment: Comment }

interface Saved {
  mine: Post[]
  liked: string[]
  comments: Record<string, Comment[]>
}

const KEY = 'ucsc-wildlife:v1'

function load(): Saved {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return JSON.parse(raw) as Saved
  } catch {
    /* storage blocked or corrupt: start fresh */
  }
  return { mine: [], liked: [], comments: {} }
}

function init(): Post[] {
  const saved = load()
  return [...saved.mine, ...seedPosts()].map((p) => ({
    ...p,
    liked: saved.liked.includes(p.id),
    likes: p.likes + (saved.liked.includes(p.id) && p.sample ? 1 : 0),
    comments: p.sample ? saved.comments[p.id] ?? [] : p.comments,
  }))
}

function reducer(posts: Post[], a: Action): Post[] {
  switch (a.type) {
    case 'add':
      return [a.post, ...posts]
    case 'like':
      return posts.map((p) => (p.id === a.id ? { ...p, liked: !p.liked, likes: p.likes + (p.liked ? -1 : 1) } : p))
    case 'comment':
      return posts.map((p) => (p.id === a.id ? { ...p, comments: [...p.comments, a.comment] } : p))
  }
}

const Ctx = createContext<{ posts: Post[]; dispatch: (a: Action) => void } | null>(null)

export function StoreProvider({ children }: { children: ReactNode }) {
  const [posts, dispatch] = useReducer(reducer, undefined, init)
  useEffect(() => {
    const saved: Saved = {
      mine: posts.filter((p) => !p.sample),
      liked: posts.filter((p) => p.liked).map((p) => p.id),
      comments: Object.fromEntries(posts.filter((p) => p.sample && p.comments.length).map((p) => [p.id, p.comments])),
    }
    try {
      localStorage.setItem(KEY, JSON.stringify(saved))
    } catch {
      /* quota or blocked storage: keep working in memory */
    }
  }, [posts])
  const value = useMemo(() => ({ posts, dispatch }), [posts])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useStore() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useStore outside StoreProvider')
  return v
}

export function primaryOf(p: Post) {
  return p.prediction.primary ?? p.prediction.detections[0] ?? null
}
