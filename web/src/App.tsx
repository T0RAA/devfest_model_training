import { TabBar, TopNav } from './components'
import { useRoute } from './router'
import { StoreProvider } from './store'
import { Feed } from './pages/Feed'
import { Identify } from './pages/Identify'
import { PostView } from './pages/PostView'
import { Rankings } from './pages/Rankings'
import { SpeciesDetail, SpeciesIndex } from './pages/Species'

export default function App() {
  const route = useRoute()
  return (
    <StoreProvider>
      <TopNav route={route} />
      <main className="mx-auto max-w-6xl px-4 pt-4 pb-28 md:px-8 md:pt-8 md:pb-16">
        {route.name === 'feed' && <Feed />}
        {route.name === 'post' && <PostView id={route.id} />}
        {route.name === 'species' && <SpeciesIndex />}
        {route.name === 'speciesDetail' && <SpeciesDetail classId={route.classId} />}
        {route.name === 'rankings' && <Rankings />}
        {route.name === 'identify' && <Identify />}
      </main>
      <TabBar route={route} />
    </StoreProvider>
  )
}
