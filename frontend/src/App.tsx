import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { Projects } from '@/pages/Projects'
import { ProjectDetail } from '@/pages/ProjectDetail'
import { Annotate } from '@/pages/Annotate'
import { Review } from '@/pages/Review'
import { Visualize } from '@/pages/Visualize'
import { Health } from '@/pages/Health'
import { Train } from '@/pages/Train'
import { Evaluate } from '@/pages/Evaluate'
import { Deploy } from '@/pages/Deploy'

/**
 * Route table.
 *
 * AppShell is a *layout route*: it has no path of its own, it just wraps its
 * children with the sidebar chrome and renders them into its <Outlet />. New
 * pages in later phases are one <Route> line each and inherit the shell for
 * free.
 *
 * A DATA router (createBrowserRouter), not <BrowserRouter>: the review page
 * buffers unsaved box edits and needs useBlocker to intercept navigation away
 * from them — which only exists on data routers. The route tree itself is
 * unchanged, still authored as JSX via createRoutesFromElements.
 */
const router = createBrowserRouter(
  createRoutesFromElements(
    <Route element={<AppShell />}>
      <Route path="/" element={<Projects />} />
      <Route path="/projects/:id" element={<ProjectDetail />} />
      <Route path="/projects/:id/annotate" element={<Annotate />} />
      <Route path="/projects/:id/visualize" element={<Visualize />} />
      <Route path="/projects/:id/health" element={<Health />} />
      {/* Two review routes: with an image id (deep-linkable — you can send
          someone a link to a specific image) and without (lands on the
          first image). */}
      <Route path="/projects/:id/review" element={<Review />} />
      <Route path="/projects/:id/review/:imageId" element={<Review />} />
      <Route path="/projects/:id/train" element={<Train />} />
      <Route path="/projects/:id/evaluate" element={<Evaluate />} />
      <Route path="/projects/:id/deploy" element={<Deploy />} />
    </Route>,
  ),
)

export default function App() {
  return <RouterProvider router={router} />
}
