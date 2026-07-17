import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { Projects } from '@/pages/Projects'
import { ProjectDetail } from '@/pages/ProjectDetail'
import { Annotate } from '@/pages/Annotate'
import { Review } from '@/pages/Review'

/**
 * Route table.
 *
 * AppShell is a *layout route*: it has no path of its own, it just wraps its
 * children with the sidebar chrome and renders them into its <Outlet />. New
 * pages in later phases are one <Route> line each and inherit the shell for
 * free.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
          <Route path="/projects/:id/annotate" element={<Annotate />} />
          {/* Two review routes: with an image id (deep-linkable — you can send
              someone a link to a specific image) and without (lands on the
              first image). */}
          <Route path="/projects/:id/review" element={<Review />} />
          <Route path="/projects/:id/review/:imageId" element={<Review />} />
          {/* Phase 4: /projects/:id/train
              Phase 5: /projects/:id/deploy   */}
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
