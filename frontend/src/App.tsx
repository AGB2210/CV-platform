import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { Projects } from '@/pages/Projects'
import { ProjectDetail } from '@/pages/ProjectDetail'
import { Annotate } from '@/pages/Annotate'

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
          {/* Phase 3: /projects/:id/annotate/:imageId  (review canvas)
              Phase 4: /projects/:id/train
              Phase 5: /projects/:id/deploy   */}
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
