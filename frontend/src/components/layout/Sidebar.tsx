import { useEffect, useState } from 'react'
import { NavLink, useMatch } from 'react-router-dom'
import { getProject } from '@/lib/api'
import { LayoutGrid, Boxes, Tags, Cpu, PlayCircle, SquarePen, Eye } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/**
 * Primary navigation.
 *
 * A fixed left sidebar — the predictable, boring choice for a tool with
 * distinct workflow stages. It keeps every stage one click away and always
 * visible, which is what you want when the mental model is a pipeline:
 * dataset -> train -> deploy.
 *
 * STRUCTURE: annotation is nested UNDER Dataset, not beside it.
 *
 * Auto-annotate, Annotate and Visualize are all operations *on the dataset* —
 * they don't exist independently of it. Listing them as peers of Dataset implied
 * they were separate stages of the pipeline, which mis-taught the mental model:
 * you don't "go to Annotate", you annotate your dataset. The real top-level
 * stages are Dataset -> Train -> Deploy.
 *
 * The pipeline stages are PROJECT-SCOPED: "Annotate" is meaningless without
 * knowing which project's images to annotate. So they only become active once
 * a project is open, and are shown greyed out otherwise rather than hidden —
 * the shape of the whole workflow stays visible from the start.
 */

interface NavItem {
  /** Path suffix appended to /projects/:id, or '' for the project root. */
  suffix: string
  label: string
  icon: LucideIcon
  /** Built yet? */
  ready: boolean
  /** Rendered indented, as an operation on the section above it. */
  nested?: boolean
}

// Ordered to match the actual pipeline, not alphabetically.
const PROJECT_NAV: NavItem[] = [
  { suffix: '', label: 'Dataset', icon: Boxes, ready: true },
  { suffix: '/visualize', label: 'Visualize', icon: Eye, ready: true, nested: true },
  { suffix: '/annotate', label: 'Auto-annotate', icon: Tags, ready: true, nested: true },
  { suffix: '/review', label: 'Annotate', icon: SquarePen, ready: true, nested: true },
  { suffix: '/train', label: 'Train', icon: Cpu, ready: true },
  { suffix: '/deploy', label: 'Deploy', icon: PlayCircle, ready: false },
]

const linkClass = (isActive: boolean, nested = false) =>
  [
    'flex items-center gap-2.5 rounded-md py-1.5 text-sm transition-colors',
    // Indent + a hairline rule on the left: the conventional way to show "this
    // belongs to the thing above" without drawing a whole tree widget.
    nested ? 'ml-3 border-l border-gray-200 pl-3.5 pr-2.5' : 'px-2.5',
    isActive
      ? 'bg-accent-50 font-medium text-accent-700'
      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900',
  ].join(' ')

export function Sidebar() {
  // useMatch, NOT useParams.
  //
  // The Sidebar renders inside AppShell, which is a *pathless layout route*.
  // useParams only returns params matched by the route that renders the calling
  // component and its ancestors — and AppShell declares no path, so `:id` is
  // matched by the child route below it and useParams() here would be empty.
  //
  // useMatch tests the URL directly, independent of where in the tree we sit.
  // `end: false` makes it match /projects/1 as well as /projects/1/annotate, so
  // the nav stays highlighted across all of a project's sub-pages.
  const match = useMatch({ path: '/projects/:id', end: false })
  const id = match?.params.id
  const inProject = Boolean(id)

  // Fetched here rather than passed down: the Sidebar renders outside every
  // page (it's in the layout route), so there's no parent holding the project
  // to hand it one. One small request per project you open.
  const [projectName, setProjectName] = useState<string | null>(null)
  useEffect(() => {
    if (!id) {
      setProjectName(null)
      return
    }
    let cancelled = false
    getProject(Number(id))
      .then((p) => !cancelled && setProjectName(p.name))
      // A failed lookup shouldn't break navigation — the nav still works
      // without a name, so fall back to the placeholder rather than an error.
      .catch(() => !cancelled && setProjectName(null))
    return () => {
      cancelled = true
    }
  }, [id])

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 bg-white">
      {/* Wordmark. No logo art — a text mark is honest and doesn't pretend to
          be a product with a brand team. */}
      <div className="flex h-14 items-center border-b border-gray-200 px-4">
        <span className="text-sm font-semibold tracking-tight text-gray-900">CV Platform</span>
        <span className="ml-2 rounded border border-gray-200 px-1.5 py-0.5 font-mono text-[10px] text-gray-500">
          v0.1
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 p-2">
        <NavLink to="/" end className={({ isActive }) => linkClass(isActive)}>
          <LayoutGrid size={16} strokeWidth={2} />
          Projects
        </NavLink>

        {inProject && (
          <>
            {/* The project's NAME, not just the word "Project".
                Every page inside a project except the Dataset page had a
                generic title ("Train", "Auto-annotate"), so with several
                projects open there was nothing on screen saying which one you
                were working in. The sidebar is the one piece of chrome present
                on all of them, so the answer belongs here. */}
            <p className="label-eyebrow px-2.5 pb-1 pt-4">Project</p>
            <p
              className="truncate px-2.5 pb-1.5 text-sm font-medium text-gray-900"
              title={projectName ?? undefined}
            >
              {projectName ?? '…'}
            </p>
            {PROJECT_NAV.map(({ suffix, label, icon: Icon, ready, nested }) =>
              ready ? (
                <NavLink
                  key={label}
                  to={`/projects/${id}${suffix}`}
                  // `end` ONLY for the project root (suffix ''), where it stops
                  // "Dataset" matching every sub-route. The others need prefix
                  // matching so "Annotate" stays highlighted on /review/5.
                  end={suffix === ''}
                  className={({ isActive }) => linkClass(isActive, nested)}
                >
                  <Icon size={nested ? 14 : 16} strokeWidth={2} />
                  {label}
                </NavLink>
              ) : (
                <div
                  key={label}
                  className={[
                    'flex cursor-not-allowed items-center gap-2.5 rounded-md py-1.5 text-sm text-gray-300',
                    nested ? 'ml-3 border-l border-gray-200 pl-3.5 pr-2.5' : 'px-2.5',
                  ].join(' ')}
                  title="Not built yet"
                >
                  <Icon size={nested ? 14 : 16} strokeWidth={2} />
                  {label}
                </div>
              ),
            )}
          </>
        )}
      </nav>

      <div className="border-t border-gray-200 p-3">
        <p className="text-xs text-gray-400">Object detection · local</p>
      </div>
    </aside>
  )
}
