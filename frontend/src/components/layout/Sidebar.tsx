import { NavLink, useMatch } from 'react-router-dom'
import { LayoutGrid, Boxes, Tags, Cpu, PlayCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/**
 * Primary navigation.
 *
 * A fixed left sidebar — the predictable, boring choice for a tool with
 * distinct workflow stages. It keeps every stage one click away and always
 * visible, which is what you want when the mental model is a pipeline:
 * dataset -> annotate -> train -> deploy.
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
}

// Ordered to match the actual pipeline, not alphabetically.
const PROJECT_NAV: NavItem[] = [
  { suffix: '', label: 'Dataset', icon: Boxes, ready: true },
  { suffix: '/annotate', label: 'Annotate', icon: Tags, ready: false },
  { suffix: '/train', label: 'Train', icon: Cpu, ready: false },
  { suffix: '/deploy', label: 'Deploy', icon: PlayCircle, ready: false },
]

const linkClass = (isActive: boolean) =>
  [
    'flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors',
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
            <p className="label-eyebrow px-2.5 pb-1 pt-4">Project</p>
            {PROJECT_NAV.map(({ suffix, label, icon: Icon, ready }) =>
              ready ? (
                <NavLink
                  key={label}
                  to={`/projects/${id}${suffix}`}
                  end
                  className={({ isActive }) => linkClass(isActive)}
                >
                  <Icon size={16} strokeWidth={2} />
                  {label}
                </NavLink>
              ) : (
                <div
                  key={label}
                  className="flex cursor-not-allowed items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-gray-300"
                  title="Not built yet"
                >
                  <Icon size={16} strokeWidth={2} />
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
