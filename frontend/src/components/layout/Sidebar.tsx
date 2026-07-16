import { NavLink } from 'react-router-dom'
import { LayoutGrid, Boxes, Tags, Cpu, PlayCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/**
 * Primary navigation.
 *
 * A fixed left sidebar — the predictable, boring choice for a tool with
 * distinct workflow stages. It keeps every stage one click away and always
 * visible, which is exactly what you want when the mental model is a pipeline:
 * dataset -> annotate -> train -> deploy.
 */

interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  /** Stages that don't exist yet are shown disabled rather than hidden, so the
   *  shape of the full workflow is visible from Phase 0 onward. */
  enabled: boolean
}

// Ordered to match the actual pipeline, not alphabetically.
const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Projects', icon: LayoutGrid, enabled: true },
  { to: '/dataset', label: 'Dataset', icon: Boxes, enabled: false },
  { to: '/annotate', label: 'Annotate', icon: Tags, enabled: false },
  { to: '/train', label: 'Train', icon: Cpu, enabled: false },
  { to: '/deploy', label: 'Deploy', icon: PlayCircle, enabled: false },
]

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 bg-white">
      {/* Wordmark. No logo art — a text mark is honest and doesn't pretend to
          be a product with a brand team. */}
      <div className="flex h-14 items-center border-b border-gray-200 px-4">
        <span className="text-sm font-semibold tracking-tight text-gray-900">
          CV Platform
        </span>
        <span className="ml-2 rounded border border-gray-200 px-1.5 py-0.5 font-mono text-[10px] text-gray-500">
          v0.1
        </span>
      </div>

      <nav className="flex-1 space-y-0.5 p-2">
        {NAV_ITEMS.map(({ to, label, icon: Icon, enabled }) =>
          enabled ? (
            <NavLink
              key={to}
              to={to}
              end
              // NavLink hands us `isActive`, so the active route styles itself
              // instead of us tracking current location by hand.
              className={({ isActive }) =>
                [
                  'flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors',
                  isActive
                    ? 'bg-accent-50 font-medium text-accent-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900',
                ].join(' ')
              }
            >
              <Icon size={16} strokeWidth={2} />
              {label}
            </NavLink>
          ) : (
            <div
              key={to}
              className="flex cursor-not-allowed items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-gray-300"
              title="Not built yet"
            >
              <Icon size={16} strokeWidth={2} />
              {label}
            </div>
          ),
        )}
      </nav>

      <div className="border-t border-gray-200 p-3">
        <p className="text-xs text-gray-400">Object detection · local</p>
      </div>
    </aside>
  )
}
