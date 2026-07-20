import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowDownUp, Plus, Search, Trash2, X } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { ConfirmDialog, Modal } from '@/components/ui/Modal'
import {
  bulkDeleteProjects,
  createProject,
  deleteProject,
  listProjects,
  type Project,
} from '@/lib/api'

/** How the list can be ordered. The value is the storage key too. */
const SORTS = {
  activity: 'Last modified',
  created: 'Date created',
  name: 'Name',
} as const

type SortKey = keyof typeof SORTS

/** Sort choice persists across visits.
 *
 *  The list is the app's home, so it's re-entered constantly — dropping back to
 *  the default every time would make a chosen order feel like it hadn't stuck.
 *  localStorage rather than a URL param because it's a preference, not a view
 *  worth linking to. Reads are guarded: a value from an older build (or a user
 *  poking at devtools) must not throw on load. */
const SORT_STORE = 'projects.sort'
const REVERSE_STORE = 'projects.reverse'

function storedSort(): SortKey {
  const v = localStorage.getItem(SORT_STORE)
  return v && v in SORTS ? (v as SortKey) : 'activity'
}

/** "3 days ago" — the useful precision for a last-touched column.
 *
 *  An absolute date is available on hover; at a glance what matters is which
 *  projects are warm, and a column of identical-looking dates doesn't show
 *  that. Thresholds are coarse on purpose — minute-level precision would imply
 *  a resolution the underlying timestamps (second-granularity) don't warrant. */
function relativeTime(iso: string | null): string {
  if (!iso) return '—'
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'just now'

  // Each entry is "divide the CURRENT value by this to reach that unit" —
  // seconds/60 = minutes, minutes/60 = hours, hours/24 = days, and so on. The
  // divisor belongs to the step INTO the unit beside it, which is easy to get
  // off by one: a table reading [60,'minute'], [24,'hour'] divides minutes by
  // 24 instead of 60 and reports six hours ago as "2 days ago".
  const steps: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, 'minute'],
    [60, 'hour'],
    [24, 'day'],
    [7, 'week'],
    [4.35, 'month'],
    [12, 'year'],
  ]
  let value = seconds
  let unit: Intl.RelativeTimeFormatUnit = 'second'
  for (const [divisor, next] of steps) {
    // Too small to be worth the coarser unit — stay where we are.
    if (Math.abs(value) < divisor) break
    value = Math.round(value / divisor)
    unit = next
  }
  return new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(-value, unit)
}

/** Projects list — the app's home. */
export function Projects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>(storedSort)
  const [reverse, setReverse] = useState(() => localStorage.getItem(REVERSE_STORE) === '1')

  useEffect(() => localStorage.setItem(SORT_STORE, sort), [sort])
  useEffect(() => localStorage.setItem(REVERSE_STORE, reverse ? '1' : '0'), [reverse])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null)
  const [busy, setBusy] = useState(false)

  // A Set, not an array: selection is membership, and Set gives O(1) `has` for
  // the checkbox on every row instead of a scan per render.
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [confirmBulk, setConfirmBulk] = useState(false)

  const selectedProjects = useMemo(
    () => projects.filter((p) => selected.has(p.id)),
    [projects, selected],
  )

  /** The rows actually on screen: searched, then ordered.
   *
   *  Both are done here rather than server-side — a local tool has tens of
   *  projects, so this is instant and a search that round-trips per keystroke
   *  would only add latency. */
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    const matched = q
      ? projects.filter(
          (p) =>
            p.name.toLowerCase().includes(q) ||
            (p.description?.toLowerCase().includes(q) ?? false),
        )
      : projects

    const time = (s: string | null) => (s ? new Date(s).getTime() : 0)
    const compare: Record<SortKey, (a: Project, b: Project) => number> = {
      // Dates default to newest-first, which is what "sort by date" means to
      // most people; the reverse toggle covers the other direction.
      activity: (a, b) => time(b.last_activity_at) - time(a.last_activity_at),
      created: (a, b) => time(b.created_at) - time(a.created_at),
      // localeCompare so "Ångström" and "apple" sort sensibly, and numeric so
      // "run 2" precedes "run 10" instead of following it.
      name: (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }),
    }

    // Sort a COPY: Array.sort mutates, and mutating state in a useMemo makes
    // React's rendering depend on how many times the memo happened to run.
    const rows = [...matched].sort((a, b) => {
      const r = compare[sort](a, b)
      // Ties broken by id, always. Without a total order, equal-comparing rows
      // can swap places between renders and the list looks like it reshuffles
      // on its own — the same reason the server orders by (created_at, id).
      return r !== 0 ? r : b.id - a.id
    })
    return reverse ? rows.reverse() : rows
  }, [projects, query, sort, reverse])

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // Scoped to what's on screen: with a filter active, "select all" must mean
  // the rows you can see, not the ones the search is hiding.
  const allSelected = visible.length > 0 && visible.every((p) => selected.has(p.id))

  async function handleBulkDelete() {
    setBusy(true)
    try {
      await bulkDeleteProjects([...selected])
      setSelected(new Set())
      setConfirmBulk(false)
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // useCallback so this identity is stable and can be both used in the effect
  // and handed to children as an explicit "reload now" without re-firing the
  // effect on every render.
  const refresh = useCallback(async () => {
    try {
      setProjects(await listProjects())
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function handleDelete() {
    if (!pendingDelete) return
    setBusy(true)
    try {
      await deleteProject(pendingDelete.id)
      setPendingDelete(null)
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Projects"
        description="Local computer vision projects on this machine"
        actions={
          <>
            {/* Only appears with a selection. A permanent "Delete selected (0)"
                is dead furniture, and a red button that does nothing most of
                the time teaches people to ignore red. */}
            {selected.size > 0 && (
              <button className="btn-reject" onClick={() => setConfirmBulk(true)}>
                <Trash2 size={14} />
                Delete {selected.size} project{selected.size === 1 ? '' : 's'}
              </button>
            )}
            <button className="btn-primary" onClick={() => setShowCreate(true)}>
              <Plus size={14} />
              New project
            </button>
          </>
        }
      />
      <PageBody>
        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            {error}
          </div>
        )}

        {/* Search + ordering. Hidden entirely with nothing to search — controls
            for an empty list are noise, and the empty state below says more. */}
        {!loading && projects.length > 0 && (
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="relative min-w-56 flex-1 sm:max-w-xs">
              <Search
                size={13}
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400"
              />
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search projects…"
                aria-label="Search projects by name or description"
                className="w-full rounded-md border border-gray-300 bg-white py-1.5 pl-8 pr-8 text-sm placeholder:text-gray-400 focus:border-accent-500 focus:outline-none"
              />
              {query && (
                <button
                  onClick={() => setQuery('')}
                  aria-label="Clear search"
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                >
                  <X size={12} />
                </button>
              )}
            </div>

            <label htmlFor="sort" className="ml-auto text-xs text-gray-500">
              Sort by
            </label>
            <select
              id="sort"
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm focus:border-accent-500 focus:outline-none"
            >
              {Object.entries(SORTS).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
            <button
              onClick={() => setReverse((r) => !r)}
              aria-pressed={reverse}
              title={reverse ? 'Reversed order' : 'Reverse order'}
              className={`flex items-center gap-1 rounded-md border px-2 py-1.5 text-xs transition-colors ${
                reverse
                  ? 'border-accent-300 bg-accent-50 text-accent-700'
                  : 'border-gray-300 bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {/* 11px glyph beside 11px text, centred on the same baseline. */}
              <ArrowDownUp size={11} />
              Reverse
            </button>
          </div>
        )}

        {loading ? (
          <p className="text-sm text-gray-500">Loading…</p>
        ) : projects.length > 0 && visible.length === 0 ? (
          // A search that matches nothing is a different situation from having
          // no projects, and offering "New project" here would answer a
          // question the user didn't ask.
          <div className="card max-w-2xl border-dashed">
            <div className="px-4 py-8">
              <h3 className="text-sm font-medium text-gray-900">No matching projects</h3>
              <p className="mt-1 text-xs text-gray-500">
                Nothing matches “{query}”.{' '}
                <button
                  onClick={() => setQuery('')}
                  className="text-accent-700 underline hover:text-accent-800"
                >
                  Clear the search
                </button>{' '}
                to see all {projects.length}.
              </p>
            </div>
          </div>
        ) : projects.length === 0 ? (
          <div className="card max-w-2xl border-dashed">
            <div className="px-4 py-8">
              <h3 className="text-sm font-medium text-gray-900">No projects yet</h3>
              <p className="mt-1 max-w-md text-xs text-gray-500">
                Create a project to upload images, define classes, and start annotating.
              </p>
              <button className="btn-primary mt-4" onClick={() => setShowCreate(true)}>
                <Plus size={14} />
                New project
              </button>
            </div>
          </div>
        ) : (
          // A table, not a grid of cards. Projects have several comparable
          // scalar attributes (counts, dates) and a table aligns them into
          // scannable columns — cards would waste space and make comparison
          // harder for no gain.
          <div className="card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50 text-left">
                  <th className="w-10 px-4 py-2">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      // Indeterminate can't be set via an attribute — it's a DOM
                      // property only. The ref callback is the standard way, and
                      // without it a partial selection shows an empty box that
                      // reads as "nothing selected".
                      ref={(el) => {
                        if (el)
                          el.indeterminate = selected.size > 0 && !allSelected
                      }}
                      onChange={() =>
                        setSelected(
                          allSelected ? new Set() : new Set(visible.map((p) => p.id)),
                        )
                      }
                      className="accent-accent-600"
                      aria-label="Select all projects"
                    />
                  </th>
                  <th className="px-4 py-2 font-medium text-gray-600">Name</th>
                  <th className="px-4 py-2 font-medium text-gray-600">Type</th>
                  <th className="px-4 py-2 text-right font-medium text-gray-600">Images</th>
                  <th className="px-4 py-2 text-right font-medium text-gray-600">Classes</th>
                  <th className="px-4 py-2 font-medium text-gray-600">Created</th>
                  <th className="px-4 py-2 font-medium text-gray-600">Last modified</th>
                  <th className="w-10 px-4 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {visible.map((p) => (
                  <tr
                    key={p.id}
                    className={`group ${
                      selected.has(p.id) ? 'bg-accent-50' : 'hover:bg-gray-50'
                    }`}
                  >
                    <td className="px-4 py-2">
                      <input
                        type="checkbox"
                        checked={selected.has(p.id)}
                        onChange={() => toggle(p.id)}
                        className="accent-accent-600"
                        aria-label={`Select ${p.name}`}
                      />
                    </td>
                    <td className="px-4 py-2">
                      <Link
                        to={`/projects/${p.id}`}
                        className="font-medium text-gray-900 hover:text-accent-700 hover:underline"
                      >
                        {p.name}
                      </Link>
                      {p.description && (
                        <p className="truncate text-xs text-gray-500">{p.description}</p>
                      )}
                    </td>
                    <td className="px-4 py-2 text-gray-600">
                      {p.task_type.replace(/_/g, ' ')}
                    </td>
                    {/* tabular-nums keeps digits the same width so the column
                        stays aligned regardless of value. */}
                    <td className="px-4 py-2 text-right tabular-nums text-gray-900">
                      {p.image_count}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-gray-900">
                      {p.class_count}
                    </td>
                    <td className="px-4 py-2 text-gray-500">
                      {new Date(p.created_at).toLocaleDateString()}
                    </td>
                    <td
                      className="px-4 py-2 text-gray-500"
                      // The exact time on hover: the relative form is easier to
                      // scan, but "3 days ago" is the wrong thing to squint at
                      // when you actually need to know which run came first.
                      title={
                        p.last_activity_at
                          ? new Date(p.last_activity_at).toLocaleString()
                          : undefined
                      }
                    >
                      {relativeTime(p.last_activity_at)}
                    </td>
                    <td className="px-4 py-2">
                      <button
                        // Revealed on row hover to keep the table calm, but
                        // focus:opacity-100 keeps it reachable by keyboard —
                        // hover-only affordances are invisible to Tab users.
                        onClick={() => setPendingDelete(p)}
                        className="rounded p-1 text-gray-400 opacity-0 transition-opacity hover:bg-red-50 hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
                        aria-label={`Delete ${p.name}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </PageBody>

      <CreateProjectModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={refresh}
      />

      <ConfirmDialog
        open={pendingDelete !== null}
        onClose={() => setPendingDelete(null)}
        onConfirm={handleDelete}
        busy={busy}
        title="Delete project"
        message={
          pendingDelete
            ? `Delete "${pendingDelete.name}"? Its ${pendingDelete.image_count} image(s) and ${pendingDelete.class_count} class(es) will be permanently removed from disk. This cannot be undone.`
            : ''
        }
      />

      {/* Totals the real cost across the selection. "Delete 6 projects?" is a
          question you can't answer; "6 projects, 1,240 images" is. */}
      <ConfirmDialog
        open={confirmBulk}
        onClose={() => setConfirmBulk(false)}
        onConfirm={handleBulkDelete}
        busy={busy}
        title={`Delete ${selected.size} project${selected.size === 1 ? '' : 's'}`}
        confirmLabel={`Delete ${selected.size}`}
        message={
          `Permanently delete ${selected.size} project${selected.size === 1 ? '' : 's'} — ` +
          `${selectedProjects.reduce((n, p) => n + p.image_count, 0)} image(s) and ` +
          `${selectedProjects.reduce((n, p) => n + p.class_count, 0)} class(es) — ` +
          `and remove their files from disk? This cannot be undone.\n\n` +
          selectedProjects
            .slice(0, 8)
            .map((p) => `• ${p.name}`)
            .join('\n') +
          (selectedProjects.length > 8 ? `\n…and ${selectedProjects.length - 8} more` : '')
        }
      />
    </>
  )
}

function CreateProjectModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean
  onClose: () => void
  onCreated: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await createProject({ name, description: description || undefined })
      // Reset before closing, so reopening the modal starts clean rather than
      // showing the previous submission's text.
      setName('')
      setDescription('')
      onCreated()
      onClose()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="New project">
      {/* A real <form> with onSubmit, not a button with onClick: this gets
          Enter-to-submit and native required-field validation for free. */}
      <form onSubmit={submit} className="space-y-3">
        <div>
          <label htmlFor="p-name" className="mb-1 block text-xs font-medium text-gray-700">
            Name
          </label>
          <input
            id="p-name"
            autoFocus
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Traffic cameras"
            className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm placeholder:text-gray-400 focus:border-accent-500 focus:outline-none"
          />
        </div>

        <div>
          <label htmlFor="p-desc" className="mb-1 block text-xs font-medium text-gray-700">
            Description <span className="font-normal text-gray-400">(optional)</span>
          </label>
          <textarea
            id="p-desc"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full resize-none rounded-md border border-gray-300 px-2.5 py-1.5 text-sm placeholder:text-gray-400 focus:border-accent-500 focus:outline-none"
          />
        </div>

        <div>
          <span className="mb-1 block text-xs font-medium text-gray-700">Task type</span>
          {/* Disabled rather than hidden: it shows the axis along which this
              tool will grow, and sets the expectation that segmentation is
              coming rather than implying detection is all there is. */}
          <select
            disabled
            className="w-full cursor-not-allowed rounded-md border border-gray-300 bg-gray-50 px-2.5 py-1.5 text-sm text-gray-500"
          >
            <option>Object detection</option>
          </select>
          <p className="mt-1 text-xs text-gray-400">Segmentation arrives in a later phase.</p>
        </div>

        {error && (
          <p className="rounded-md border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs text-red-800">
            {error}
          </p>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button type="button" className="btn-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn-primary" disabled={busy || !name.trim()}>
            {busy ? 'Creating…' : 'Create project'}
          </button>
        </div>
      </form>
    </Modal>
  )
}
