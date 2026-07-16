import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Trash2 } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { ConfirmDialog, Modal } from '@/components/ui/Modal'
import {
  createProject,
  deleteProject,
  listProjects,
  type Project,
} from '@/lib/api'

/** Projects list — the app's home. */
export function Projects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null)
  const [busy, setBusy] = useState(false)

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
          <button className="btn-primary" onClick={() => setShowCreate(true)}>
            <Plus size={14} />
            New project
          </button>
        }
      />
      <PageBody>
        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            {error}
          </div>
        )}

        {loading ? (
          <p className="text-sm text-gray-500">Loading…</p>
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
                  <th className="px-4 py-2 font-medium text-gray-600">Name</th>
                  <th className="px-4 py-2 font-medium text-gray-600">Type</th>
                  <th className="px-4 py-2 text-right font-medium text-gray-600">Images</th>
                  <th className="px-4 py-2 text-right font-medium text-gray-600">Classes</th>
                  <th className="px-4 py-2 font-medium text-gray-600">Created</th>
                  <th className="w-10 px-4 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {projects.map((p) => (
                  <tr key={p.id} className="group hover:bg-gray-50">
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
