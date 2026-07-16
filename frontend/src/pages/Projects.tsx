import { useEffect, useState } from 'react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { StatusBadge, type Status } from '@/components/StatusBadge'
import { health, type HealthResponse } from '@/lib/api'

/**
 * Phase 0 landing page.
 *
 * There are no projects yet (that's Phase 1), so this page does the one thing
 * worth proving at this stage: that the React app can reach the FastAPI backend
 * and that the backend can reach SQLite. It's a real end-to-end trace through
 * every layer we just built — browser -> Vite proxy -> FastAPI -> SQLAlchemy ->
 * SQLite and back.
 *
 * Phase 1 replaces the body with the actual project list.
 */
export function Projects() {
  // Three-state async: null = still loading, Error = failed, value = loaded.
  // Tracking loading/error/data explicitly avoids the classic bug where an
  // empty list and a failed request render identically.
  const [data, setData] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    health()
      .then((res) => {
        if (!cancelled) setData(res)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })

    // Cleanup guards against setting state after unmount — React 18+ StrictMode
    // intentionally double-invokes effects in dev to surface exactly this bug.
    return () => {
      cancelled = true
    }
  }, [])

  const apiStatus: Status = error ? 'failed' : data ? 'done' : 'queued'
  const dbStatus: Status = error
    ? 'unknown'
    : data
      ? data.database === 'ok'
        ? 'done'
        : 'failed'
      : 'queued'

  return (
    <>
      <PageHeader
        title="Projects"
        description="Local computer vision projects on this machine"
        actions={
          <button className="btn-primary" disabled title="Available in Phase 1">
            New project
          </button>
        }
      />
      <PageBody>
        {/* System status — the Phase 0 deliverable. */}
        <section className="card mb-6 max-w-2xl">
          <div className="border-b border-gray-200 px-4 py-3">
            <h2 className="text-sm font-medium text-gray-900">System status</h2>
          </div>

          <dl className="divide-y divide-gray-100">
            <StatusRow label="API server" hint="FastAPI on :8000">
              <StatusBadge
                status={apiStatus}
                label={error ? 'Unreachable' : data ? 'Connected' : 'Checking…'}
              />
            </StatusRow>

            <StatusRow label="Database" hint="SQLite via SQLAlchemy">
              <StatusBadge
                status={dbStatus}
                label={
                  error
                    ? 'Unknown'
                    : data
                      ? data.database === 'ok'
                        ? 'Connected'
                        : data.database
                      : 'Checking…'
                }
              />
            </StatusRow>

            <StatusRow label="Storage" hint="Images, weights, runs">
              <span className="max-w-xs truncate font-mono text-xs text-gray-600">
                {data?.storage_dir ?? '—'}
              </span>
            </StatusRow>
          </dl>

          {error && (
            <div className="border-t border-gray-200 bg-red-50 px-4 py-3">
              <p className="text-xs text-red-800">
                <span className="font-medium">Cannot reach the API.</span> Start it with{' '}
                <code className="rounded bg-red-100 px-1 py-0.5 font-mono">
                  uvicorn app.main:app --reload --port 8000
                </code>{' '}
                from the <code className="font-mono">backend/</code> directory.
              </p>
            </div>
          )}
        </section>

        {/* Empty state. Left-aligned and plain — a centered illustration would
            be the "marketing page" reflex this tool should avoid. */}
        <section className="card max-w-2xl border-dashed">
          <div className="px-4 py-8">
            <h3 className="text-sm font-medium text-gray-900">No projects yet</h3>
            <p className="mt-1 max-w-md text-xs text-gray-500">
              Project creation, image upload, and class management arrive in Phase 1. The
              scaffolding is in place — backend, database, and frontend are wired together
              and talking.
            </p>
          </div>
        </section>
      </PageBody>
    </>
  )
}

/** One label/value row inside a definition-list card. */
function StatusRow({
  label,
  hint,
  children,
}: {
  label: string
  hint: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between px-4 py-2.5">
      <div>
        <dt className="text-sm text-gray-900">{label}</dt>
        <dd className="text-xs text-gray-500">{hint}</dd>
      </div>
      <div>{children}</div>
    </div>
  )
}
