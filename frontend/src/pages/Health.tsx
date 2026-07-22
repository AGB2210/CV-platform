import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, TriangleAlert } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { getDatasetHealth, type DatasetHealth } from '@/lib/api'

/**
 * Dataset health — the answer to "why is my mAP low?".
 *
 * Disappointing training results are usually the DATA, not the model: one
 * class with 40x the boxes of another, half the boxes too small to learn at
 * training resolution, a class defined but never labelled. Counts alone hide
 * all of that, so this page draws the distributions and turns the
 * pathological ones into named, actionable warnings.
 *
 * Charts are hand-rolled SVG like MetricsChart — same reasoning: no dependency
 * for two bar charts, and the styling stays on the design tokens.
 */
export function Health() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [health, setHealth] = useState<DatasetHealth | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getDatasetHealth(projectId)
      .then((h) => !cancelled && setHealth(h))
      .catch((e: Error) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [projectId])

  return (
    <>
      <PageHeader
        title="Dataset health"
        description="Class balance and box sizes — where weak training results usually start"
        actions={
          <Link to={`/projects/${projectId}`} className="btn-secondary">
            <ArrowLeft size={14} />
            Dataset
          </Link>
        }
      />
      <PageBody>
        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            {error}
          </div>
        )}
        {!health ? (
          <p className="text-sm text-gray-500">Loading…</p>
        ) : health.total_boxes === 0 ? (
          <div className="card p-6 text-sm text-gray-600">
            No accepted boxes yet — health is a property of labelled data.{' '}
            <Link to={`/projects/${projectId}/annotate`} className="text-accent-700 underline">
              Auto-annotate
            </Link>{' '}
            or{' '}
            <Link to={`/projects/${projectId}/review`} className="text-accent-700 underline">
              draw some boxes
            </Link>{' '}
            first.
          </div>
        ) : (
          <div className="space-y-4">
            {/* Warnings first: they're the point of the page. Each one names
                the problem AND what to do about it. */}
            {health.warnings.map((w) => (
              <div
                key={w}
                className="flex gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
              >
                <TriangleAlert size={16} className="mt-0.5 shrink-0" />
                {w}
              </div>
            ))}
            {health.warnings.length === 0 && (
              <div className="rounded-md border border-status-good/30 bg-status-good/5 px-3 py-2 text-sm text-gray-700">
                No structural problems found — classes are balanced, box sizes are
                learnable, and every class has examples.
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <ClassBalance health={health} />
              <BoxSizes health={health} />
            </div>
          </div>
        )}
      </PageBody>
    </>
  )
}

/** Horizontal bars, one per class — boxes labelled, image coverage beside. */
function ClassBalance({ health }: { health: DatasetHealth }) {
  const max = Math.max(1, ...health.classes.map((c) => c.boxes))
  return (
    <div className="card">
      <div className="border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-medium text-gray-900">Class balance</h2>
        <p className="text-xs text-gray-500">
          {health.total_boxes} boxes across {health.classes.length} classes ·{' '}
          {health.annotated_images}/{health.total_images} images annotated
        </p>
      </div>
      <ul className="space-y-2 p-4">
        {[...health.classes]
          .sort((a, b) => b.boxes - a.boxes)
          .map((c) => (
            <li key={c.id}>
              <div className="mb-0.5 flex items-baseline justify-between text-xs">
                <span className="flex items-center gap-1.5 font-medium text-gray-800">
                  <span
                    className="h-2.5 w-2.5 rounded-sm border border-black/10"
                    style={{ backgroundColor: c.color }}
                  />
                  {c.name}
                </span>
                <span className="font-mono tabular-nums text-gray-500">
                  {c.boxes} box{c.boxes === 1 ? '' : 'es'} · {c.images} img
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded bg-gray-100">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${Math.max(c.boxes > 0 ? 2 : 0, (c.boxes / max) * 100)}%`,
                    backgroundColor: c.color,
                  }}
                />
              </div>
            </li>
          ))}
      </ul>
    </div>
  )
}

/** Relative-size histogram + the COCO small/medium/large buckets as chips. */
function BoxSizes({ health }: { health: DatasetHealth }) {
  const { relative_hist: hist, small, medium, large, tiny } = health.box_sizes
  const max = Math.max(1, ...hist)
  const W = 320
  const H = 120
  const barW = W / hist.length

  return (
    <div className="card">
      <div className="border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-medium text-gray-900">Box sizes</h2>
        <p className="text-xs text-gray-500">
          How much of its image each box spans (fraction of image width)
        </p>
      </div>
      <div className="space-y-3 p-4">
        <svg viewBox={`0 0 ${W} ${H + 18}`} className="w-full">
          {hist.map((n, i) => {
            const h = (n / max) * H
            return (
              <g key={i}>
                <rect
                  x={i * barW + 2}
                  y={H - h}
                  width={barW - 4}
                  height={h}
                  rx={2}
                  // The under-3% bin is the trouble bin — colour it as such.
                  fill={i === 0 ? '#d97706' : 'var(--color-accent-600)'}
                  opacity={n === 0 ? 0.15 : 0.9}
                />
                {n > 0 && (
                  <text
                    x={i * barW + barW / 2}
                    y={H - h - 3}
                    textAnchor="middle"
                    className="fill-gray-500 text-[8px] tabular-nums"
                  >
                    {n}
                  </text>
                )}
                <text
                  x={i * barW + barW / 2}
                  y={H + 12}
                  textAnchor="middle"
                  className="fill-gray-400 text-[8px] tabular-nums"
                >
                  {i * 10}–{(i + 1) * 10}%
                </text>
              </g>
            )
          })}
        </svg>
        <div className="flex flex-wrap gap-2 text-xs">
          <SizeChip label="COCO small (<32²px)" value={small} />
          <SizeChip label="medium (32²–96²px)" value={medium} />
          <SizeChip label="large (>96²px)" value={large} />
          <SizeChip label="tiny (<3% of image)" value={tiny} warn={tiny > 0} />
        </div>
      </div>
    </div>
  )
}

function SizeChip({ label, value, warn }: { label: string; value: number; warn?: boolean }) {
  return (
    <span
      className={`rounded border px-2 py-0.5 tabular-nums ${
        warn
          ? 'border-amber-300 bg-amber-50 text-amber-900'
          : 'border-gray-200 bg-gray-50 text-gray-700'
      }`}
    >
      {label}: <span className="font-medium">{value}</span>
    </span>
  )
}
