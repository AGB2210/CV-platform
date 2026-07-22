import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Gauge, Play, TriangleAlert } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { MlSetupGate } from '@/components/MlSetupGate'
import { StatusBadge, type Status } from '@/components/StatusBadge'
import {
  getEvaluation,
  listDatasetVersions,
  listEvaluations,
  listModels,
  startEvaluation,
  versionLabel,
  type DatasetVersion,
  type DeployableModel,
  type EvaluationDetails,
  type EvaluationJob,
} from '@/lib/api'

/**
 * Evaluate — score a trained model on the TEST split and see its test mAP.
 *
 * The test split is the one split training never uses: it trains on train,
 * watches val to choose the checkpoint, and leaves test untouched. So a test
 * mAP is the first genuinely independent measure of the model, and per-class AP
 * shows which class is dragging it down. If the chosen dataset version has no
 * test images, there is nothing honest to measure against, so the page says so
 * and points at where to add a test set.
 */
export function Evaluate() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [models, setModels] = useState<DeployableModel[] | null>(null)
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [evals, setEvals] = useState<EvaluationJob[]>([])
  const [modelId, setModelId] = useState<number | null>(null)
  const [versionId, setVersionId] = useState<number | null>(null)
  const [active, setActive] = useState<EvaluationJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    Promise.all([listModels(projectId), listDatasetVersions(projectId), listEvaluations(projectId)])
      .then(([m, v, e]) => {
        setModels(m)
        setVersions(v)
        setEvals(e)
        if (m.length) setModelId(m[0].job_id)
        // Default to the version with the most test images, so the honest path
        // is the pre-selected one.
        const best = [...v].sort((a, b) => b.test_images - a.test_images)[0]
        if (best) setVersionId(best.id)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [projectId])

  const version = useMemo(
    () => versions.find((v) => v.id === versionId) ?? null,
    [versions, versionId],
  )
  const testImages = version?.test_images ?? 0

  // Poll a running evaluation until it settles, then refresh history.
  useEffect(() => {
    if (!active || (active.status !== 'running' && active.status !== 'queued')) {
      if (pollRef.current) window.clearInterval(pollRef.current)
      pollRef.current = null
      return
    }
    pollRef.current = window.setInterval(async () => {
      try {
        const fresh = await getEvaluation(active.id)
        setActive(fresh)
        if (fresh.status === 'done' || fresh.status === 'failed') {
          setEvals(await listEvaluations(projectId))
        }
      } catch {
        /* transient; the next tick retries */
      }
    }, 2000)
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [active, projectId])

  // Between click and the job row existing — greys the button so the click
  // visibly registered and a double-click can't queue two evaluations.
  const [starting, setStarting] = useState(false)

  const run = async () => {
    if (modelId == null || versionId == null || starting) return
    setError(null)
    setStarting(true)
    try {
      setActive(await startEvaluation(projectId, {
        training_job_id: modelId,
        dataset_version_id: versionId,
        split: 'test',
      }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  const running = starting || active?.status === 'running' || active?.status === 'queued'

  return (
    <>
      <PageHeader
        title="Evaluate"
        description="Score a trained model on the held-out test split"
        actions={
          <Link to={`/projects/${projectId}`} className="btn-secondary">
            <ArrowLeft size={14} />
            Dataset
          </Link>
        }
      />
      <PageBody>
        <MlSetupGate feature="Evaluate">
          {error && (
            <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              {error}
            </div>
          )}

          {models !== null && models.length === 0 ? (
            <NoModelsCard projectId={projectId} />
          ) : (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
              {/* --- Left: choose what to score --- */}
              <div className="card space-y-4 p-4">
                <div>
                  <label className="label-eyebrow mb-1 block">Model</label>
                  <select
                    value={modelId ?? ''}
                    onChange={(e) => setModelId(Number(e.target.value))}
                    className="w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm"
                  >
                    {(models ?? []).map((m) => (
                      <option key={m.job_id} value={m.job_id}>
                        {m.label}
                        {m.best_map != null ? ` · val mAP ${m.best_map.toFixed(3)}` : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="label-eyebrow mb-1 block">Dataset version</label>
                  <select
                    value={versionId ?? ''}
                    onChange={(e) => setVersionId(Number(e.target.value))}
                    className="w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm"
                  >
                    {versions.map((v) => (
                      <option key={v.id} value={v.id}>
                        {versionLabel(v)} · {v.test_images} test
                      </option>
                    ))}
                  </select>
                </div>

                {testImages === 0 ? (
                  <NoTestData projectId={projectId} />
                ) : (
                  <>
                    <p className="text-xs text-gray-500">
                      Scoring on <span className="font-medium">{testImages}</span> held-out
                      test image{testImages === 1 ? '' : 's'} — data the model never trained
                      on.
                    </p>
                    <button
                      type="button"
                      onClick={run}
                      disabled={running || modelId == null}
                      className="btn-primary inline-flex w-full items-center justify-center gap-1.5"
                    >
                      <Play size={14} />
                      {running ? 'Evaluating…' : 'Evaluate on test split'}
                    </button>
                  </>
                )}
              </div>

              {/* --- Right: result --- */}
              <div className="space-y-4">
                {active && <ResultCard job={active} />}
                {evals.length > 0 && (
                  <HistoryCard
                    evals={evals}
                    versions={versions}
                    onSelect={setActive}
                    activeId={active?.id ?? null}
                  />
                )}
                {!active && evals.length === 0 && (
                  <div className="card flex min-h-40 items-center justify-center p-6 text-sm text-gray-400">
                    Choose a model and version, then evaluate to see the test mAP.
                  </div>
                )}
              </div>
            </div>
          )}
        </MlSetupGate>
      </PageBody>
    </>
  )
}

function ResultCard({ job }: { job: EvaluationJob }) {
  const status = job.status as Status
  return (
    <div className="card">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <div className="flex items-center gap-2">
          <Gauge size={14} className="text-gray-400" />
          <h2 className="text-sm font-medium text-gray-900">Test evaluation</h2>
        </div>
        <StatusBadge status={status} />
      </div>

      {job.status === 'failed' ? (
        <div className="flex items-start gap-2 p-4 text-xs text-red-800">
          <TriangleAlert size={14} className="mt-0.5 shrink-0" />
          <span>{job.error}</span>
        </div>
      ) : job.status !== 'done' ? (
        <p className="p-4 text-sm text-gray-500">
          Running the model over {job.num_images || '…'} test image
          {job.num_images === 1 ? '' : 's'}…
        </p>
      ) : (
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Test mAP@50-95" value={fmt(job.map_50_95)} accent />
            <Metric label="mAP@50" value={fmt(job.map_50)} />
            <Metric label="mAP@75" value={fmt(job.map_75)} />
          </div>
          <div>
            <p className="label-eyebrow mb-1.5">Per-class AP@50-95</p>
            <div className="space-y-1.5">
              {job.per_class.map((c) => (
                <div key={c.name} className="flex items-center gap-2 text-xs">
                  <span className="w-20 shrink-0 truncate text-gray-600">{c.name}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
                    <div
                      className="h-full rounded-full bg-accent-500"
                      style={{ width: `${Math.max(0, (c.ap ?? 0) * 100)}%` }}
                    />
                  </div>
                  <span className="w-12 shrink-0 text-right font-mono tabular-nums text-gray-900">
                    {fmt(c.ap)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <p className="text-xs text-gray-400">
            {job.num_images} test image{job.num_images === 1 ? '' : 's'}. mAP@50-95 is the
            COCO headline metric — the average over IoU thresholds 0.50 to 0.95.
          </p>

          {/* The diagnostics the headline hides: what got confused with what,
              how precision trades against recall, and WHICH images fail. */}
          {job.details && (
            <>
              <PRCurves curves={job.details.pr_curves} />
              <ConfusionMatrix confusion={job.details.confusion} />
              <WorstImages worst={job.details.worst} projectId={job.project_id} />
            </>
          )}
        </div>
      )}
    </div>
  )
}

/** One line per class in a fixed palette (class colors live on the project;
 *  a deterministic assignment here keeps the chart self-contained). */
const PR_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d']

function PRCurves({ curves }: { curves: EvaluationDetails['pr_curves'] }) {
  if (!curves.length) return null
  const W = 320
  const H = 180
  const PAD = { l: 30, r: 8, t: 8, b: 24 }
  const pw = W - PAD.l - PAD.r
  const ph = H - PAD.t - PAD.b
  const sx = (r: number) => PAD.l + r * pw
  const sy = (p: number) => PAD.t + (1 - p) * ph

  return (
    <div>
      <p className="label-eyebrow mb-1.5">Precision–recall at IoU 0.50</p>
      <div className="mb-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-gray-500">
        {curves.map((c, i) => (
          <span key={c.name} className="flex items-center gap-1">
            <span
              className="inline-block h-0.5 w-3 rounded"
              style={{ backgroundColor: PR_COLORS[i % PR_COLORS.length] }}
            />
            {c.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded border border-gray-100">
        {/* Axis frame + quarter gridlines. */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={PAD.t} x2={sx(t)} y2={H - PAD.b} stroke="#f4f4f5" />
            <line x1={PAD.l} y1={sy(t)} x2={W - PAD.r} y2={sy(t)} stroke="#f4f4f5" />
            <text x={sx(t)} y={H - PAD.b + 12} textAnchor="middle" className="fill-gray-400 text-[8px] tabular-nums">
              {t}
            </text>
            <text x={PAD.l - 4} y={sy(t)} dy="0.32em" textAnchor="end" className="fill-gray-400 text-[8px] tabular-nums">
              {t}
            </text>
          </g>
        ))}
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#d4d4d8" />
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#d4d4d8" />
        <text x={PAD.l + pw / 2} y={H - 2} textAnchor="middle" className="fill-gray-500 text-[8px]">
          recall
        </text>
        {curves.map((c, i) => (
          <path
            key={c.name}
            d={c.recall
              .map((r, j) => `${j === 0 ? 'M' : 'L'}${sx(r).toFixed(1)},${sy(c.precision[j]).toFixed(1)}`)
              .join(' ')}
            fill="none"
            stroke={PR_COLORS[i % PR_COLORS.length]}
            strokeWidth={1.5}
          />
        ))}
      </svg>
    </div>
  )
}

function ConfusionMatrix({ confusion }: { confusion: EvaluationDetails['confusion'] }) {
  const { classes, matrix } = confusion
  const max = Math.max(1, ...matrix.flat())
  return (
    <div>
      <p className="label-eyebrow mb-1.5">Confusion at conf 0.25 / IoU 0.45</p>
      <div className="overflow-x-auto">
        <table className="text-[10px] tabular-nums">
          <thead>
            <tr>
              {/* Rows = what the model SAID; columns = what was actually there. */}
              <th className="p-1 text-left font-normal text-gray-400">pred \ actual</th>
              {classes.map((c) => (
                <th key={c} className="max-w-16 truncate p-1 font-medium text-gray-600">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={classes[i]}>
                <td className="max-w-16 truncate p-1 pr-2 font-medium text-gray-600">
                  {classes[i]}
                </td>
                {row.map((v, j) => {
                  const diagonal = i === j && i < classes.length - 1
                  return (
                    <td
                      key={j}
                      className="h-8 w-12 border border-gray-100 text-center"
                      style={{
                        // Diagonal = correct, green scale; everything else is
                        // an error, red scale. Intensity by count.
                        backgroundColor:
                          v === 0
                            ? undefined
                            : diagonal
                              ? `rgba(22, 163, 74, ${0.15 + 0.6 * (v / max)})`
                              : `rgba(220, 38, 38, ${0.12 + 0.55 * (v / max)})`,
                        color: v / max > 0.5 ? 'white' : undefined,
                      }}
                    >
                      {v || ''}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-[10px] text-gray-400">
        "background" row = missed objects; "background" column = invented ones.
      </p>
    </div>
  )
}

function WorstImages({
  worst,
  projectId,
}: {
  worst: EvaluationDetails['worst']
  projectId: number
}) {
  if (!worst.length) {
    return (
      <p className="rounded border border-status-good/30 bg-status-good/5 px-2.5 py-1.5 text-xs text-gray-700">
        No test image had errors at the reviewing operating point (conf 0.25).
      </p>
    )
  }
  return (
    <div>
      <p className="label-eyebrow mb-1.5">Worst test images — the ones to look at</p>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
        {worst.map((w) => (
          <Link
            key={w.image_id}
            // Straight into the editor on that image, where the failure can
            // be seen (and often turns out to be a labelling error).
            to={`/projects/${projectId}/review/${w.image_id}`}
            className="group relative overflow-hidden rounded border border-gray-200 hover:border-accent-400"
            title={`${w.original_filename} — open in editor`}
          >
            <img
              src={`/api/thumbs/${projectId}/${w.filename}`}
              alt={w.original_filename}
              loading="lazy"
              className="aspect-square w-full object-cover"
            />
            <span className="absolute bottom-1 left-1 flex gap-1 text-[9px] font-medium">
              {w.fn > 0 && (
                <span className="rounded bg-amber-600/90 px-1 leading-4 text-white">
                  {w.fn} missed
                </span>
              )}
              {w.fp > 0 && (
                <span className="rounded bg-red-600/90 px-1 leading-4 text-white">
                  {w.fp} wrong
                </span>
              )}
            </span>
          </Link>
        ))}
      </div>
    </div>
  )
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded border border-gray-200 p-3">
      <p className="label-eyebrow">{label}</p>
      <p className={`mt-0.5 text-xl font-semibold tabular-nums ${accent ? 'text-accent-700' : 'text-gray-900'}`}>
        {value}
      </p>
    </div>
  )
}

function HistoryCard({
  evals,
  versions,
  onSelect,
  activeId,
}: {
  evals: EvaluationJob[]
  versions: DatasetVersion[]
  onSelect: (j: EvaluationJob) => void
  activeId: number | null
}) {
  const vlabel = (vid: number) => {
    const v = versions.find((x) => x.id === vid)
    return v ? versionLabel(v) : `v?`
  }
  return (
    <div className="card">
      <div className="border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-medium text-gray-900">History</h2>
      </div>
      <div className="divide-y divide-gray-100">
        {evals.map((e) => (
          <button
            key={e.id}
            type="button"
            onClick={() => onSelect(e)}
            className={`flex w-full items-center justify-between px-4 py-2 text-left text-xs hover:bg-gray-50 ${
              e.id === activeId ? 'bg-accent-50' : ''
            }`}
          >
            <span className="text-gray-600">
              model #{e.training_job_id} · {vlabel(e.dataset_version_id)} · {e.num_images} img
            </span>
            <span className="flex items-center gap-3">
              {e.status === 'done' && e.map_50_95 != null && (
                <span className="font-mono tabular-nums text-gray-900">mAP {fmt(e.map_50_95)}</span>
              )}
              <StatusBadge status={e.status as Status} />
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

function NoTestData({ projectId }: { projectId: number }) {
  return (
    <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2.5 text-xs text-amber-900">
      <p className="font-medium">This version has no test images.</p>
      <p className="mt-1">
        A test score is only honest against data the model never trained on. On the
        Dataset page, assign images to the test split (or upload a labelled test set),
        save a version, then evaluate that version.
      </p>
      <Link
        to={`/projects/${projectId}`}
        className="mt-2 inline-flex items-center gap-1 font-medium underline underline-offset-2"
      >
        Go to Dataset
      </Link>
    </div>
  )
}

function NoModelsCard({ projectId }: { projectId: number }) {
  return (
    <div className="card max-w-xl">
      <div className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <Gauge size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">No trained model yet</h2>
      </div>
      <div className="space-y-2 p-4 text-sm text-gray-600">
        <p>Evaluation scores a model you have trained. Train one first.</p>
        <Link
          to={`/projects/${projectId}/train`}
          className="btn-primary inline-flex items-center gap-1.5"
        >
          Go to Train
        </Link>
      </div>
    </div>
  )
}

function fmt(v: number | null | undefined): string {
  return v == null ? '—' : v.toFixed(3)
}
