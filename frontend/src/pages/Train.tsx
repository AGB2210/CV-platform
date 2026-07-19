import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Cpu, Database, Layers, Play } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { StatusBadge, type Status } from '@/components/StatusBadge'
import {
  getDevice,
  getTrainPreview,
  listTrainers,
  listTrainingJobs,
  getTrainingJob,
  startTraining,
  type DeviceInfo,
  type TrainerInfo,
  type TrainingJob,
  type TrainPreview,
} from '@/lib/api'

/** Poll cadence while a run is active. Training epochs take seconds to minutes,
 *  so 2s is plenty — faster would mostly return the same epoch. */
const POLL_MS = 2000

export function Train() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [trainers, setTrainers] = useState<TrainerInfo[]>([])
  const [device, setDevice] = useState<DeviceInfo | null>(null)
  const [preview, setPreview] = useState<TrainPreview | null>(null)
  const [jobs, setJobs] = useState<TrainingJob[]>([])

  const [trainerKey, setTrainerKey] = useState('')
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(8)
  const [imageSize, setImageSize] = useState(640)
  // Learning rate is a STRING so "empty" is expressible — empty means "use the
  // framework's own schedule", which is a real, distinct choice from any number.
  const [lr, setLr] = useState('')

  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Scroll the job card into view when a run STARTS — once per job, not per
  // poll. Same reasoning as the annotate page: the Start button is below the
  // fold, so a card appearing at the top is invisible at the moment it matters.
  const jobRef = useRef<HTMLDivElement>(null)
  const scrolledForJob = useRef<number | null>(null)
  useEffect(() => {
    if (!activeJob || scrolledForJob.current === activeJob.id) return
    scrolledForJob.current = activeJob.id
    jobRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [activeJob])

  const refresh = useCallback(async () => {
    const [p, j] = await Promise.all([
      getTrainPreview(projectId),
      listTrainingJobs(projectId),
    ])
    setPreview(p)
    setJobs(j)
  }, [projectId])

  // Initial load: everything the page needs, in parallel.
  useEffect(() => {
    let cancelled = false
    Promise.all([
      listTrainers(),
      getDevice(),
      getTrainPreview(projectId),
      listTrainingJobs(projectId),
    ])
      .then(([t, d, p, j]) => {
        if (cancelled) return
        setTrainers(t)
        setDevice(d)
        setPreview(p)
        setJobs(j)
        // Default to the first registered trainer and pre-fill the form with
        // ITS defaults — a sane batch for one backend isn't sane for another.
        if (t.length) {
          setTrainerKey((k) => k || t[0].key)
          setEpochs(t[0].default_epochs)
          setBatchSize(t[0].default_batch_size)
          setImageSize(t[0].default_image_size)
        }
        // Resume polling if a run is already in flight (e.g. page reloaded
        // mid-training) — exactly why job state lives in the DB, not memory.
        const running = j.find((x) => x.status === 'running' || x.status === 'queued')
        if (running) setActiveJob(running)
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [projectId])

  // When the trainer changes, adopt its defaults (but not mid-run).
  function selectTrainer(key: string) {
    setTrainerKey(key)
    const t = trainers.find((x) => x.key === key)
    if (t) {
      setEpochs(t.default_epochs)
      setBatchSize(t.default_batch_size)
      setImageSize(t.default_image_size)
    }
  }

  // --- Polling (identical shape to the annotate page) --------------------
  const pollRef = useRef<number | null>(null)
  useEffect(() => {
    const isActive = activeJob?.status === 'running' || activeJob?.status === 'queued'
    if (!isActive || !activeJob) {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
      return
    }
    pollRef.current = window.setInterval(async () => {
      try {
        const fresh = await getTrainingJob(activeJob.id)
        setActiveJob(fresh)
        if (fresh.status === 'done' || fresh.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          void refresh()
        }
      } catch (e) {
        setError((e as Error).message)
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
      }
    }, POLL_MS)
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [activeJob, refresh])

  async function run() {
    setError(null)
    try {
      const job = await startTraining(projectId, {
        trainer_key: trainerKey,
        epochs,
        batch_size: batchSize,
        image_size: imageSize,
        // Empty field => omit => backend uses the framework default.
        learning_rate: lr.trim() ? Number(lr) : null,
      })
      setActiveJob(job)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const selected = trainers.find((t) => t.key === trainerKey)
  const isRunning = activeJob?.status === 'running' || activeJob?.status === 'queued'
  const noTrainers = trainers.length === 0
  const canRun = !isRunning && !noTrainers && !!preview?.can_train

  if (loading) {
    return (
      <>
        <PageHeader title="Train" />
        <PageBody>
          <p className="text-sm text-gray-500">Loading…</p>
        </PageBody>
      </>
    )
  }

  return (
    <>
      <PageHeader
        title="Train"
        description="Fine-tune a detector on this project's accepted annotations"
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

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          {/* --- Left: configure + run --- */}
          <section className="space-y-4">
            {activeJob && (
              <div ref={jobRef}>
                <TrainProgress job={activeJob} />
              </div>
            )}

            {noTrainers ? (
              <NoTrainersCard />
            ) : (
              <div className="card">
                <div className="border-b border-gray-200 px-4 py-3">
                  <h2 className="text-sm font-medium text-gray-900">Model</h2>
                </div>
                <div className="space-y-3 p-4">
                  <div>
                    <label
                      htmlFor="trainer"
                      className="mb-1 block text-xs font-medium text-gray-700"
                    >
                      Training backend
                    </label>
                    <select
                      id="trainer"
                      value={trainerKey}
                      onChange={(e) => selectTrainer(e.target.value)}
                      disabled={isRunning}
                      className="w-full rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-sm focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                    >
                      {trainers.map((t) => (
                        <option key={t.key} value={t.key}>
                          {t.display_name} (~{t.approx_vram_gb} GB VRAM)
                        </option>
                      ))}
                    </select>
                    {selected && (
                      <p className="mt-1 text-xs text-gray-500">{selected.description}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <NumberField
                      label="Epochs"
                      value={epochs}
                      onChange={setEpochs}
                      min={1}
                      max={1000}
                      disabled={isRunning}
                    />
                    <NumberField
                      label="Batch size"
                      hint="lower if OOM"
                      value={batchSize}
                      onChange={setBatchSize}
                      min={1}
                      max={128}
                      disabled={isRunning}
                    />
                    <NumberField
                      label="Image size"
                      hint="px, square"
                      value={imageSize}
                      onChange={setImageSize}
                      min={64}
                      max={2048}
                      step={32}
                      disabled={isRunning}
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="lr"
                      className="mb-1 block text-xs font-medium text-gray-700"
                    >
                      Learning rate{' '}
                      <span className="font-normal text-gray-400">(optional)</span>
                    </label>
                    <input
                      id="lr"
                      type="number"
                      value={lr}
                      onChange={(e) => setLr(e.target.value)}
                      disabled={isRunning}
                      placeholder="framework default"
                      step="0.0001"
                      min="0"
                      className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm placeholder:text-gray-300 focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                    />
                    <p className="mt-0.5 text-xs text-gray-400">
                      Leave empty to use the backend's tuned schedule — usually the
                      right choice.
                    </p>
                  </div>

                  {/* 4 GB reality check, stated where the knobs are. */}
                  <p className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">
                    On a 4 GB GPU, keep batch and image size small. If a run fails
                    with an out-of-memory error, halve the batch size and retry.
                  </p>
                </div>
              </div>
            )}

            <div className="flex items-center gap-3">
              <button className="btn-primary" onClick={() => void run()} disabled={!canRun}>
                {isRunning ? (
                  <>Training…</>
                ) : (
                  <>
                    <Play size={14} />
                    Start training
                  </>
                )}
              </button>
              {/* Say WHY it's disabled — an inert button with no explanation is
                  the annotate page's lesson applied here. */}
              {!isRunning && noTrainers && (
                <span className="text-xs text-gray-500">No backend installed</span>
              )}
              {!isRunning && !noTrainers && preview && !preview.can_train && (
                <span className="text-xs text-gray-500">
                  {preview.num_classes === 0
                    ? 'Add a class first'
                    : 'No accepted boxes in the train split'}
                </span>
              )}
            </div>
          </section>

          {/* --- Right: status --- */}
          <aside className="space-y-4">
            {device && <DeviceCard device={device} />}
            {preview && <DatasetCard preview={preview} projectId={projectId} />}
            {jobs.length > 0 && <JobHistory jobs={jobs} />}
          </aside>
        </div>
      </PageBody>
    </>
  )
}

function toStatus(s: TrainingJob['status']): Status {
  return s === 'done' ? 'done' : s === 'failed' ? 'failed' : s === 'running' ? 'running' : 'queued'
}

function fmtMap(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(3)
}

function TrainProgress({ job }: { job: TrainingJob }) {
  const isRunning = job.status === 'running' || job.status === 'queued'
  return (
    <div
      className={`card transition-shadow ${
        isRunning ? 'ring-2 ring-accent-400 ring-offset-2' : ''
      }`}
    >
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">
          {isRunning ? 'Training…' : `Run #${job.id}`}
        </h2>
        <StatusBadge status={toStatus(job.status)} />
      </div>
      <div className="p-4">
        {/* Epoch progress bar, driven by current/total from the DB. */}
        <div className="mb-1.5 flex items-baseline justify-between text-xs">
          <span className="text-gray-600">
            Epoch {job.current_epoch} / {job.total_epochs}
          </span>
          <span className="font-mono tabular-nums text-gray-600">{job.progress_pct}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              job.status === 'failed' ? 'bg-status-bad' : 'bg-accent-600'
            }`}
            style={{ width: `${job.progress_pct}%` }}
          />
        </div>

        {/* The numbers that say whether it's actually learning. */}
        <div className="mt-3 grid grid-cols-3 gap-2">
          <Metric label="Train loss" value={job.train_loss === null ? '—' : job.train_loss?.toFixed(3)} />
          <Metric label="Val mAP" value={fmtMap(job.val_map)} />
          <Metric label="Best mAP" value={fmtMap(job.best_map)} accent />
        </div>

        {/* A mAP curve, so a plateau or a collapse is visible at a glance. Only
            once there are two points to draw a line between. */}
        {job.metrics.length > 1 && (
          <MapSparkline points={job.metrics} className="mt-3" />
        )}

        {job.status === 'done' && (
          <p className="mt-3 rounded-md border border-status-good/30 bg-status-good/5 px-2.5 py-1.5 text-xs text-gray-700">
            Best checkpoint saved. Evaluation and a prediction playground arrive in
            the next phase.
          </p>
        )}

        {job.error && (
          <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded border border-red-200 bg-red-50 p-2 text-[11px] text-red-900">
            {job.error}
          </pre>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value, accent }: { label: string; value: string | undefined; accent?: boolean }) {
  return (
    <div className="rounded-md border border-gray-200 bg-gray-50 px-2 py-1.5">
      <p className="text-[10px] uppercase tracking-wide text-gray-400">{label}</p>
      <p className={`font-mono text-sm tabular-nums ${accent ? 'font-semibold text-accent-700' : 'text-gray-900'}`}>
        {value ?? '—'}
      </p>
    </div>
  )
}

/** A tiny inline-SVG sparkline of val mAP over epochs. Self-contained — no chart
 *  dependency for a five-line trend line. Ignores epochs with no measurement. */
function MapSparkline({ points, className }: { points: TrainingJob['metrics']; className?: string }) {
  const data = points.filter((p) => p.val_map !== null) as { epoch: number; val_map: number }[]
  if (data.length < 2) return null

  const W = 280
  const H = 40
  const maxMap = Math.max(...data.map((d) => d.val_map), 0.01)
  const xs = (i: number) => (i / (data.length - 1)) * W
  const ys = (v: number) => H - (v / maxMap) * H
  const path = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(d.val_map).toFixed(1)}`).join(' ')

  return (
    <div className={className}>
      <p className="mb-1 text-[10px] uppercase tracking-wide text-gray-400">Val mAP over epochs</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" height={H}>
        <path d={path} fill="none" stroke="var(--color-accent-600)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  )
}

function NoTrainersCard() {
  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <Layers size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">No training backend installed</h2>
      </div>
      <div className="space-y-2 p-4 text-sm text-gray-600">
        <p>
          The training pipeline is ready, but no backend is registered yet. Training
          pulls in heavy dependencies (PyTorch is already here; the trainer adds its
          own), kept out of the base install so the app stays light.
        </p>
        <p className="text-xs text-gray-500">
          Install a backend into the backend venv, then restart the server — it will
          appear here automatically, like the annotation models do.
        </p>
        <pre className="overflow-auto rounded border border-gray-200 bg-gray-50 p-2 text-[11px] text-gray-700">
          pip install ultralytics
        </pre>
      </div>
    </div>
  )
}

function DeviceCard({ device }: { device: DeviceInfo }) {
  const onGpu = device.device === 'cuda'
  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2.5">
        <Cpu size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">Compute</h2>
      </div>
      <dl className="space-y-1.5 p-3 text-xs">
        <div className="flex justify-between gap-2">
          <dt className="text-gray-500">Device</dt>
          <dd className="truncate font-medium text-gray-900">{device.name}</dd>
        </div>
        {device.total_vram_gb && (
          <div className="flex justify-between">
            <dt className="text-gray-500">VRAM</dt>
            <dd className="font-mono tabular-nums text-gray-900">{device.total_vram_gb} GB</dd>
          </div>
        )}
        <div className="flex justify-between">
          <dt className="text-gray-500">Backend</dt>
          <dd className="font-mono text-gray-900">{device.device}</dd>
        </div>
      </dl>
      {/* Training on CPU is not minutes-slow like inference — it's hours-to-days
          slow. Warn even more firmly than the annotate page. */}
      {!onGpu && (
        <p className="border-t border-gray-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          No CUDA GPU detected. Training on CPU is impractically slow — expect hours
          per epoch.
        </p>
      )}
    </div>
  )
}

function DatasetCard({ preview, projectId }: { preview: TrainPreview; projectId: number }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2.5">
        <Database size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">Dataset</h2>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100 text-gray-400">
            <th className="px-3 py-1.5 text-left font-medium">Split</th>
            <th className="px-3 py-1.5 text-right font-medium">Images</th>
            <th className="px-3 py-1.5 text-right font-medium">Boxes</th>
          </tr>
        </thead>
        <tbody>
          {(['train', 'val', 'test'] as const).map((s) => (
            <tr key={s} className="border-b border-gray-50 last:border-0">
              <td className="px-3 py-1.5 text-gray-700 capitalize">{s}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums text-gray-900">
                {preview.splits[s].images}
              </td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums text-gray-900">
                {preview.splits[s].boxes}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {preview.warnings.map((w) => (
        <p key={w} className="border-t border-gray-100 bg-amber-50 px-3 py-1.5 text-xs text-amber-900">
          {w}
        </p>
      ))}
      <div className="border-t border-gray-200 p-3">
        <Link to={`/projects/${projectId}`} className="btn-secondary w-full">
          <Database size={13} />
          Edit dataset & split
        </Link>
      </div>
    </div>
  )
}

function NumberField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  hint?: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  disabled: boolean
}) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-gray-700">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
      />
      {hint && <p className="mt-0.5 text-[11px] text-gray-400">{hint}</p>}
    </div>
  )
}

function JobHistory({ jobs }: { jobs: TrainingJob[] }) {
  return (
    <div className="card">
      <div className="border-b border-gray-200 px-3 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">Recent runs</h2>
      </div>
      <ul className="divide-y divide-gray-100">
        {jobs.slice(0, 6).map((j) => (
          <li key={j.id} className="flex items-center justify-between px-3 py-2 text-xs">
            <div className="min-w-0">
              <p className="truncate font-medium text-gray-800">{j.trainer_key}</p>
              <p className="tabular-nums text-gray-500">
                {j.best_map !== null ? `mAP ${j.best_map.toFixed(3)} · ` : ''}
                {new Date(j.created_at).toLocaleTimeString()}
              </p>
            </div>
            <StatusBadge status={toStatus(j.status)} />
          </li>
        ))}
      </ul>
    </div>
  )
}
