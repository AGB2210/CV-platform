import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Cpu, Download, Play, Sparkles, SquarePen } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { StatusBadge, type Status } from '@/components/StatusBadge'
import {
  exportUrl,
  getAnnotatePreview,
  getAnnotationSummary,
  getDevice,
  getJob,
  listAnnotators,
  listClasses,
  listExportFormats,
  listJobs,
  startAnnotation,
  type AnnotationJob,
  type AnnotatePreview,
  type AnnotationSummary,
  type AnnotatorInfo,
  type JobScope,
  type DeviceInfo,
  type ExportFormatInfo,
  type ProjectClass,
} from '@/lib/api'

/** How often to poll a running job. 1s is responsive without hammering the API;
 *  inference takes ~1s/image on this GPU, so faster polling would mostly return
 *  identical numbers. */
const POLL_MS = 1000

export function Annotate() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [annotators, setAnnotators] = useState<AnnotatorInfo[]>([])
  const [device, setDevice] = useState<DeviceInfo | null>(null)
  const [classes, setClasses] = useState<ProjectClass[]>([])
  const [formats, setFormats] = useState<ExportFormatInfo[]>([])
  const [summary, setSummary] = useState<AnnotationSummary | null>(null)
  const [jobs, setJobs] = useState<AnnotationJob[]>([])

  const [modelKey, setModelKey] = useState('')
  const [boxThreshold, setBoxThreshold] = useState(0.3)
  const [textThreshold, setTextThreshold] = useState(0.25)
  const [prompts, setPrompts] = useState<Record<string, string>>({})
  const [clearExisting, setClearExisting] = useState(false)
  const [scope, setScope] = useState<JobScope>('staging')
  const [pre, setPre] = useState<AnnotatePreview | null>(null)

  const [activeJob, setActiveJob] = useState<AnnotationJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refreshSummary = useCallback(async () => {
    const [s, j, p] = await Promise.all([
      getAnnotationSummary(projectId),
      listJobs(projectId),
      getAnnotatePreview(projectId),
    ])
    setSummary(s)
    setJobs(j)
    setPre(p)
  }, [projectId])

  // Initial load: everything the page needs, in parallel.
  useEffect(() => {
    let cancelled = false
    Promise.all([
      listAnnotators(),
      getDevice(),
      listClasses(projectId),
      listExportFormats(),
      getAnnotationSummary(projectId),
      listJobs(projectId),
      getAnnotatePreview(projectId),
    ])
      .then(([a, d, c, f, s, j, p]) => {
        if (cancelled) return
        setAnnotators(a)
        setDevice(d)
        setClasses(c)
        setFormats(f)
        setSummary(s)
        setJobs(j)
        setPre(p)
        // Default to the first registered model rather than hardcoding a key —
        // the backend decides what exists.
        if (a.length) setModelKey((k) => k || a[0].key)
        // Resume polling if a job is already in flight (e.g. you reloaded the
        // page mid-run). This is exactly why job state lives in the DB.
        const running = j.find((x) => x.status === 'running' || x.status === 'queued')
        if (running) setActiveJob(running)
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [projectId])

  // --- Polling -----------------------------------------------------------
  // setInterval in a ref so the cleanup can always clear it, and so a re-render
  // never stacks a second interval on top of the first.
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
        const fresh = await getJob(activeJob.id)
        setActiveJob(fresh)
        if (fresh.status === 'done' || fresh.status === 'failed') {
          // Terminal state — stop polling and refresh the counts once.
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          void refreshSummary()
        }
      } catch (e) {
        setError((e as Error).message)
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
      }
    }, POLL_MS)

    // Cleanup runs on unmount AND before each re-run of this effect. Without it
    // navigating away leaves the interval firing forever against a dead
    // component — a real memory leak, not a theoretical one.
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [activeJob, refreshSummary])

  async function run() {
    setError(null)
    try {
      const job = await startAnnotation(projectId, {
        model_key: modelKey,
        box_threshold: boxThreshold,
        text_threshold: textThreshold,
        clear_existing: clearExisting,
        scope,
        // Only send non-empty overrides; the backend falls back to class names.
        prompts: Object.fromEntries(
          Object.entries(prompts).filter(([, v]) => v.trim()),
        ),
      })
      setActiveJob(job)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const selected = annotators.find((a) => a.key === modelKey)
  const isRunning = activeJob?.status === 'running' || activeJob?.status === 'queued'
  // Gate on the SELECTED scope's count, not the project total: with everything
  // committed, "Staging only" has nothing to do and the run would 400.
  const scopeCount = pre?.scope_counts?.[scope] ?? 0
  const canRun = !isRunning && classes.length > 0 && scopeCount > 0

  if (loading) {
    return (
      <>
        <PageHeader title="Auto-annotate" />
        <PageBody>
          <p className="text-sm text-gray-500">Loading…</p>
        </PageBody>
      </>
    )
  }

  return (
    <>
      <PageHeader
        title="Auto-annotate"
        description="Generate draft bounding boxes with a zero-shot model"
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
            <div className="card">
              <div className="border-b border-gray-200 px-4 py-3">
                <h2 className="text-sm font-medium text-gray-900">Model</h2>
              </div>
              <div className="space-y-3 p-4">
                <div>
                  <label
                    htmlFor="model"
                    className="mb-1 block text-xs font-medium text-gray-700"
                  >
                    Auto-annotation model
                  </label>
                  <select
                    id="model"
                    value={modelKey}
                    onChange={(e) => setModelKey(e.target.value)}
                    disabled={isRunning}
                    className="w-full rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-sm focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                  >
                    {annotators.map((a) => (
                      <option key={a.key} value={a.key}>
                        {a.display_name} (~{a.approx_vram_gb} GB VRAM)
                      </option>
                    ))}
                  </select>
                  {selected && (
                    <p className="mt-1 text-xs text-gray-500">{selected.description}</p>
                  )}
                </div>

                {/* Thresholds. Sliders rather than number inputs: these are
                    values you tune by feel against results, not by typing an
                    exact figure. */}
                <div className="grid grid-cols-2 gap-3">
                  <Slider
                    label="Box threshold"
                    hint="Min detection confidence"
                    value={boxThreshold}
                    onChange={setBoxThreshold}
                    disabled={isRunning}
                  />
                  <Slider
                    label="Text threshold"
                    hint="Min text-match score"
                    value={textThreshold}
                    onChange={setTextThreshold}
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>

            <div className="card">
              <div className="border-b border-gray-200 px-4 py-3">
                <h2 className="text-sm font-medium text-gray-900">Prompts</h2>
                <p className="text-xs text-gray-500">
                  Grounding DINO is sensitive to wording. Override a class name with a
                  fuller phrase if detection is poor — the stored label stays the class
                  name.
                </p>
              </div>
              {classes.length === 0 ? (
                <p className="px-4 py-4 text-xs text-gray-500">
                  No classes defined.{' '}
                  <Link to={`/projects/${projectId}`} className="text-accent-700 underline">
                    Add classes
                  </Link>{' '}
                  before annotating.
                </p>
              ) : (
                <ul className="divide-y divide-gray-100">
                  {classes.map((c) => (
                    <li key={c.id} className="flex items-center gap-2 px-4 py-2">
                      <span
                        className="h-3 w-3 shrink-0 rounded-sm border border-black/10"
                        style={{ backgroundColor: c.color }}
                      />
                      <span className="w-28 shrink-0 truncate text-sm text-gray-800">
                        {c.name}
                      </span>
                      <input
                        value={prompts[c.name] ?? ''}
                        onChange={(e) =>
                          setPrompts((p) => ({ ...p, [c.name]: e.target.value }))
                        }
                        disabled={isRunning}
                        placeholder={c.name}
                        className="min-w-0 flex-1 rounded-md border border-gray-300 px-2 py-1 text-sm placeholder:text-gray-300 focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                      />
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* --- Which images ---
                An unscoped run re-annotated images already committed to the
                dataset, which bounced every one back to staging — so labelling
                three new uploads silently emptied a whole dataset. Scoping to
                staging by default makes the normal case incapable of touching
                committed work. */}
            <div className="card">
              <div className="border-b border-gray-200 px-4 py-3">
                <h2 className="text-sm font-medium text-gray-900">Which images</h2>
                <p className="text-xs text-gray-500">
                  Images already in your dataset are left alone unless you say otherwise.
                </p>
              </div>
              <div className="space-y-1.5 p-4">
                {(
                  [
                    {
                      value: 'staging' as const,
                      label: 'Staging only',
                      blurb:
                        'Images not yet added to the dataset. Your dataset is not touched.',
                    },
                    {
                      value: 'unannotated' as const,
                      label: 'Unannotated only',
                      blurb: 'Images with no boxes yet. Fills gaps, changes nothing else.',
                    },
                    {
                      value: 'all' as const,
                      label: 'All images',
                      blurb:
                        'Re-annotate everything, including images already in the dataset.',
                    },
                  ]
                ).map((o) => (
                  <label
                    key={o.value}
                    className={`flex cursor-pointer gap-2 rounded-md border p-2 transition-colors ${
                      scope === o.value
                        ? 'border-accent-500 bg-accent-50'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <input
                      type="radio"
                      name="scope"
                      checked={scope === o.value}
                      onChange={() => setScope(o.value)}
                      disabled={isRunning}
                      className="mt-0.5 accent-accent-600"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-baseline justify-between gap-2">
                        <span className="text-sm font-medium text-gray-900">{o.label}</span>
                        <span className="font-mono text-xs tabular-nums text-gray-500">
                          {pre?.scope_counts?.[o.value] ?? 0} image
                          {(pre?.scope_counts?.[o.value] ?? 0) === 1 ? '' : 's'}
                        </span>
                      </span>
                      <span className="block text-xs text-gray-500">{o.blurb}</span>
                    </span>
                  </label>
                ))}

                {/* No dataset warning here any more, and that's deliberate: a
                    run writes proposals, which don't change accepted
                    annotations, so even scope="all" leaves the dataset exactly
                    as it was. Keeping a scary warning that is no longer true
                    would just teach people to ignore warnings. */}
                {scope === 'all' && pre && pre.images_in_dataset > 0 && (
                  <p className="rounded-md border border-gray-200 bg-gray-50 px-2.5 py-1.5 text-xs text-gray-600">
                    Includes {pre.images_in_dataset} image
                    {pre.images_in_dataset === 1 ? '' : 's'} already in your dataset. Their
                    existing boxes are untouched — the model's output arrives as proposals
                    you accept or reject in Annotate.
                  </p>
                )}
              </div>
            </div>

            {/* --- What this run will do to what's already there ---
                Auto-annotation is NOT additive: it clears prior output before
                writing new output. Leaving that implicit meant a re-run could
                silently delete hand-drawn boxes, and it meant a project with
                manual boxes showed a confusing mix afterwards with no way to
                get a clean model-only result. */}
            <div className="card">
              <div className="border-b border-gray-200 px-4 py-3">
                <h2 className="text-sm font-medium text-gray-900">Existing annotations</h2>
                <p className="text-xs text-gray-500">
                  A run always replaces its own previous output. Choose what happens to
                  everything else.
                </p>
              </div>
              <div className="space-y-3 p-4">
                {pre && (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
                    <Stat label="From model" value={pre.auto_boxes} note="always replaced" />
                    <Stat
                      label="Hand-drawn"
                      value={pre.manual_boxes}
                      note={clearExisting ? 'will be DELETED' : 'kept'}
                      danger={clearExisting && pre.manual_boxes > 0}
                    />
                    <Stat
                      label="Imported"
                      value={pre.imported_boxes}
                      note={clearExisting ? 'will be DELETED' : 'kept'}
                      danger={clearExisting && pre.imported_boxes > 0}
                    />
                  </div>
                )}

                <label className="flex cursor-pointer items-start gap-2">
                  <input
                    type="checkbox"
                    checked={clearExisting}
                    onChange={(e) => setClearExisting(e.target.checked)}
                    disabled={isRunning}
                    className="mt-0.5 accent-accent-600"
                  />
                  <span>
                    <span className="block text-sm text-gray-900">
                      Replace all annotations
                    </span>
                    <span className="block text-xs text-gray-500">
                      Delete every existing box first, so the result is only this model's
                      output. Leave unticked to keep hand-drawn and imported boxes.
                    </span>
                  </span>
                </label>

                {clearExisting && pre && pre.manual_boxes + pre.imported_boxes > 0 && (
                  <p className="rounded-md border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs text-red-800">
                    <span className="font-medium">
                      {pre.manual_boxes + pre.imported_boxes} hand-drawn/imported box
                      {pre.manual_boxes + pre.imported_boxes === 1 ? '' : 'es'} will be
                      permanently deleted.
                    </span>{' '}
                    This cannot be undone.
                  </p>
                )}

              </div>
            </div>

            <div className="flex items-center gap-3">
              <button className="btn-primary" onClick={() => void run()} disabled={!canRun}>
                {isRunning ? (
                  <>Running…</>
                ) : (
                  <>
                    <Play size={14} />
                    Annotate {scopeCount} image{scopeCount === 1 ? '' : 's'}
                  </>
                )}
              </button>
              {classes.length === 0 && (
                <span className="text-xs text-gray-500">Add a class first</span>
              )}
              {/* Say WHY it's disabled. "0 images" with a full grid on screen is
                  baffling unless you know they're all committed. */}
              {classes.length > 0 && scopeCount === 0 && !isRunning && (
                <span className="text-xs text-gray-500">
                  {(summary?.total_images ?? 0) === 0
                    ? 'Upload images first'
                    : scope === 'staging'
                      ? 'No staging images — everything is already in the dataset. Choose "All images" to re-annotate.'
                      : 'No images match this scope.'}
                </span>
              )}
            </div>

            {activeJob && <JobProgress job={activeJob} />}
          </section>

          {/* --- Right: status --- */}
          <aside className="space-y-4">
            {device && <DeviceCard device={device} />}
            {summary && <SummaryCard summary={summary} />}
            <ExportCard projectId={projectId} formats={formats} />
            {jobs.length > 0 && <JobHistory jobs={jobs} />}
          </aside>
        </div>
      </PageBody>
    </>
  )
}

function Stat({
  label,
  value,
  note,
  danger,
}: {
  label: string
  value: number
  note: string
  danger?: boolean
}) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono font-medium tabular-nums text-gray-900">{value}</span>
      <span className={danger ? 'font-medium text-red-700' : 'text-gray-400'}>
        ({note})
      </span>
    </span>
  )
}

function Slider({
  label,
  hint,
  value,
  onChange,
  disabled,
}: {
  label: string
  hint: string
  value: number
  onChange: (v: number) => void
  disabled: boolean
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label className="text-xs font-medium text-gray-700">{label}</label>
        <span className="font-mono text-xs tabular-nums text-gray-600">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={0.05}
        max={0.95}
        step={0.05}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent-600"
      />
      <p className="mt-0.5 text-xs text-gray-400">{hint}</p>
    </div>
  )
}

function JobProgress({ job }: { job: AnnotationJob }) {
  const status: Status =
    job.status === 'done'
      ? 'done'
      : job.status === 'failed'
        ? 'failed'
        : job.status === 'running'
          ? 'running'
          : 'queued'

  return (
    <div className="card">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">Job #{job.id}</h2>
        <StatusBadge status={status} />
      </div>
      <div className="p-4">
        {/* A real progress bar, driven by processed/total from the DB. */}
        <div className="mb-1.5 flex items-baseline justify-between text-xs">
          <span className="text-gray-600">
            {job.processed_images} / {job.total_images} images
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

        <div className="mt-2 flex items-center justify-between">
          <p className="text-xs text-gray-600">
            <span className="font-medium tabular-nums text-gray-900">
              {job.boxes_created}
            </span>{' '}
            {/* "proposed", not "created": they aren't part of the dataset until
                accepted, and calling them created implies work that's done. */}
            boxes proposed
          </p>
          {/* The step that makes the whole run meaningful. A box count is not
              evidence — it's equally consistent with 9 good boxes and 9 boxes
              in the wrong place. Without this link the only way to see what the
              model did is to export the dataset and open it somewhere else. */}
          {job.status === 'done' && job.boxes_created > 0 && (
            <Link to={`/projects/${job.project_id}/review`} className="btn-primary">
              <SquarePen size={13} />
              Review {job.boxes_created} proposals
            </Link>
          )}
        </div>

        {job.error && (
          // whitespace-pre-wrap + a scroll cap: tracebacks are long, and
          // truncating the one thing that explains a failure is cruel.
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded border border-red-200 bg-red-50 p-2 text-[11px] text-red-900">
            {job.error}
          </pre>
        )}
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
            <dd className="font-mono tabular-nums text-gray-900">
              {device.total_vram_gb} GB
            </dd>
          </div>
        )}
        <div className="flex justify-between">
          <dt className="text-gray-500">Backend</dt>
          <dd className="font-mono text-gray-900">{device.device}</dd>
        </div>
      </dl>
      {/* Warn loudly on CPU. Grounding DINO on CPU is minutes per image — the
          user deserves to know before queueing 500 of them. */}
      {!onGpu && (
        <p className="border-t border-gray-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {device.note ?? 'Running on CPU — expect very slow inference.'}
        </p>
      )}
    </div>
  )
}

function SummaryCard({ summary }: { summary: AnnotationSummary }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2.5">
        <Sparkles size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">Annotations</h2>
      </div>
      <dl className="space-y-1.5 p-3 text-xs">
        <Row label="Images annotated" value={`${summary.annotated_images} / ${summary.total_images}`} />
        <Row label="Total boxes" value={summary.total_boxes} />
        <Row label="From model" value={summary.auto_boxes} />
        <Row label="Manual" value={summary.manual_boxes} />
        <Row label="Reviewed" value={summary.reviewed_boxes} />
      </dl>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-mono tabular-nums text-gray-900">{value}</dd>
    </div>
  )
}

function ExportCard({
  projectId,
  formats,
}: {
  projectId: number
  formats: ExportFormatInfo[]
}) {
  const [format, setFormat] = useState('coco')
  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2.5">
        <Download size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">Export</h2>
      </div>
      <div className="space-y-2 p-3">
        <select
          value={format}
          onChange={(e) => setFormat(e.target.value)}
          className="w-full rounded-md border border-gray-300 bg-white px-2 py-1 text-xs focus:border-accent-500 focus:outline-none"
        >
          {formats.map((f) => (
            <option key={f.key} value={f.key}>
              {f.display_name}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-400">
          {formats.find((f) => f.key === format)?.description}
        </p>
        {/* An <a download>, not a fetch — native download UI and streaming. */}
        <a href={exportUrl(projectId, format)} download className="btn-secondary w-full">
          <Download size={13} />
          Download dataset
        </a>
      </div>
    </div>
  )
}

function JobHistory({ jobs }: { jobs: AnnotationJob[] }) {
  return (
    <div className="card">
      <div className="border-b border-gray-200 px-3 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">Recent jobs</h2>
      </div>
      <ul className="divide-y divide-gray-100">
        {jobs.slice(0, 6).map((j) => (
          <li key={j.id} className="flex items-center justify-between px-3 py-2 text-xs">
            <div className="min-w-0">
              <p className="truncate font-medium text-gray-800">{j.model_key}</p>
              <p className="tabular-nums text-gray-500">
                {j.boxes_created} boxes · {new Date(j.created_at).toLocaleTimeString()}
              </p>
            </div>
            <StatusBadge
              status={
                j.status === 'done'
                  ? 'done'
                  : j.status === 'failed'
                    ? 'failed'
                    : j.status === 'running'
                      ? 'running'
                      : 'queued'
              }
            />
          </li>
        ))}
      </ul>
    </div>
  )
}
