import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Pencil,
  Play,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
} from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { StatusBadge, type Status } from '@/components/StatusBadge'
import { MetricsChart } from '@/components/MetricsChart'
import { ConfirmDialog } from '@/components/ui/Modal'
import {
  RenameDialog,
  RowAction,
  RowCheckbox,
  SelectionToolbar,
} from '@/components/VersionAdmin'
import { useVersionSelection } from '@/lib/useVersionSelection'
import {
  ApiError,
  bulkDeleteTrainingJobs,
  cancelTrainingJob,
  deleteTrainingJob,
  getDevice,
  getTrainPreview,
  listDatasetVersions,
  listTrainers,
  listTrainingJobs,
  getTrainingJob,
  renameTrainingJob,
  startTraining,
  stopTrainingJob,
  versionLabel,
  type DatasetVersion,
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
  const [datasetVersions, setDatasetVersions] = useState<DatasetVersion[]>([])
  // Which saved dataset version to train. null = the latest save.
  const [datasetVersionId, setDatasetVersionId] = useState<number | null>(null)

  const [trainerKey, setTrainerKey] = useState('')
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(8)
  const [imageSize, setImageSize] = useState(640)
  const [lr, setLr] = useState('')
  // Tuning knobs stay hidden until asked for — see the config card.
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Finetune source: a completed run's id, or null to start from the pretrained
  // base. Lets you keep improving a model instead of re-learning from zero.
  const [initFromId, setInitFromId] = useState<number | null>(null)

  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null)
  // Which past run the user is inspecting in the detail panel.
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  //: Non-error feedback, e.g. a cancel that completed as intended.
  const [notice, setNotice] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  async function requestStop() {
    setError(null)
    try {
      setActiveJob(await stopTrainingJob(activeJob!.id))
      setNotice('Stopping after the current epoch — the model so far will be kept.')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function requestCancel() {
    setError(null)
    try {
      await cancelTrainingJob(activeJob!.id)
      // The runner tears the row down at the end of the epoch in flight, so the
      // poller sees a 404 shortly and clears the card.
      setNotice('Cancelling after the current epoch — this run will be discarded.')
      setActiveJob((j) => (j ? { ...j, control: 'cancel' } : j))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const jobRef = useRef<HTMLDivElement>(null)
  const scrolledForJob = useRef<number | null>(null)
  useEffect(() => {
    if (!activeJob || scrolledForJob.current === activeJob.id) return
    scrolledForJob.current = activeJob.id
    jobRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [activeJob])

  const refresh = useCallback(async () => {
    const [p, j, dv] = await Promise.all([
      getTrainPreview(projectId),
      listTrainingJobs(projectId),
      listDatasetVersions(projectId),
    ])
    setPreview(p)
    setJobs(j)
    setDatasetVersions(dv)
  }, [projectId])

  useEffect(() => {
    let cancelled = false
    Promise.all([
      listTrainers(),
      getDevice(),
      getTrainPreview(projectId),
      listTrainingJobs(projectId),
      listDatasetVersions(projectId),
    ])
      .then(([t, d, p, j, dv]) => {
        if (cancelled) return
        setTrainers(t)
        setDevice(d)
        setPreview(p)
        setJobs(j)
        setDatasetVersions(dv)
        if (t.length) {
          setTrainerKey((k) => k || t[0].key)
          setEpochs(t[0].default_epochs)
          setBatchSize(t[0].default_batch_size)
          setImageSize(t[0].default_image_size)
        }
        const running = j.find((x) => x.status === 'running' || x.status === 'queued')
        if (running) {
          setActiveJob(running)
          // Show the model that's actually training, not just the first in the
          // list — otherwise the form and the live version disagree.
          setTrainerKey(running.trainer_key)
        }
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [projectId])

  /** Apply a backend's recommended settings — what a run uses unless overridden. */
  function applyDefaults(t: TrainerInfo | undefined) {
    if (!t) return
    setEpochs(t.default_epochs)
    setBatchSize(t.default_batch_size)
    setImageSize(t.default_image_size)
    setLr('')
  }

  function resetParams() {
    applyDefaults(trainers.find((x) => x.key === trainerKey))
  }

  function selectTrainer(key: string) {
    setTrainerKey(key)
    // Versions are per-model, so a selection from the old model's list would
    // point at something no longer shown. Fall back to that model's latest.
    setSelectedId(null)
    setInitFromId(null)
    // Defaults are per-architecture: a sane batch for YOLO-nano isn't one for a
    // DETR, so switching backend must not carry the old numbers across.
    applyDefaults(trainers.find((x) => x.key === key))
  }

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
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
        // A cancelled run deletes itself, so the very next poll 404s. That's the
        // expected end of a cancel, not a failure to report as one.
        if ((e as ApiError).status === 404) {
          setActiveJob(null)
          setNotice('Training cancelled — nothing was saved.')
          void refresh()
          return
        }
        setError((e as Error).message)
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
        learning_rate: lr.trim() ? Number(lr) : null,
        init_from_job_id: initFromId,
        dataset_version_id: datasetVersionId,
      })
      setActiveJob(job)
      setSelectedId(null) // show the live run, not whatever was being inspected
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const selected = trainers.find((t) => t.key === trainerKey)
  const isRunning = activeJob?.status === 'running' || activeJob?.status === 'queued'
  const noTrainers = trainers.length === 0
  const canRun = !isRunning && !noTrainers && !!preview?.can_train

  // Versions are scoped to THIS project (the endpoint) AND this model — a YOLO
  // history shouldn't list RF-DETR versions, and you can't continue one
  // architecture's weights from another's. Filtered client-side so switching the
  // model dropdown re-scopes instantly with no round trip.
  const modelJobs = jobs.filter((j) => j.trainer_key === trainerKey)
  // Versions that produced a checkpoint — the candidates to continue from.
  const completedRuns = modelJobs.filter((j) => j.status === 'done' && j.checkpoint_path)
  const latestVersion = modelJobs.length ? Math.max(...modelJobs.map((j) => j.version)) : 0

  // What the detail panel shows: the live run takes precedence, else the version
  // the user clicked, else the newest version of this model.
  const runningJob = isRunning ? activeJob : null
  const displayedJob =
    runningJob ?? modelJobs.find((j) => j.id === selectedId) ?? activeJob ?? modelJobs[0] ?? null

  /** How a job id presents in text — its NAME if it has one, else "v{n}".
   *
   *  Resolved from the live job list on every render rather than stored with
   *  the run that continued it. Renaming v1 to "baseline" has to change every
   *  place v1 is referred to, including "continued from v1" on the three runs
   *  built off it — a name that only updates in one list is worse than no name,
   *  because the two disagree on screen at the same time. */
  const labelOf = (jobId: number) => {
    const job = jobs.find((j) => j.id === jobId)
    return job ? versionLabel(job) : null
  }
  /** Dataset version number a run trained on. */
  const datasetVersionOf = (id: number | null) =>
    id === null ? null : (datasetVersions.find((v) => v.id === id)?.version ?? null)

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
        {/* Neutral, not red: a cancel that worked is an outcome, not a fault. */}
        {notice && (
          <div className="mb-4 flex items-start justify-between gap-2 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-700">
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} className="text-gray-400 hover:text-gray-600">
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          {/* --- Left: run detail + configure --- */}
          <section className="space-y-4">
            {displayedJob && (
              <div ref={jobRef}>
                <RunDetail
                  job={displayedJob}
                  live={displayedJob.id === runningJob?.id}
                  fromVersion={
                    displayedJob.init_from_job_id !== null
                      ? labelOf(displayedJob.init_from_job_id)
                      : null
                  }
                  datasetVersion={datasetVersionOf(displayedJob.dataset_version_id)}
                  onStop={requestStop}
                  onCancel={requestCancel}
                />
              </div>
            )}

            {noTrainers ? (
              <NoTrainersCard />
            ) : (
              <div className="card">
                <div className="border-b border-gray-200 px-4 py-3">
                  <h2 className="text-sm font-medium text-gray-900">
                    {isRunning ? 'Configuration' : 'New run'}
                  </h2>
                </div>
                <div className="space-y-3 p-4">
                  <div>
                    <label htmlFor="trainer" className="mb-1 block text-xs font-medium text-gray-700">
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
                          {/* Just the name. A "~3 GB VRAM" suffix reads as a
                              property of the model, but what matters is whether
                              it fits THIS machine — which the memory hint below
                              works out from the detected device instead. */}
                          {t.display_name}
                        </option>
                      ))}
                    </select>
                    {selected && <p className="mt-1 text-xs text-gray-500">{selected.description}</p>}
                  </div>

                  {/* --- Which saved dataset to train ---
                      Training always runs against a SAVED version, never the
                      live rows, so a run's results stay attributable. */}
                  <div>
                    <label htmlFor="dsver" className="mb-1 block text-xs font-medium text-gray-700">
                      Dataset version
                    </label>
                    <select
                      id="dsver"
                      value={datasetVersionId ?? ''}
                      onChange={(e) =>
                        setDatasetVersionId(e.target.value ? Number(e.target.value) : null)
                      }
                      disabled={isRunning || datasetVersions.length === 0}
                      className="w-full rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-sm focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                    >
                      {/* The default is the version the dataset ON SCREEN
                          matches, which after a restore is NOT the newest save
                          point. Saying "latest" there would promise to train
                          something other than what the user is looking at. */}
                      <option value="">
                        {!datasetVersions.length
                          ? 'No saved dataset yet'
                          : preview?.current_version
                            ? `Current dataset (v${preview.current_version})`
                            : `Latest saved (v${datasetVersions[0].version})`}
                      </option>
                      {datasetVersions.map((v) => (
                        <option key={v.id} value={v.id}>
                          v{v.version}
                          {v.is_current ? ' (current)' : ''} · {v.total_images} imgs ·{' '}
                          {v.total_boxes} boxes{v.note ? ` · ${v.note}` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="mt-0.5 flex items-center gap-1 text-xs text-gray-400">
                      <Database size={11} />
                      Trains that saved snapshot — later dataset edits don't change it.
                    </p>
                    {preview?.has_unsaved_changes && (
                      <p className="mt-1 rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-900">
                        The dataset has changed since it was last saved. Save it to train
                        those changes, or pick a version above.
                      </p>
                    )}
                  </div>

                  {/* --- Start from: pretrained vs continue a previous run --- */}
                  <div>
                    <label htmlFor="initfrom" className="mb-1 block text-xs font-medium text-gray-700">
                      Initialize from
                    </label>
                    <select
                      id="initfrom"
                      value={initFromId ?? ''}
                      onChange={(e) => setInitFromId(e.target.value ? Number(e.target.value) : null)}
                      disabled={isRunning}
                      className="w-full rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-sm focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
                    >
                      <option value="">Pretrained weights (from scratch)</option>
                      {completedRuns.map((j) => (
                        <option key={j.id} value={j.id}>
                          Continue {versionLabel(j)}
                          {j.best_map !== null ? ` · mAP ${j.best_map.toFixed(3)}` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="mt-0.5 flex items-center gap-1 text-xs text-gray-400">
                      <GitBranch size={11} />
                      {initFromId
                        ? `Builds on ${labelOf(initFromId)}'s weights, trained on the current dataset.`
                        : 'Continue a finished version to keep improving it instead of re-learning from zero.'}
                    </p>
                  </div>

                  {/* --- Tuning knobs, hidden by default ---
                      Most runs want the backend's tuned defaults, and a wall of
                      numeric fields makes a simple "train it" look like it needs
                      expertise. So the defaults are stated in one line and the
                      controls only appear when asked for. */}
                  <div className="rounded-md border border-gray-200">
                    {/* The WHOLE collapsed box is the control, not just its top
                        row: the summary line underneath is part of the same
                        affordance, so clicking it should open the section too.
                        Spans rather than <p>/<div> because a <button> may only
                        contain phrasing content. */}
                    <button
                      type="button"
                      onClick={() => setShowAdvanced((v) => !v)}
                      className="block w-full rounded-md text-left hover:bg-gray-50"
                    >
                      <span className="flex items-center justify-between px-2.5 py-2">
                        <span className="flex items-center gap-1.5 text-xs font-medium text-gray-700">
                          <SlidersHorizontal size={12} className="text-gray-400" />
                          Custom parameters
                        </span>
                        <span className="flex items-center gap-1.5">
                          {!showAdvanced && (
                            <span className="font-mono text-[11px] tabular-nums text-gray-400">
                              {epochs} epochs · batch {batchSize} · {imageSize}px
                            </span>
                          )}
                          {showAdvanced ? (
                            <ChevronUp size={13} className="text-gray-400" />
                          ) : (
                            <ChevronDown size={13} className="text-gray-400" />
                          )}
                        </span>
                      </span>
                      {!showAdvanced && (
                        <span className="block px-2.5 pb-2 text-xs text-gray-400">
                          Using {selected?.display_name ?? 'the backend'}'s recommended
                          settings. Open to override.
                        </span>
                      )}
                    </button>

                    {showAdvanced && (
                      <div className="space-y-3 border-t border-gray-200 p-2.5">
                        <div className="grid grid-cols-3 gap-3">
                          <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} max={1000} disabled={isRunning} />
                          <NumberField label="Batch size" hint="lower if OOM" value={batchSize} onChange={setBatchSize} min={1} max={128} disabled={isRunning} />
                          <NumberField label="Image size" hint="px, square" value={imageSize} onChange={setImageSize} min={64} max={2048} step={32} disabled={isRunning} />
                        </div>

                        <div>
                          <label htmlFor="lr" className="mb-1 block text-xs font-medium text-gray-700">
                            Learning rate <span className="font-normal text-gray-400">(optional)</span>
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
                            Leave empty to use the backend's tuned schedule — usually the right choice.
                          </p>
                        </div>

                        <div className="flex items-center justify-between gap-2">
                          <p className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">
                            {vramHint(device, selected)}
                          </p>
                          <button
                            type="button"
                            onClick={() => resetParams()}
                            disabled={isRunning}
                            className="shrink-0 text-xs text-accent-700 hover:underline disabled:text-gray-300"
                          >
                            Reset
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
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
                    {initFromId
                      ? `Continue ${labelOf(initFromId)} → v${latestVersion + 1}`
                      : `Train v${latestVersion + 1}`}
                  </>
                )}
              </button>
              {!isRunning && noTrainers && <span className="text-xs text-gray-500">No backend installed</span>}
              {/* Say exactly what's missing. "Save the dataset" is the common
                  one and is actionable, so it links straight there. */}
              {!isRunning && !noTrainers && preview && !preview.can_train && (
                <span className="text-xs text-gray-500">
                  {!preview.has_saved_version ? (
                    <>
                      <Link to={`/projects/${projectId}`} className="text-accent-700 underline">
                        Save the dataset
                      </Link>{' '}
                      first — training runs against a saved version.
                    </>
                  ) : preview.num_classes === 0 ? (
                    'Add a class first'
                  ) : preview.splits.val.images === 0 ? (
                    <>
                      Needs a{' '}
                      <Link to={`/projects/${projectId}`} className="text-accent-700 underline">
                        validation split
                      </Link>{' '}
                      — without held-out images the mAP would be scored on the
                      training data.
                    </>
                  ) : (
                    'The saved dataset has no boxes in its train split'
                  )}
                </span>
              )}
            </div>
          </section>

          {/* --- Right: status --- */}
          <aside className="space-y-4">
            {device && <DeviceCard device={device} />}
            {preview && <DatasetCard preview={preview} projectId={projectId} />}
            {modelJobs.length > 0 && (
              <VersionHistory
                projectId={projectId}
                jobs={modelJobs}
                latestVersion={latestVersion}
                selectedId={displayedJob?.id ?? null}
                onSelect={setSelectedId}
                onChanged={refresh}
                onError={setError}
              />
            )}
          </aside>
        </div>
      </PageBody>
    </>
  )
}

/**
 * Memory guidance for THIS machine, not the one the app was written on.
 *
 * The app ships to other people's computers, so a hardcoded "on a 4 GB GPU…"
 * is wrong for almost everyone who runs it. The two numbers needed are already
 * available and both come from the running system: how much VRAM the detected
 * device reports, and roughly how much the selected backend wants. Comparing
 * them says something true on a 2 GB laptop and on a 24 GB workstation.
 */
function vramHint(device: DeviceInfo | null, trainer: TrainerInfo | undefined): string {
  const fallback =
    'If a run fails with an out-of-memory error, halve the batch size and try again.'
  if (!device || device.device !== 'cuda' || !device.total_vram_gb) return fallback

  const have = device.total_vram_gb
  const need = trainer?.approx_vram_gb ?? 0
  const name = trainer?.display_name ?? 'This backend'
  if (!need) return `This GPU reports ${have} GB. ${fallback}`

  if (have < need) {
    return `${name} needs roughly ${need} GB and this GPU has ${have} GB — lower the batch size and image size, or expect an out-of-memory error.`
  }
  if (have < need * 1.5) {
    return `${name} needs roughly ${need} GB of the ${have} GB on this GPU. It fits, but keep batch and image size modest — if a run runs out of memory, halve the batch size.`
  }
  return `This GPU has ${have} GB, comfortably above the ~${need} GB ${name} needs. ${fallback}`
}

function toStatus(s: TrainingJob['status']): Status {
  return s === 'done' ? 'done' : s === 'failed' ? 'failed' : s === 'running' ? 'running' : 'queued'
}

function fmt(v: number | null | undefined, dp = 3): string {
  return v === null || v === undefined ? '—' : v.toFixed(dp)
}

function RunDetail({
  job,
  live,
  fromVersion,
  datasetVersion,
  onStop,
  onCancel,
}: {
  job: TrainingJob
  live: boolean
  fromVersion: string | null
  datasetVersion: number | null
  onStop: () => void
  onCancel: () => void
}) {
  // Once an instruction is in, both buttons go — pressing Cancel after Stop (or
  // twice) has no further meaning and would only invite doubt about what's
  // happening.
  const winding = job.control !== null
  return (
    <div className={`card transition-shadow ${live ? 'ring-2 ring-accent-400 ring-offset-2' : ''}`}>
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
        <h2 className="flex items-center gap-2 text-sm font-medium text-gray-900">
          {/* The NAME once it has one — the detail panel and the version list
              must not disagree about what a run is called. */}
          {live ? `Training ${versionLabel(job)}…` : versionLabel(job)}
          {job.init_from_job_id !== null && (
            <span className="flex items-center gap-1 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-normal text-gray-500">
              <GitBranch size={10} />
              {fromVersion !== null ? `continued from ${fromVersion}` : 'continued'}
            </span>
          )}
          {job.stopped_early && (
            <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-normal text-gray-500">
              stopped early
            </span>
          )}
        </h2>
        <span className="flex items-center gap-2">
          {live &&
            (winding ? (
              <span className="text-[11px] text-gray-500">
                {job.control === 'cancel' ? 'Cancelling…' : 'Stopping…'} after this epoch
              </span>
            ) : (
              <>
                {/* Stop keeps the model; Cancel throws it away. Different words,
                    different colours: cancel is the destructive one. */}
                {/* flex + gap rather than an inline icon with a margin: at
                    11px text a 9px glyph reads as undersized and sits off the
                    baseline. Matching the icon to the text size and centring
                    them on the same axis is what makes the pair look
                    deliberate. */}
                <button
                  onClick={onStop}
                  className="flex items-center gap-1 rounded border border-gray-300 px-1.5 py-0.5 text-[11px] leading-none text-gray-700 hover:bg-gray-50"
                  title="Finish the current epoch, then stop and keep this model"
                >
                  <Square size={11} />
                  Stop early
                </button>
                <button
                  onClick={onCancel}
                  className="flex items-center gap-1 rounded border border-red-300 px-1.5 py-0.5 text-[11px] leading-none text-red-700 hover:bg-red-50"
                  title="Stop and discard this run entirely — no version is kept"
                >
                  <X size={11} />
                  Cancel
                </button>
              </>
            ))}
          <StatusBadge status={toStatus(job.status)} />
        </span>
      </div>
      <div className="p-4">
        <div className="mb-1.5 flex items-baseline justify-between text-xs">
          <span className="text-gray-600">
            Epoch {job.current_epoch} / {job.total_epochs}
          </span>
          <span className="font-mono tabular-nums text-gray-600">{job.progress_pct}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className={`h-full rounded-full transition-all duration-300 ${job.status === 'failed' ? 'bg-status-bad' : 'bg-accent-600'}`}
            style={{ width: `${job.progress_pct}%` }}
          />
        </div>

        <div className="mt-3 grid grid-cols-3 gap-2">
          <Metric label="Train loss" value={fmt(job.train_loss)} />
          <Metric label="Val mAP" value={fmt(job.val_map)} />
          <Metric label="Best mAP" value={fmt(job.best_map)} accent />
        </div>

        {/* Epoch-vs-mAP chart — markers, hover values, zoom/pan. */}
        <div className="mt-3 rounded-md border border-gray-100 p-2">
          <MetricsChart points={job.metrics} />
        </div>

        {/* How this run was configured — the "how it did it" for history. */}
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
          <Detail label="Backend" value={job.trainer_key} />
          <Detail
            label="Dataset"
            value={datasetVersion !== null ? `v${datasetVersion}` : '—'}
          />
          <Detail label="Epochs" value={String(job.total_epochs)} />
          <Detail label="Batch" value={String(job.batch_size)} />
          <Detail label="Image size" value={`${job.image_size}px`} />
          <Detail label="Learning rate" value={job.learning_rate === null ? 'auto' : String(job.learning_rate)} />
          <Detail label="Train imgs" value={String(job.train_images)} />
          <Detail label="Val imgs" value={String(job.val_images)} />
          {/* 0 means "not recorded" — runs from before the column existed. Show
              a dash rather than a misleading zero. */}
          <Detail label="Classes" value={job.num_classes ? String(job.num_classes) : '—'} />
        </dl>

        {job.status === 'done' && (
          <p className="mt-3 rounded-md border border-status-good/30 bg-status-good/5 px-2.5 py-1.5 text-xs text-gray-700">
            Best checkpoint saved. Evaluation and a prediction playground arrive in the next phase.
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

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-gray-200 bg-gray-50 px-2 py-1.5">
      <p className="text-[10px] uppercase tracking-wide text-gray-400">{label}</p>
      <p className={`font-mono text-sm tabular-nums ${accent ? 'font-semibold text-accent-700' : 'text-gray-900'}`}>{value}</p>
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-[10px] uppercase tracking-wide text-gray-400">{label}</dt>
      <dd className="truncate font-mono tabular-nums text-gray-700">{value}</dd>
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
          The training pipeline is ready, but no backend is registered yet. Training pulls in
          heavy dependencies (PyTorch is already here; the trainer adds its own), kept out of the
          base install so the app stays light.
        </p>
        <p className="text-xs text-gray-500">
          Install a backend into the backend venv, then restart the server — it will appear here
          automatically, like the annotation models do.
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
      {!onGpu && (
        <p className="border-t border-gray-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          No CUDA GPU detected. Training on CPU is impractically slow — expect hours per epoch.
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
              <td className="px-3 py-1.5 capitalize text-gray-700">{s}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums text-gray-900">{preview.splits[s].images}</td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums text-gray-900">{preview.splits[s].boxes}</td>
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

/**
 * Versions of the CURRENTLY SELECTED model in this project, newest first.
 *
 * Scoped deliberately: mixing another model's versions (or another project's)
 * into one list makes the numbering meaningless — v3 of YOLO and v3 of RF-DETR
 * are unrelated models. The caller passes an already-filtered list.
 */
function VersionHistory({
  projectId,
  jobs,
  latestVersion,
  selectedId,
  onSelect,
  onChanged,
  onError,
}: {
  projectId: number
  jobs: TrainingJob[]
  latestVersion: number
  selectedId: number | null
  onSelect: (id: number) => void
  onChanged: () => void
  onError: (msg: string | null) => void
}) {
  const { selected, toggle, toggleAll, clear } = useVersionSelection()
  const [renaming, setRenaming] = useState<TrainingJob | null>(null)
  const [deleting, setDeleting] = useState<TrainingJob | null>(null)
  const [bulkOpen, setBulkOpen] = useState(false)
  const [busy, setBusy] = useState(false)

  const inFlight = (j: TrainingJob) => j.status === 'running' || j.status === 'queued'

  async function removeOne(j: TrainingJob) {
    setBusy(true)
    onError(null)
    try {
      await deleteTrainingJob(j.id)
      setDeleting(null)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  /** Same idea as the page-level resolver: a run's provenance is rendered from
   *  the source's CURRENT name, so renaming it updates every reference. Scoped
   *  to this model's list, which is where a finetune source always comes from. */
  const labelOf = (jobId: number) => {
    const source = jobs.find((j) => j.id === jobId)
    return source ? versionLabel(source) : null
  }

  async function removeSelected() {
    setBusy(true)
    onError(null)
    try {
      const r = await bulkDeleteTrainingJobs(projectId, [...selected])
      const skipped = Object.keys(r.skipped).length
      // A version still training is skipped, not fatal — say so rather than
      // reporting a clean success that quietly did less than asked.
      onError(skipped ? `Deleted ${r.deleted}. ${skipped} still training and kept.` : null)
      clear()
      setBulkOpen(false)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <div className="border-b border-gray-200 px-3 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">Versions</h2>
        <p className="text-xs text-gray-500">This model, this project</p>
      </div>

      <SelectionToolbar
        count={selected.size}
        total={jobs.length}
        onToggleAll={() => toggleAll(jobs.map((j) => j.id))}
        onDelete={() => setBulkOpen(true)}
        busy={busy}
      />

      <ul className="divide-y divide-gray-100">
        {jobs.slice(0, 12).map((j) => (
          <li
            key={j.id}
            onClick={() => onSelect(j.id)}
            className={`flex cursor-pointer gap-2 px-3 py-2 text-xs transition-colors ${
              selectedId === j.id ? 'bg-accent-50' : 'hover:bg-gray-50'
            }`}
          >
            <RowCheckbox
              checked={selected.has(j.id)}
              onChange={() => toggle(j.id)}
              label={`Select ${versionLabel(j)}`}
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-1">
                <span className="flex min-w-0 items-center gap-1.5 font-medium text-gray-800">
                  <span className="truncate">{versionLabel(j)}</span>
                  {j.version === latestVersion && (
                    <span className="shrink-0 rounded bg-accent-100 px-1 py-px text-[9px] font-medium uppercase tracking-wide text-accent-700">
                      latest
                    </span>
                  )}
                  {j.init_from_job_id !== null && (
                    <GitBranch
                      size={10}
                      className="shrink-0 text-gray-400"
                      // Resolved live, so renaming the source updates this too.
                      aria-label={`Continued from ${labelOf(j.init_from_job_id) ?? 'an earlier run'}`}
                    >
                      <title>{`Continued from ${labelOf(j.init_from_job_id) ?? 'an earlier run'}`}</title>
                    </GitBranch>
                  )}
                </span>
                <span className="flex shrink-0 items-center gap-0.5">
                  <RowAction onClick={() => setRenaming(j)} title={`Rename ${versionLabel(j)}`}>
                    <Pencil size={11} />
                  </RowAction>
                  {/* No delete on a run that's still going — the runner is
                      actively writing to that row. */}
                  {!inFlight(j) && (
                    <RowAction onClick={() => setDeleting(j)} title={`Delete ${versionLabel(j)}`} danger>
                      <Trash2 size={11} />
                    </RowAction>
                  )}
                  <StatusBadge status={toStatus(j.status)} />
                </span>
              </div>
              <p className="tabular-nums text-gray-500">
                {j.best_map !== null ? `mAP ${j.best_map.toFixed(3)} · ` : ''}
                {new Date(j.created_at).toLocaleTimeString()}
              </p>
            </div>
          </li>
        ))}
      </ul>

      <RenameDialog
        open={renaming !== null}
        currentName={renaming?.name ?? null}
        fallbackLabel={`v${renaming?.version ?? ''}`}
        onClose={() => setRenaming(null)}
        onSave={async (name) => {
          if (!renaming) return
          await renameTrainingJob(renaming.id, name)
          onChanged()
        }}
      />

      <ConfirmDialog
        open={deleting !== null}
        onClose={() => setDeleting(null)}
        onConfirm={() => void (deleting && removeOne(deleting))}
        title={`Delete ${deleting ? versionLabel(deleting) : ''}?`}
        message={
          `This permanently deletes the trained weights and the dataset copy this ` +
          `run exported.\n\nAny version continued from it keeps its own weights, but ` +
          `loses the link back.`
        }
        confirmLabel="Delete"
        busy={busy}
      />

      <ConfirmDialog
        open={bulkOpen}
        onClose={() => setBulkOpen(false)}
        onConfirm={() => void removeSelected()}
        title={`Delete ${selected.size} version(s)?`}
        message={
          `This permanently deletes the trained weights for ${selected.size} version(s).\n\n` +
          `Any still training are kept.`
        }
        confirmLabel={`Delete ${selected.size}`}
        busy={busy}
      />
    </div>
  )
}
