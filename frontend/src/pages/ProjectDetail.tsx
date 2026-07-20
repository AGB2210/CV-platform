import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Cpu,
  Eye,
  History,
  Plus,
  RotateCcw,
  Save,
  Shuffle,
  SquarePen,
  Tags,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { ConfirmDialog } from '@/components/ui/Modal'
import {
  createClass,
  deleteClass,
  deleteImage,
  getProject,
  listClasses,
  listDatasetVersions,
  listImages,
  restoreDatasetVersion,
  resplitDataset,
  saveDatasetVersion,
  setSplitForImages,
  uploadImages,
  type DatasetImage,
  type DatasetVersion,
  type Project,
  type ProjectClass,
  type Split,
  type UploadResult,
} from '@/lib/api'

export function ProjectDetail() {
  // Route params always arrive as strings — the router can't know the type.
  // Convert once here rather than scattering Number(id) through the file.
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [project, setProject] = useState<Project | null>(null)
  const [images, setImages] = useState<DatasetImage[]>([])
  const [classes, setClasses] = useState<ProjectClass[]>([])
  const [versions, setVersions] = useState<DatasetVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Image selection drives two things: which images auto-annotate runs on, and
  // which images a manual split assignment moves. Both needed a way to say
  // "these ones" and there wasn't one.
  const [selected, setSelected] = useState<Set<number>>(new Set())

  function toggleImage(id: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function selectAll() {
    setSelected((prev) =>
      prev.size === images.length ? new Set() : new Set(images.map((i) => i.id)),
    )
  }

  const refresh = useCallback(async () => {
    try {
      // Promise.all, not three sequential awaits: the requests don't depend on
      // each other, so serialising them would triple the time to first paint
      // for no reason.
      const [p, imgs, cls, vers] = await Promise.all([
        getProject(projectId),
        listImages(projectId),
        listClasses(projectId),
        listDatasetVersions(projectId),
      ])
      setProject(p)
      setImages(imgs)
      setClasses(cls)
      setVersions(vers)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  if (loading) {
    return (
      <>
        <PageHeader title="Loading…" />
        <PageBody>
          <p className="text-sm text-gray-500">Loading project…</p>
        </PageBody>
      </>
    )
  }

  if (!project) {
    return (
      <>
        <PageHeader title="Project not found" />
        <PageBody>
          <p className="text-sm text-gray-600">{error ?? 'This project does not exist.'}</p>
          <Link to="/" className="btn-secondary mt-3">
            <ArrowLeft size={14} />
            Back to projects
          </Link>
        </PageBody>
      </>
    )
  }

  return (
    <>
      <PageHeader
        title={project.name}
        description={project.description ?? 'Object detection'}
        actions={
          <>
            {/* The three things you do TO a dataset, offered at the point you're
                looking at it. "Annotate" is deliberately a peer of
                "Auto-annotate", not a step after it: nobody should have to run a
                model they don't want in order to reach the canvas. */}
            {images.length > 0 && (
              <>
                <Link to={`/projects/${projectId}/visualize`} className="btn-secondary">
                  <Eye size={14} />
                  Visualize
                </Link>
                {/* Carries the selection through in the URL, so Auto-annotate
                    opens already scoped to the images you picked. Previously
                    the only choices were coarse buckets — running the model on
                    six specific images was impossible. */}
                <Link
                  to={
                    selected.size > 0
                      ? `/projects/${projectId}/annotate?images=${[...selected].join(',')}`
                      : `/projects/${projectId}/annotate`
                  }
                  className="btn-secondary"
                >
                  <Tags size={14} />
                  {selected.size > 0
                    ? `Auto-annotate ${selected.size}`
                    : 'Auto-annotate'}
                </Link>
                <Link to={`/projects/${projectId}/review`} className="btn-primary">
                  <SquarePen size={14} />
                  Annotate
                </Link>
              </>
            )}
            <Link to="/" className="btn-secondary">
              <ArrowLeft size={14} />
              All projects
            </Link>
          </>
        }
      />
      <PageBody>
        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            {error}
          </div>
        )}

        {/* Two-column: dataset is the main work surface, classes are a
            reference list you set once and rarely revisit. Sizing follows
            frequency of use, not equal division. */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
          <section>
            <UploadPanel projectId={projectId} onUploaded={refresh} />
            <ImageGrid
              images={images}
              projectId={projectId}
              onDeleted={refresh}
              selected={selected}
              onToggle={toggleImage}
              onSelectAll={selectAll}
              onClearSelection={() => setSelected(new Set())}
            />
            {/* Split lives here, under the images, because it's a property OF
                the images you're looking at — and because with the commit step
                gone there is nowhere else it belongs. */}
            {images.length > 0 && (
              <SplitPanel
                projectId={projectId}
                images={images}
                selected={selected}
                onChanged={refresh}
              />
            )}
          </section>

          <div className="space-y-4">
            <ClassPanel projectId={projectId} classes={classes} onChanged={refresh} />
            <VersionPanel
              projectId={projectId}
              versions={versions}
              hasImages={images.length > 0}
              onChanged={refresh}
              onError={setError}
            />
          </div>
        </div>
      </PageBody>
    </>
  )
}

/**
 * Dataset versions: save points, and the way back from a bad one.
 *
 * "Save dataset" is the only thing that creates a version — one deliberate
 * gesture rather than a version per box drawn — and it's the gate into training,
 * so a run always points at a dataset that still exists exactly as it was.
 *
 * Restore is offered without a red button on purpose: it saves the current state
 * first, so it's consequential but reversible, and colouring it like a delete
 * would teach people to fear the recovery tool.
 */
function VersionPanel({
  projectId,
  versions,
  hasImages,
  onChanged,
  onError,
}: {
  projectId: number
  versions: DatasetVersion[]
  hasImages: boolean
  onChanged: () => void
  onError: (msg: string | null) => void
}) {
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [target, setTarget] = useState<DatasetVersion | null>(null)
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<string | null>(null)

  async function save() {
    setSaving(true)
    onError(null)
    try {
      const v = await saveDatasetVersion(projectId, note.trim() || undefined)
      setNote('')
      setOutcome(`Saved as v${v.version}.`)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function restore() {
    if (!target) return
    setBusy(true)
    onError(null)
    try {
      const r = await restoreDatasetVersion(projectId, target.id)
      // Report what actually happened, including a partial restore — silently
      // presenting an incomplete recovery as a complete one would be the worst
      // possible outcome for a safety feature.
      const bits = [
        `Restored v${r.restored_version}: ${r.images_restored} image(s), ${r.boxes_restored} box(es).`,
        r.images_removed ? `${r.images_removed} later image(s) removed.` : '',
        `Previous state saved as v${r.backup_version}.`,
        r.missing_files.length
          ? `${r.missing_files.length} image(s) could not be recovered — their files are gone.`
          : '',
      ].filter(Boolean)
      setOutcome(bits.join(' '))
      setTarget(null)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
      setTarget(null)
    } finally {
      setBusy(false)
    }
  }

  const latest = versions[0] ?? null

  return (
    <div className="card">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2.5">
        <History size={14} className="text-gray-400" />
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-gray-900">Dataset versions</h2>
          <p className="text-xs text-gray-500">Save points you can return to</p>
        </div>
      </div>

      <div className="space-y-2 border-b border-gray-200 p-3">
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Note (optional)"
          maxLength={255}
          disabled={saving || !hasImages}
          className="w-full rounded-md border border-gray-300 px-2 py-1 text-xs placeholder:text-gray-300 focus:border-accent-500 focus:outline-none disabled:bg-gray-50"
        />
        <button
          className="btn-primary w-full"
          onClick={() => void save()}
          disabled={saving || !hasImages}
        >
          <Save size={13} />
          {saving ? 'Saving…' : 'Save dataset'}
        </button>
        {!hasImages ? (
          <p className="text-xs text-gray-400">Upload images before saving a version.</p>
        ) : !latest ? (
          <p className="text-xs text-amber-700">
            Not saved yet — save to create v1. Training needs a saved version.
          </p>
        ) : (
          <Link to={`/projects/${projectId}/train`} className="btn-secondary w-full">
            <Cpu size={13} />
            Train v{latest.version}
          </Link>
        )}
        {outcome && <p className="text-xs text-gray-600">{outcome}</p>}
      </div>

      {versions.length > 0 && (
        <ul className="max-h-72 divide-y divide-gray-100 overflow-y-auto">
          {versions.map((v) => (
            <li key={v.id} className="px-3 py-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 font-medium text-gray-800">
                  v{v.version}
                  {v.id === latest?.id && (
                    <span className="rounded bg-accent-100 px-1 py-px text-[9px] font-medium uppercase tracking-wide text-accent-700">
                      latest
                    </span>
                  )}
                </span>
                {/* Every version is restorable, INCLUDING the latest. Being the
                    newest save doesn't mean the live dataset still matches it —
                    delete some images and the two diverge immediately, and
                    restoring the latest save is exactly the recovery you want. */}
                <button
                  onClick={() => setTarget(v)}
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-accent-700 hover:bg-accent-50"
                >
                  <RotateCcw size={11} />
                  Restore
                </button>
              </div>
              <p className="tabular-nums text-gray-500">
                {v.total_images} imgs · {v.total_boxes} boxes ·{' '}
                {new Date(v.created_at).toLocaleString()}
              </p>
              {v.note && <p className="truncate text-gray-400">{v.note}</p>}
            </li>
          ))}
        </ul>
      )}

      <ConfirmDialog
        open={target !== null}
        onClose={() => setTarget(null)}
        onConfirm={() => void restore()}
        title={`Restore dataset v${target?.version ?? ''}?`}
        message={
          `The dataset will be reset to v${target?.version ?? ''}: ` +
          `${target?.total_images ?? 0} image(s) and ${target?.total_boxes ?? 0} box(es).\n\n` +
          `Images added since then are removed, and images deleted since then come back.\n\n` +
          `Your current dataset is saved as a new version first, so you can undo this.`
        }
        confirmLabel="Restore"
        busy={busy}
        destructive={false}
      />
    </div>
  )
}

/** Drag-and-drop + click-to-browse uploader. */
function UploadPanel({
  projectId,
  onUploaded,
}: {
  projectId: number
  onUploaded: () => void
}) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  // Keep the whole UploadResult, not just a count. A zip that turned out to be
  // an annotated COCO dataset did far more than "upload 1,200 images", and
  // reporting it as a plain upload hides the interesting half.
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function send(files: File[]) {
    if (!files.length) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await uploadImages(projectId, files))
      onUploaded()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
      // Clear the input so re-picking the SAME file fires onChange again.
      // Without this, <input type=file> sees no value change and stays silent —
      // a classic "upload works once then stops" bug.
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div className="mb-4">
      <div
        // preventDefault on dragOver is mandatory. The browser's default is to
        // REFUSE the drop; without this the onDrop handler never fires at all.
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void send(Array.from(e.dataTransfer.files))
        }}
        onClick={() => inputRef.current?.click()}
        className={[
          'flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-6 transition-colors',
          dragging
            ? 'border-accent-500 bg-accent-50'
            : 'border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50',
        ].join(' ')}
      >
        <Upload size={18} className="mb-1.5 text-gray-400" />
        <p className="text-sm text-gray-700">
          {busy ? 'Uploading…' : 'Drop images here, or click to browse'}
        </p>
        <p className="mt-0.5 text-xs text-gray-400">
          JPG, PNG, BMP, WEBP — or a .zip
        </p>
        {/* Say that dataset import exists. It's the kind of feature nobody
            discovers by guessing, and "just drop the whole export" is a much
            better first experience than manually recreating classes. */}
        <p className="mt-1 text-xs text-gray-400">
          A COCO export or Roboflow zip (train/valid/test) imports its annotations
          and splits automatically.
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.bmp,.webp,.zip"
          className="hidden"
          onChange={(e) => void send(Array.from(e.target.files ?? []))}
        />
      </div>

      {error && (
        <p className="mt-2 rounded-md border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs text-red-800">
          {error}
        </p>
      )}

      {/* Report skipped files explicitly. The backend allows partial success,
          so silently showing "12 uploaded" when 3 were rejected would hide a
          real problem with the user's data. */}
      {result && (
        <div className="mt-2 rounded-md border border-gray-200 bg-gray-50 px-2.5 py-1.5 text-xs">
          <span className="font-medium text-gray-800">{result.uploaded_count} uploaded</span>

          {/* An import did more than copy files — say what. */}
          {result.annotations_imported > 0 && (
            <span className="text-gray-600">
              {' · '}
              <span className="font-medium text-gray-800">
                {result.annotations_imported} annotations imported
              </span>
            </span>
          )}
          {result.classes_created.length > 0 && (
            <span className="text-gray-600">
              {' · '}
              {result.classes_created.length} class
              {result.classes_created.length === 1 ? '' : 'es'} created (
              {result.classes_created.slice(0, 4).join(', ')}
              {result.classes_created.length > 4 ? '…' : ''})
            </span>
          )}
          {result.has_split_folders && Object.keys(result.splits).length > 0 && (
            <span className="text-gray-600">
              {' · splits '}
              {(['train', 'val', 'test'] as const)
                .filter((s) => result.splits[s])
                .map((s) => `${s} ${result.splits[s]}`)
                .join(' / ')}
            </span>
          )}

          {/* Imported train data with no validation set. Training without one
              produces a model that looks flawless and generalises like a rock,
              and nothing else would ever tell you. */}
          {result.needs_val_split && (
            <p className="mt-1.5 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-900">
              <span className="font-medium">No validation set in this dataset.</span> Use
              the split control before training, or you'll have no way to detect
              overfitting.
            </p>
          )}

          {result.skipped.length > 0 && (
            <>
              <span className="text-gray-500"> · {result.skipped.length} skipped</span>
              <ul className="mt-1 space-y-0.5 text-gray-500">
                {result.skipped.slice(0, 5).map((s) => (
                  <li key={s} className="truncate">
                    {s}
                  </li>
                ))}
                {result.skipped.length > 5 && <li>…and {result.skipped.length - 5} more</li>}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Train/val/test split, under the grid.
 *
 * It lives here because a split is a property OF these images — and because
 * with the staging->dataset commit gone, the dialog that used to ask for
 * percentages went with it. Setting it next to the thing it describes beats
 * hiding it behind a modal you only meet at commit time.
 *
 * Two ways to set it, because there are genuinely two situations:
 *   - by percentage, shuffled — the normal case, and reproducible (fixed seed)
 *   - by selection — for when you KNOW these twelve images belong in val, and
 *     a random shuffle is exactly the wrong tool
 */
function SplitPanel({
  projectId,
  images,
  selected,
  onChanged,
}: {
  projectId: number
  images: DatasetImage[]
  selected: Set<number>
  onChanged: () => void
}) {
  const [trainPct, setTrainPct] = useState(80)
  const [valPct, setValPct] = useState(20)
  const [testPct, setTestPct] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const counts = useMemo(() => {
    const c = { train: 0, val: 0, test: 0 } as Record<string, number>
    for (const i of images) c[i.split] = (c[i.split] ?? 0) + 1
    return c
  }, [images])

  const total = trainPct + valPct + testPct
  const pctValid = total === 100

  async function applyPercentages() {
    setBusy(true)
    setError(null)
    try {
      await resplitDataset(projectId, {
        train_pct: trainPct / 100,
        val_pct: valPct / 100,
        test_pct: testPct / 100,
      })
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function assignSelected(split: Split) {
    setBusy(true)
    setError(null)
    try {
      await setSplitForImages(projectId, [...selected], split)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card mt-4">
      <div className="flex items-baseline justify-between border-b border-gray-200 px-4 py-3">
        <div>
          <h2 className="text-sm font-medium text-gray-900">Train / val / test split</h2>
          <p className="text-xs text-gray-500">
            Which images the trainer learns from, tunes on, and is scored against.
          </p>
        </div>
        <div className="flex gap-3 text-xs tabular-nums">
          <SplitStat label="train" value={counts.train ?? 0} className="text-gray-700" />
          <SplitStat label="val" value={counts.val ?? 0} className="text-accent-700" />
          <SplitStat label="test" value={counts.test ?? 0} className="text-amber-700" />
        </div>
      </div>

      <div className="space-y-3 p-4">
        {/* No validation set = no way to detect overfitting, and nothing else
            in the app would ever tell you. */}
        {(counts.val ?? 0) === 0 && images.length > 1 && (
          <p className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">
            <span className="font-medium">No validation set.</span> Without one you'll
            have no way to tell whether the model is learning or memorising.
          </p>
        )}

        {selected.size > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-600">
              Move <span className="font-medium">{selected.size} selected</span> to:
            </span>
            {(['train', 'val', 'test'] as const).map((s) => (
              <button
                key={s}
                className="btn-secondary"
                disabled={busy}
                onClick={() => void assignSelected(s)}
              >
                {s}
              </button>
            ))}
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-end gap-2">
              <PctField label="Train" value={trainPct} onChange={setTrainPct} />
              <PctField label="Val" value={valPct} onChange={setValPct} />
              <PctField label="Test" value={testPct} onChange={setTestPct} />
              <button
                className="btn-primary"
                onClick={() => void applyPercentages()}
                disabled={busy || !pctValid || images.length === 0}
              >
                <Shuffle size={13} />
                {busy ? 'Splitting…' : 'Apply split'}
              </button>
            </div>
            {!pctValid && (
              <p className="text-xs text-red-600">Must sum to 100% — currently {total}%.</p>
            )}
            <p className="text-xs text-gray-400">
              Shuffled with a fixed seed, so the same dataset splits the same way every
              time. Select images above to assign them manually instead.
            </p>
          </>
        )}

        {error && (
          <p className="rounded-md border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs text-red-800">
            {error}
          </p>
        )}
      </div>
    </div>
  )
}

function SplitStat({
  label,
  value,
  className,
}: {
  label: string
  value: number
  className: string
}) {
  return (
    <span className={className}>
      <span className="font-medium">{value}</span>{' '}
      <span className="text-gray-400">{label}</span>
    </span>
  )
}

function PctField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block">
      <span className="mb-0.5 block text-xs text-gray-500">{label}</span>
      <div className="flex w-20 items-center rounded-md border border-gray-300 focus-within:border-accent-500">
        <input
          type="number"
          min={0}
          max={100}
          value={value}
          onChange={(e) => onChange(Math.max(0, Math.min(100, Number(e.target.value))))}
          className="w-full rounded-md px-2 py-1 text-sm tabular-nums focus:outline-none"
        />
        <span className="pr-2 text-xs text-gray-400">%</span>
      </div>
    </label>
  )
}

function ImageGrid({
  images,
  projectId,
  onDeleted,
  selected,
  onToggle,
  onSelectAll,
  onClearSelection,
}: {
  images: DatasetImage[]
  projectId: number
  onDeleted: () => void
  selected: Set<number>
  onToggle: (id: number) => void
  onSelectAll: () => void
  onClearSelection: () => void
}) {
  const [pending, setPending] = useState<DatasetImage | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleDelete() {
    if (!pending) return
    setBusy(true)
    try {
      await deleteImage(pending.id)
      setPending(null)
      onDeleted()
    } finally {
      setBusy(false)
    }
  }

  if (images.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-400">
        No images yet. Upload some to get started.
      </p>
    )
  }

  return (
    <>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-medium text-gray-900">Images</h2>
        <div className="flex items-center gap-3 text-xs">
          {selected.size > 0 && (
            <>
              <span className="font-medium tabular-nums text-accent-800">
                {selected.size} selected
              </span>
              <button className="text-gray-500 underline" onClick={onClearSelection}>
                Clear
              </button>
            </>
          )}
          <button className="text-gray-500 underline" onClick={onSelectAll}>
            {selected.size === images.length ? 'Deselect all' : 'Select all'}
          </button>
          <span className="tabular-nums text-gray-500">{images.length} total</span>
        </div>
      </div>

      {/* Dense auto-fill grid: as many ~130px columns as fit. Deliberately
          tighter than a typical gallery — the job here is to scan a dataset,
          not admire photos. */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(130px,1fr))] gap-2">
        {images.map((img) => (
          <div
            key={img.id}
            className={[
              'group relative overflow-hidden rounded border bg-gray-100',
              selected.has(img.id)
                ? 'border-accent-500 ring-2 ring-accent-500'
                : 'border-gray-200',
            ].join(' ')}
          >
            {/* The tile links into review. The whole point of the grid is to
                get you to the image you want to look at. */}
            <Link to={`/projects/${projectId}/review/${img.id}`} className="block">
              {/* aspect-square + object-cover: uniform tiles regardless of source
                  aspect ratio, so the grid stays a grid. */}
              <img
                src={img.url}
                alt={img.original_filename}
                // Native lazy loading — a 5,000-image dataset must not issue
                // 5,000 requests on mount.
                loading="lazy"
                className={`aspect-square w-full object-cover ${
                  selected.has(img.id) ? 'opacity-80' : ''
                }`}
              />
            </Link>

            {/* Checkbox sits ON the tile, always visible once anything is
                selected — a hover-only checkbox makes multi-select a hunt.
                Its own element, not the tile, so clicking the picture still
                opens it: selecting and opening are different intents. */}
            <label
              className={[
                'absolute bottom-1 left-1 flex h-5 w-5 cursor-pointer items-center justify-center rounded bg-white/90 shadow-sm transition-opacity',
                selected.size > 0 || selected.has(img.id)
                  ? 'opacity-100'
                  : 'opacity-0 focus-within:opacity-100 group-hover:opacity-100',
              ].join(' ')}
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={selected.has(img.id)}
                onChange={() => onToggle(img.id)}
                className="accent-accent-600"
                aria-label={`Select ${img.original_filename}`}
              />
            </label>

            {/* Split chip — every image has one now that staging is gone. */}
            <span
              className={[
                'absolute bottom-1 right-1 rounded px-1 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white',
                img.split === 'val'
                  ? 'bg-accent-600'
                  : img.split === 'test'
                    ? 'bg-amber-600'
                    : 'bg-gray-700',
              ].join(' ')}
            >
              {img.split}
            </span>

            {/* Annotation state, visible without opening the image. A dataset is
                mostly "which of these still needs work", and answering that from
                the grid saves opening 500 images to find out. */}
            {(img.annotation_count > 0 || img.proposed_count > 0) && (
              <span
                className={[
                  'absolute left-1 top-1 rounded px-1 py-0.5 text-[10px] font-medium tabular-nums shadow-sm',
                  // Amber when the model is waiting on you. Green means done.
                  img.proposed_count > 0
                    ? 'bg-accent-600 text-white'
                    : 'bg-green-700 text-white',
                ].join(' ')}
                title={
                  img.proposed_count > 0
                    ? `${img.annotation_count} box(es) · ${img.proposed_count} model proposal(s) awaiting review`
                    : `${img.annotation_count} box(es)`
                }
              >
                {img.proposed_count > 0
                  ? `${img.annotation_count}+${img.proposed_count}`
                  : img.annotation_count}
              </span>
            )}

            <button
              onClick={() => setPending(img)}
              className="absolute right-1 top-1 rounded bg-white/90 p-1 text-gray-500 opacity-0 shadow-sm transition-opacity hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
              aria-label={`Delete ${img.original_filename}`}
            >
              <X size={12} />
            </button>

            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-1.5 py-1">
              <p className="truncate text-[10px] text-white" title={img.original_filename}>
                {img.original_filename}
              </p>
              <p className="text-[10px] tabular-nums text-white/70">
                {img.width}×{img.height}
              </p>
            </div>
          </div>
        ))}
      </div>

      <ConfirmDialog
        open={pending !== null}
        onClose={() => setPending(null)}
        onConfirm={handleDelete}
        busy={busy}
        title="Delete image"
        message={`Delete "${pending?.original_filename}"? This removes the file from disk and cannot be undone.`}
      />
    </>
  )
}

function ClassPanel({
  projectId,
  classes,
  onChanged,
}: {
  projectId: number
  classes: ProjectClass[]
  onChanged: () => void
}) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function add(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    setBusy(true)
    setError(null)
    try {
      await createClass(projectId, name.trim())
      setName('')
      onChanged()
    } catch (err) {
      // Surfaces the backend's 409 as a readable message rather than a console
      // error — duplicate class names are a normal thing to try.
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function remove(classId: number) {
    await deleteClass(classId)
    onChanged()
  }

  return (
    <aside className="card h-fit">
      <div className="border-b border-gray-200 px-3 py-2.5">
        <h2 className="text-sm font-medium text-gray-900">Classes</h2>
        <p className="text-xs text-gray-500">Object types to detect</p>
      </div>

      <form onSubmit={add} className="flex gap-1.5 border-b border-gray-200 p-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="car"
          className="min-w-0 flex-1 rounded-md border border-gray-300 px-2 py-1 text-sm placeholder:text-gray-400 focus:border-accent-500 focus:outline-none"
        />
        <button type="submit" className="btn-primary shrink-0 px-2" disabled={busy || !name.trim()}>
          <Plus size={14} />
        </button>
      </form>

      {error && (
        <p className="border-b border-gray-200 bg-red-50 px-3 py-1.5 text-xs text-red-800">
          {error}
        </p>
      )}

      {classes.length === 0 ? (
        <p className="px-3 py-4 text-xs text-gray-400">
          No classes yet. Add the object types you want to detect, e.g. “car”, “person”.
        </p>
      ) : (
        <ul className="divide-y divide-gray-100">
          {classes.map((c) => (
            <li key={c.id} className="group flex items-center gap-2 px-3 py-2">
              {/* The class colour, shown where the class is defined — so the
                  swatch you see here is the box colour you'll see on the
                  annotation canvas in Phase 3. */}
              <span
                className="h-3 w-3 shrink-0 rounded-sm border border-black/10"
                style={{ backgroundColor: c.color }}
              />
              <span className="flex-1 truncate text-sm text-gray-800">{c.name}</span>
              <button
                onClick={() => void remove(c.id)}
                className="rounded p-0.5 text-gray-400 opacity-0 hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
                aria-label={`Delete class ${c.name}`}
              >
                <Trash2 size={13} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}
