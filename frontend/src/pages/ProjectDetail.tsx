import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Cpu,
  Eye,
  History,
  Pencil,
  Plus,
  RotateCcw,
  Save,
  Shuffle,
  SquarePen,
  Tags,
  ChevronLeft,
  ChevronRight,
  HardDrive,
  Trash2,
  Undo2,
  TriangleAlert,
  FileUp,
  FolderUp,
  Upload,
  X,
} from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { ConfirmDialog } from '@/components/ui/Modal'
import {
  RenameDialog,
  RowAction,
  RowCheckbox,
  SelectionToolbar,
} from '@/components/VersionAdmin'
import { useVersionSelection } from '@/lib/useVersionSelection'
import {
  bulkDeleteDatasetVersions,
  createClass,
  deleteClass,
  deleteDatasetVersion,
  bulkDeleteImages,
  deleteImage,
  getProject,
  listClasses,
  listDatasetVersions,
  listImagePage,
  renameDatasetVersion,
  restoreDatasetVersion,
  resplitDataset,
  saveDatasetVersion,
  setSplitForImages,
  discardUnsavedImages,
  getStorageReport,
  reclaimStorage,
  undoImport,
  uploadImages,
  type UploadProgress,
  versionLabel,
  type DatasetImage,
  type DatasetVersion,
  type Project,
  type ProjectClass,
  type Split,
  type StorageReport,
  type UploadResult,
} from '@/lib/api'

/** Images per page in the grid. Large enough that most projects are one page,
 *  small enough that a page of thumbnails stays quick to render and scroll. */
const PAGE_SIZE = 200

export function ProjectDetail() {
  // Route params always arrive as strings — the router can't know the type.
  // Convert once here rather than scattering Number(id) through the file.
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [project, setProject] = useState<Project | null>(null)
  const [images, setImages] = useState<DatasetImage[]>([])
  // Server-side paging. A dataset reaches thousands of images, and the grid
  // used to render whatever the default page happened to be — 200 rows — as if
  // that were the whole project.
  const [page, setPage] = useState(0)
  const [totalImages, setTotalImages] = useState(0)
  // Bumped on every refresh so the storage figures re-read after an upload,
  // a delete or a save rather than showing what was true on page load.
  const [storageKey, setStorageKey] = useState(0)
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

  // Scoped to the page on screen. "Select all" cannot honestly mean 5,000
  // images the browser has never loaded — and a delete built on that would act
  // on rows the user never saw.
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
        listImagePage(projectId, PAGE_SIZE, page * PAGE_SIZE),
        listClasses(projectId),
        listDatasetVersions(projectId),
      ])
      setProject(p)
      setImages(imgs.images)
      setTotalImages(imgs.total)
      setClasses(cls)
      setVersions(vers)
      setStorageKey((k) => k + 1)
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [projectId, page])

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
              // Deleting an image keeps its bytes once the project has any
              // saved version, so a restore can bring the row back. The
              // confirm dialogs must say which of the two is happening.
              hasVersions={versions.length > 0}
              page={page}
              pageSize={PAGE_SIZE}
              total={totalImages}
              onPageChange={(p) => {
                setPage(p)
                // Selection is per-page, so carrying it across would leave
                // "12 selected" pointing at rows no longer on screen.
                setSelected(new Set())
              }}
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
            {/* Last: housekeeping is something you check occasionally, not a
                thing to put above the work. */}
            <StoragePanel
              projectId={projectId}
              refreshKey={storageKey}
              onChanged={refresh}
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
  const [renaming, setRenaming] = useState<DatasetVersion | null>(null)
  const [deleting, setDeleting] = useState<DatasetVersion | null>(null)
  const [bulkOpen, setBulkOpen] = useState(false)
  const { selected, toggle, toggleAll, clear } = useVersionSelection()

  // No version matching the live dataset means there is unsaved work on screen.
  // Restore no longer auto-saves a backup, so that work is about to be lost —
  // this is the whole reason the confirm dialog changes shape below.
  const hasUnsavedChanges = versions.length > 0 && !versions.some((v) => v.is_current)

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
        r.classes_removed.length
          ? `Removed class(es) added since: ${r.classes_removed.join(', ')}.`
          : '',
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

  async function removeOne(v: DatasetVersion) {
    setBusy(true)
    onError(null)
    try {
      await deleteDatasetVersion(projectId, v.id)
      setOutcome(`Deleted ${versionLabel(v)}.`)
      setDeleting(null)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function removeSelected() {
    setBusy(true)
    onError(null)
    try {
      const r = await bulkDeleteDatasetVersions(projectId, [...selected])
      setOutcome(`Deleted ${r.deleted} version(s).`)
      clear()
      setBulkOpen(false)
      onChanged()
    } catch (e) {
      onError((e as Error).message)
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
        <>
          <SelectionToolbar
            count={selected.size}
            total={versions.length}
            onToggleAll={() => toggleAll(versions.map((v) => v.id))}
            onDelete={() => setBulkOpen(true)}
            busy={busy}
          />
          <ul className="max-h-72 divide-y divide-gray-100 overflow-y-auto">
            {versions.map((v) => (
              <li key={v.id} className="flex gap-2 px-3 py-2 text-xs">
                <RowCheckbox
                  checked={selected.has(v.id)}
                  onChange={() => toggle(v.id)}
                  label={`Select ${versionLabel(v)}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1">
                    <span className="flex min-w-0 items-center gap-1.5 font-medium text-gray-800">
                      <span className="truncate">{versionLabel(v)}</span>
                      {/* "current" = what the dataset on screen actually is;
                          "latest" = merely the most recent save. They're the
                          same until you restore an older version, and after
                          that the difference is the whole point. */}
                      {v.is_current && (
                        <span className="shrink-0 rounded bg-accent-100 px-1 py-px text-[9px] font-medium uppercase tracking-wide text-accent-700">
                          current
                        </span>
                      )}
                      {v.id === latest?.id && !v.is_current && (
                        <span className="shrink-0 rounded bg-gray-100 px-1 py-px text-[9px] font-medium uppercase tracking-wide text-gray-500">
                          latest
                        </span>
                      )}
                    </span>
                    <span className="flex shrink-0 items-center gap-0.5">
                      <RowAction onClick={() => setRenaming(v)} title={`Rename ${versionLabel(v)}`}>
                        <Pencil size={11} />
                      </RowAction>
                      <RowAction onClick={() => setDeleting(v)} title={`Delete ${versionLabel(v)}`} danger>
                        <Trash2 size={11} />
                      </RowAction>
                    </span>
                  </div>
                  {/* Every version is restorable, INCLUDING the latest. Being the
                      newest save doesn't mean the live dataset still matches it —
                      delete some images and the two diverge immediately, and
                      restoring the latest save is exactly the recovery you want. */}
                  <button
                    onClick={() => setTarget(v)}
                    className="flex items-center gap-1 rounded text-[11px] text-accent-700 hover:underline"
                  >
                    <RotateCcw size={10} />
                    Restore
                  </button>
                  <p className="tabular-nums text-gray-500">
                    {v.total_images} imgs · {v.total_boxes} boxes ·{' '}
                    {new Date(v.created_at).toLocaleString()}
                  </p>
                  {v.note && <p className="truncate text-gray-400">{v.note}</p>}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      <RenameDialog
        open={renaming !== null}
        currentName={renaming?.name ?? null}
        fallbackLabel={`v${renaming?.version ?? ''}`}
        onClose={() => setRenaming(null)}
        onSave={async (name) => {
          if (!renaming) return
          await renameDatasetVersion(projectId, renaming.id, name)
          onChanged()
        }}
      />

      <ConfirmDialog
        open={deleting !== null}
        onClose={() => setDeleting(null)}
        onConfirm={() => void (deleting && removeOne(deleting))}
        title={`Delete ${deleting ? versionLabel(deleting) : ''}?`}
        message={
          `This permanently deletes the saved snapshot, so this dataset state can no ` +
          `longer be restored or trained.\n\n` +
          `Your images and their annotations are NOT affected — only this save point.`
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
          `This permanently deletes ${selected.size} saved snapshot(s). Those dataset ` +
          `states can no longer be restored or trained.\n\n` +
          `Your images and their annotations are NOT affected.`
        }
        confirmLabel={`Delete ${selected.size}`}
        busy={busy}
      />

      <ConfirmDialog
        open={target !== null}
        onClose={() => setTarget(null)}
        onConfirm={() => void restore()}
        title={`Restore dataset v${target?.version ?? ''}?`}
        message={
          `The dataset will be reset to v${target?.version ?? ''}: ` +
          `${target?.total_images ?? 0} image(s) and ${target?.total_boxes ?? 0} box(es).\n\n` +
          `Images added since then are removed, images deleted since then come back, ` +
          `and classes added since then are removed.\n\n` +
          // The consequential half. Nothing is auto-saved any more, so restoring
          // over unsaved work destroys it — say so plainly, and only when it's
          // actually true, so the warning keeps its meaning.
          (hasUnsavedChanges
            ? `The dataset has UNSAVED CHANGES that are not in any version. ` +
              `Restoring discards them permanently. Save the dataset first if you want to keep them.`
            : `The current state is already saved as a version, so you can get back to it.`)
        }
        confirmLabel="Restore"
        busy={busy}
        // Red only when something genuinely unrecoverable is at stake. A restore
        // over saved work is still the recovery tool it has always been.
        destructive={hasUnsavedChanges}
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
  // A folder upload is sent in batches, so it takes long enough that silence
  // reads as a hang. See planUploadBatches for why it's batched at all.
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)

  async function undo(importId: string) {
    setBusy(true)
    setError(null)
    try {
      const r = await undoImport(projectId, importId)
      setResult(null)
      // Images a version has since captured are kept, and staying quiet about
      // that would leave the count looking wrong.
      setNotice(
        `Removed ${r.deleted} image(s), freeing ${(r.bytes_freed / 1048576).toFixed(1)} MB.` +
          (r.kept_in_versions
            ? ` ${r.kept_in_versions} kept — a saved version depends on them.`
            : ''),
      )
      onUploaded()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function send(files: File[]) {
    if (!files.length) return
    setBusy(true)
    setError(null)
    setResult(null)
    setProgress(null)
    try {
      setResult(await uploadImages(projectId, files, setProgress))
      onUploaded()
    } catch (err) {
      // The whole message, newlines and all. A big upload fails for specific
      // reasons — a rejected file, an unreachable backend — and truncating that
      // to one line is what made "Failed to fetch" the only thing anyone saw.
      setError((err as Error).message)
    } finally {
      setProgress(null)
      setBusy(false)
      // Clear the inputs so re-picking the SAME file fires onChange again.
      // Without this, <input type=file> sees no value change and stays silent —
      // a classic "upload works once then stops" bug.
      if (inputRef.current) inputRef.current.value = ''
      if (folderRef.current) folderRef.current.value = ''
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
        className={[
          'flex flex-col items-center justify-center rounded-lg border border-dashed px-4 py-6 transition-colors',
          dragging
            ? 'border-accent-500 bg-accent-50'
            : 'border-gray-300 bg-white',
        ].join(' ')}
      >
        <Upload size={18} className="mb-1.5 text-gray-400" />
        <p className="text-sm text-gray-700">
          {progress
            ? `Uploading ${progress.filesSent.toLocaleString()} of ` +
              `${progress.filesTotal.toLocaleString()}…` +
              (progress.batches > 1 ? ` (batch ${progress.batch}/${progress.batches})` : '')
            : busy
              ? 'Uploading…'
              : 'Drop images, a folder, or a .zip here'}
        </p>
        {progress && progress.filesTotal > 0 && (
          <div className="mt-1.5 h-1 w-56 overflow-hidden rounded-full bg-gray-200">
            <div
              className="h-full bg-accent-600 transition-[width]"
              style={{
                width: `${Math.round((progress.filesSent / progress.filesTotal) * 100)}%`,
              }}
            />
          </div>
        )}

        {/* Two explicit buttons rather than one click target on the whole box.
            <input type="file"> and the same input with `webkitdirectory` are
            different pickers — one cannot choose a folder and the other cannot
            choose loose files — so the choice has to be made before the dialog
            opens, and only the user can make it. */}
        <div className="mt-2.5 flex items-center gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            <FileUp size={11} />
            Select files
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => folderRef.current?.click()}
            className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            <FolderUp size={11} />
            Select folder
          </button>
        </div>

        <p className="mt-2 text-xs text-gray-400">
          JPG, PNG, BMP, WEBP — plus COCO .json, YOLO .txt or data.yaml
        </p>
        {/* Say that dataset import exists. It's the kind of feature nobody
            discovers by guessing, and "just point at the whole export" is a
            much better first experience than manually recreating classes. */}
        <p className="mt-1 max-w-md text-center text-xs text-gray-400">
          COCO or YOLO is detected automatically. A folder with train/val/test
          subfolders keeps those splits; anything else goes to train, and you can
          split it below.
        </p>

        <input
          ref={inputRef}
          type="file"
          multiple
          // Annotation files are accepted alongside images: picking your
          // pictures AND _annotations.coco.json in one go used to upload the
          // images and silently discard every label.
          accept=".jpg,.jpeg,.png,.bmp,.webp,.zip,.json,.txt,.yaml,.yml"
          className="hidden"
          onChange={(e) => void send(Array.from(e.target.files ?? []))}
        />
        <input
          ref={folderRef}
          type="file"
          multiple
          // Non-standard but supported everywhere that matters. React doesn't
          // know these attributes, hence the cast — they must reach the DOM
          // verbatim or the picker stays a file picker.
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
          className="hidden"
          onChange={(e) => void send(Array.from(e.target.files ?? []))}
        />
      </div>

      {/* whitespace-pre-line: upload failures are multi-line on purpose — a
          rejected batch lists every reason, and a connection failure lists the
          causes worth checking. Collapsing that to one line is what left
          "Failed to fetch" as the entire diagnosis. */}
      {error && (
        <div className="mt-2 rounded-md border border-red-200 bg-red-50 px-2.5 py-2 text-xs text-red-800">
          <p className="mb-1 flex items-center gap-1 font-medium">
            <TriangleAlert size={11} />
            Upload failed
          </p>
          <p className="max-h-48 overflow-auto whitespace-pre-line font-mono text-[11px] leading-relaxed">
            {error}
          </p>
        </div>
      )}

      {notice && (
        <p className="mt-2 rounded-md border border-gray-200 bg-gray-50 px-2.5 py-1.5 text-xs text-gray-700">
          {notice}
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
          {/* Annotations for images already here. They can't just overwrite
              existing work, so they queue as proposals and go through the same
              review as an auto-annotate run. Without this line the upload would
              look like it did nothing. */}
          {result.proposals_created > 0 && (
            <p className="mt-1.5 rounded border border-accent-200 bg-accent-50 px-2 py-1 text-accent-900">
              <span className="font-medium">
                {result.proposals_created} box(es) proposed on {result.reannotated_images}{' '}
                image(s) already in this project.
              </span>{' '}
              Their existing boxes are untouched — open{' '}
              <Link to={`/projects/${projectId}/review`} className="underline">
                Annotate
              </Link>{' '}
              to accept or reject the new ones.
            </p>
          )}

          {/* Re-uploading a folder used to double the dataset silently. Now it
              adds nothing and says so, which is the only way to tell a
              successful no-op from a broken one. */}
          {result.duplicates_skipped > 0 && (
            <span className="text-gray-600">
              {' · '}
              <span className="font-medium text-gray-800">
                {result.duplicates_skipped} already in this project
              </span>
              {' (skipped)'}
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

          {/* EVERY rejection, in a scroll box — not the first five.
              With a folder partly rejected, the one line explaining why is
              rarely among the first few, and "…and 812 more" is precisely the
              information you needed. Scrolling keeps it from taking the page
              over, and not truncating keeps it useful. */}
          {/* Undo. Offered on EVERY import, not just a failed one — a 27-batch
              folder that succeeded is just as likely to be the wrong folder,
              and picking 5,000 images out of the grid by hand is not a
              recovery path. */}
          {result.import_id && result.uploaded_count > 0 && (
            <button
              onClick={() => void undo(result.import_id!)}
              disabled={busy}
              className="mt-1.5 flex items-center gap-1 text-gray-600 underline hover:text-red-700 disabled:opacity-50"
            >
              <Undo2 size={11} />
              Undo this import ({result.uploaded_count} image
              {result.uploaded_count === 1 ? '' : 's'})
            </button>
          )}

          {result.skipped.length > 0 && (
            <details className="mt-1.5">
              <summary className="cursor-pointer text-gray-600">
                <span className="font-medium">{result.skipped.length} skipped</span> — show
                reasons
              </summary>
              <ul className="mt-1 max-h-48 space-y-0.5 overflow-auto rounded border border-gray-200 bg-white p-1.5 font-mono text-[11px] text-gray-600">
                {result.skipped.map((s, i) => (
                  <li key={`${s}-${i}`}>{s}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

const mb = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MB`

/**
 * Disk usage, and the two things that can be freed.
 *
 * The two are NOT the same and the panel keeps them apart, because conflating
 * them is how a cleanup feature deletes something a restore needed:
 *
 *   Unsaved images  real dataset content, just not captured in a save point.
 *                   Deleting is the USER's call — the whole upload -> annotate
 *                   -> save workflow lives in this state, so "unsaved" does not
 *                   mean "unwanted".
 *   Orphaned files  bytes nothing in the app can reach. Pure waste.
 *
 * Retained files are shown but have no action: they have no live row and look
 * like waste, but a version depends on them and removing them would break the
 * restore they exist for.
 */
function StoragePanel({
  projectId,
  refreshKey,
  onChanged,
}: {
  projectId: number
  refreshKey: number
  onChanged: () => void
}) {
  const [report, setReport] = useState<StorageReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [confirmDiscard, setConfirmDiscard] = useState(false)
  const [outcome, setOutcome] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setReport(await getStorageReport(projectId))
    } catch {
      // Housekeeping figures are not worth breaking the page over; the panel
      // simply doesn't render.
      setReport(null)
    }
  }, [projectId])

  // refreshKey changes whenever the dataset does, so the figures can't go stale
  // behind an upload or a delete.
  useEffect(() => {
    void load()
  }, [load, refreshKey])

  async function run(action: () => Promise<string>) {
    setBusy(true)
    setError(null)
    setOutcome(null)
    try {
      setOutcome(await action())
      await load()
      onChanged()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
      setConfirmDiscard(false)
    }
  }

  if (!report) return null

  const nothingToDo = report.unsaved_images === 0 && report.orphan_files === 0

  return (
    <div className="card">
      <div className="border-b border-gray-200 px-3 py-2.5">
        <h2 className="flex items-center gap-1.5 text-sm font-medium text-gray-900">
          <HardDrive size={13} />
          Storage
        </h2>
        <p className="text-xs text-gray-500">What this project keeps on disk</p>
      </div>

      <div className="space-y-2 px-3 py-2.5 text-xs">
        {report.unsaved_images > 0 && (
          <div>
            <p className="text-gray-700">
              <span className="font-medium tabular-nums">{report.unsaved_images}</span>{' '}
              image{report.unsaved_images === 1 ? '' : 's'} not in any saved version.
            </p>
            <p className="mt-0.5 text-gray-500">
              Still part of the dataset — save a version to keep them.
            </p>
            <button
              onClick={() => setConfirmDiscard(true)}
              disabled={busy}
              className="mt-1 flex items-center gap-1 text-red-700 underline hover:text-red-800 disabled:opacity-50"
            >
              <Trash2 size={11} />
              Discard them
            </button>
          </div>
        )}

        {report.orphan_files > 0 && (
          <div className={report.unsaved_images > 0 ? 'border-t border-gray-100 pt-2' : ''}>
            <p className="text-gray-700">
              <span className="font-medium tabular-nums">{report.orphan_files}</span>{' '}
              unreachable file{report.orphan_files === 1 ? '' : 's'} ·{' '}
              {mb(report.orphan_bytes)}
            </p>
            <p className="mt-0.5 text-gray-500">
              Left by images deleted after every version holding them was removed.
              Nothing can reach these.
            </p>
            <button
              onClick={() =>
                void run(async () => {
                  const r = await reclaimStorage(projectId)
                  return `Reclaimed ${r.files_removed} file(s), ${mb(r.bytes_freed)}.`
                })
              }
              disabled={busy}
              // Not destructive-red: nothing recoverable is at stake, and
              // colouring it like a delete would make routine cleanup feel
              // dangerous.
              className="mt-1 text-accent-700 underline hover:text-accent-800 disabled:opacity-50"
            >
              Reclaim {mb(report.orphan_bytes)}
            </button>
          </div>
        )}

        {report.retained_files > 0 && (
          <p className="border-t border-gray-100 pt-2 text-gray-500">
            <span className="tabular-nums">{report.retained_files}</span> file
            {report.retained_files === 1 ? '' : 's'} ({mb(report.retained_bytes)}) kept for
            saved versions — deleted from the dataset, but a version can restore them.
          </p>
        )}

        {report.unreadable_versions.length > 0 && (
          <p className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-900">
            {report.unreadable_versions.join(', ')} could not be read, so nothing will be
            reclaimed until that is resolved — the files those versions need are unknown.
          </p>
        )}

        {nothingToDo && <p className="text-gray-500">Nothing to clean up.</p>}
        {outcome && <p className="text-gray-600">{outcome}</p>}
        {error && <p className="whitespace-pre-line text-red-700">{error}</p>}
      </div>

      <ConfirmDialog
        open={confirmDiscard}
        onClose={() => setConfirmDiscard(false)}
        onConfirm={() =>
          void run(async () => {
            const r = await discardUnsavedImages(projectId)
            return `Discarded ${r.deleted} image(s), freeing ${mb(r.bytes_freed)}.`
          })
        }
        title={`Discard ${report.unsaved_images} unsaved image(s)?`}
        message={
          `These images are in no saved version, so nothing can bring them back — ` +
          `their annotations go with them.\n\n` +
          `If you want to keep them, close this and click "Save dataset" first.`
        }
        confirmLabel={`Discard ${report.unsaved_images}`}
        busy={busy}
        destructive
      />
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
  hasVersions,
  page,
  pageSize,
  total,
  onPageChange,
}: {
  images: DatasetImage[]
  projectId: number
  onDeleted: () => void
  selected: Set<number>
  onToggle: (id: number) => void
  onSelectAll: () => void
  onClearSelection: () => void
  hasVersions: boolean
  page: number
  pageSize: number
  total: number
  onPageChange: (page: number) => void
}) {
  const [pending, setPending] = useState<DatasetImage | null>(null)
  const [bulkOpen, setBulkOpen] = useState(false)
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

  async function handleBulkDelete() {
    setBusy(true)
    try {
      await bulkDeleteImages(projectId, [...selected])
      setBulkOpen(false)
      onClearSelection()
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
              {/* Red: this is the irreversible-action colour, and deleting
                  images is exactly what it's reserved for. 11px glyph beside
                  11px text. */}
              <button
                className="flex items-center gap-1 text-red-700 underline"
                onClick={() => setBulkOpen(true)}
              >
                <Trash2 size={11} />
                Delete {selected.size}
              </button>
              <button className="text-gray-500 underline" onClick={onClearSelection}>
                Clear
              </button>
            </>
          )}
          <button className="text-gray-500 underline" onClick={onSelectAll}>
            {selected.size === images.length ? 'Deselect all' : 'Select all'}
          </button>
          {/* Say WHICH images these are, not just how many are on screen. The
              old "N total" counted the rendered page, so a 638-image project
              read "200 total" and gave no hint the rest existed. */}
          <span className="tabular-nums text-gray-500">
            {total > images.length
              ? `${page * pageSize + 1}–${page * pageSize + images.length} of ${total}`
              : `${total} total`}
          </span>
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

      {/* Pager. Hidden entirely on a single-page project — controls that can
          never do anything are furniture. */}
      {total > pageSize && (
        <div className="mt-3 flex items-center justify-center gap-2 text-xs">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page === 0}
            className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ChevronLeft size={11} />
            Previous
          </button>
          <span className="tabular-nums text-gray-500">
            Page {page + 1} of {Math.ceil(total / pageSize)}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={(page + 1) * pageSize >= total}
            className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
            <ChevronRight size={11} />
          </button>
        </div>
      )}

      <ConfirmDialog
        open={pending !== null}
        onClose={() => setPending(null)}
        onConfirm={handleDelete}
        busy={busy}
        title="Delete image"
        // Which of the two things is actually about to happen. Saying "cannot
        // be undone" when a saved version can restore it teaches people to
        // distrust the versions feature; saying the opposite would be worse.
        message={
          `Delete "${pending?.original_filename}"?\n\n` +
          (hasVersions
            ? 'It leaves the dataset now, but its file is kept — restoring a ' +
              'saved version that contains it will bring it back.'
            : 'This removes the file from disk and cannot be undone. Save a ' +
              'dataset version first if you might want it back.')
        }
      />

      <ConfirmDialog
        open={bulkOpen}
        onClose={() => setBulkOpen(false)}
        onConfirm={handleBulkDelete}
        busy={busy}
        title={`Delete ${selected.size} image${selected.size === 1 ? '' : 's'}?`}
        message={
          `${selected.size} image${selected.size === 1 ? '' : 's'} will be removed ` +
          `from the dataset, along with their boxes.\n\n` +
          (hasVersions
            ? 'Their files are kept, so restoring a saved version that contains ' +
              'them will bring them back.'
            : 'This removes the files from disk and cannot be undone. Save a ' +
              'dataset version first if you might want them back.')
        }
        confirmLabel={`Delete ${selected.size}`}
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
