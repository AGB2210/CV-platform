import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Plus, Trash2, Upload, X } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { ConfirmDialog } from '@/components/ui/Modal'
import {
  createClass,
  deleteClass,
  deleteImage,
  getProject,
  listClasses,
  listImages,
  uploadImages,
  type DatasetImage,
  type Project,
  type ProjectClass,
} from '@/lib/api'

export function ProjectDetail() {
  // Route params always arrive as strings — the router can't know the type.
  // Convert once here rather than scattering Number(id) through the file.
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [project, setProject] = useState<Project | null>(null)
  const [images, setImages] = useState<DatasetImage[]>([])
  const [classes, setClasses] = useState<ProjectClass[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      // Promise.all, not three sequential awaits: the requests don't depend on
      // each other, so serialising them would triple the time to first paint
      // for no reason.
      const [p, imgs, cls] = await Promise.all([
        getProject(projectId),
        listImages(projectId),
        listClasses(projectId),
      ])
      setProject(p)
      setImages(imgs)
      setClasses(cls)
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
          <Link to="/" className="btn-secondary">
            <ArrowLeft size={14} />
            All projects
          </Link>
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
            <ImageGrid images={images} onDeleted={refresh} />
          </section>

          <ClassPanel projectId={projectId} classes={classes} onChanged={refresh} />
        </div>
      </PageBody>
    </>
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
  const [result, setResult] = useState<{ ok: number; skipped: string[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function send(files: File[]) {
    if (!files.length) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const res = await uploadImages(projectId, files)
      setResult({ ok: res.uploaded_count, skipped: res.skipped })
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
          JPG, PNG, BMP, WEBP — or a .zip containing them
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
          <span className="font-medium text-gray-800">{result.ok} uploaded</span>
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

function ImageGrid({
  images,
  onDeleted,
}: {
  images: DatasetImage[]
  onDeleted: () => void
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
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-medium text-gray-900">Images</h2>
        <span className="text-xs tabular-nums text-gray-500">{images.length} total</span>
      </div>

      {/* Dense auto-fill grid: as many ~130px columns as fit. Deliberately
          tighter than a typical gallery — the job here is to scan a dataset,
          not admire photos. */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(130px,1fr))] gap-2">
        {images.map((img) => (
          <div
            key={img.id}
            className="group relative overflow-hidden rounded border border-gray-200 bg-gray-100"
          >
            {/* aspect-square + object-cover: uniform tiles regardless of source
                aspect ratio, so the grid stays a grid. */}
            <img
              src={img.url}
              alt={img.original_filename}
              // Native lazy loading — a 5,000-image dataset must not issue
              // 5,000 requests on mount.
              loading="lazy"
              className="aspect-square w-full object-cover"
            />

            <button
              onClick={() => setPending(img)}
              className="absolute right-1 top-1 rounded bg-white/90 p-1 text-gray-500 opacity-0 shadow-sm transition-opacity hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
              aria-label={`Delete ${img.original_filename}`}
            >
              <X size={12} />
            </button>

            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-1.5 py-1">
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
