import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Database, SquarePen } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { CommitDialog } from '@/components/CommitDialog'
import {
  getDatasetStats,
  listAnnotations,
  listClasses,
  listImages,
  type Annotation,
  type DatasetImage,
  type DatasetStats,
  type ProjectClass,
  type Split,
} from '@/lib/api'

/**
 * Visualize — the whole dataset with its annotations drawn on.
 *
 * The Dataset grid answers "what did I upload". This answers "what does my
 * dataset actually contain", which is a different and more important question:
 * a box count tells you nothing about whether the boxes are on the objects.
 *
 * Rendered with the same SVG-over-<img> + viewBox technique as the editing
 * canvas (see AnnotationCanvas), so tile size is irrelevant to correctness —
 * one SVG unit is one image pixel at any thumbnail scale, no conversion maths.
 * This is a read-only view; editing lives in Annotate.
 */
export function Visualize() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [images, setImages] = useState<DatasetImage[]>([])
  const [classes, setClasses] = useState<ProjectClass[]>([])
  /** image_id -> boxes. Fetched per image, then cached. */
  const [boxes, setBoxes] = useState<Record<number, Annotation[]>>({})
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [showCommit, setShowCommit] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [splitFilter, setSplitFilter] = useState<'all' | Split>('all')
  const [stageFilter, setStageFilter] = useState<'all' | 'dataset' | 'staging'>('all')
  const [classFilter, setClassFilter] = useState<number | 'all'>('all')
  const [size, setSize] = useState(220)

  const load = useCallback(async () => {
    try {
      const [imgs, cls, s] = await Promise.all([
        listImages(projectId),
        listClasses(projectId),
        getDatasetStats(projectId),
      ])
      setImages(imgs)
      setClasses(cls)
      setStats(s)

      // Fetch every image's boxes in parallel rather than sequentially. For a
      // few hundred images this is fine; past that the right answer is a bulk
      // endpoint, not a faster loop. Noted rather than pre-built.
      const pairs = await Promise.all(
        imgs.map(async (i) => [i.id, await listAnnotations(i.id)] as const),
      )
      setBoxes(Object.fromEntries(pairs))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    void load()
  }, [load])

  const visible = useMemo(
    () =>
      images.filter((img) => {
        if (splitFilter !== 'all' && img.split !== splitFilter) return false
        if (stageFilter === 'dataset' && !img.in_dataset) return false
        if (stageFilter === 'staging' && img.in_dataset) return false
        if (classFilter !== 'all') {
          const b = boxes[img.id] ?? []
          if (!b.some((a) => a.category_id === classFilter)) return false
        }
        return true
      }),
    [images, boxes, splitFilter, stageFilter, classFilter],
  )

  const totalBoxes = visible.reduce((n, i) => n + (boxes[i.id]?.length ?? 0), 0)

  if (loading) {
    return (
      <>
        <PageHeader title="Visualize" />
        <PageBody>
          <p className="text-sm text-gray-500">Loading annotations…</p>
        </PageBody>
      </>
    )
  }

  return (
    <>
      <PageHeader
        title="Visualize"
        description="Every image with its annotations drawn on"
        actions={
          <>
            {/* The commit action belongs wherever you're LOOKING at the
                dataset, not only in the review screen. Seeing "staging" here
                with no way to act on it was a dead end. */}
            {stats && stats.staging_approved > 0 && (
              <button className="btn-primary" onClick={() => setShowCommit(true)}>
                <Database size={14} />
                Add {stats.staging_approved} to dataset
              </button>
            )}
            <Link to={`/projects/${projectId}`} className="btn-secondary">
              Dataset
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

        {/* Explain "staging" rather than leaving a chip nobody asked for.
            The label is meaningless without knowing what moves an image out of
            it, and the answer differs depending on whether the boxes are
            approved yet — so the banner states which case you're in and links
            to the one action that resolves it. */}
        {stats && stats.staging_total > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs">
            <span className="text-amber-900">
              <span className="font-medium">
                {stats.staging_total} image{stats.staging_total === 1 ? '' : 's'} in
                staging.
              </span>{' '}
              Staging images are not part of the trainable dataset and are excluded from
              exports and training.
            </span>

            {stats.staging_approved > 0 ? (
              <span className="text-amber-900">
                {stats.staging_approved}{' '}
                {stats.staging_approved === stats.staging_total ? 'are' : 'of them are'}{' '}
                approved and ready to add.
              </span>
            ) : (
              <span className="text-amber-900">
                They still have unreviewed boxes — approve them first.
              </span>
            )}

            <Link
              to={`/projects/${projectId}/review`}
              className="ml-auto inline-flex items-center gap-1 font-medium text-amber-900 underline underline-offset-2"
            >
              <SquarePen size={12} />
              {stats.staging_approved > 0 ? 'Review' : 'Approve them'}
            </Link>
          </div>
        )}

        {/* Filter bar. A dense row of small controls, not a settings panel —
            these get toggled constantly while auditing a dataset. */}
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md border border-gray-200 bg-white px-3 py-2">
          <Filter label="Split">
            <Select
              value={splitFilter}
              onChange={(v) => setSplitFilter(v as 'all' | Split)}
              options={[
                { value: 'all', label: 'All' },
                { value: 'train', label: 'Train' },
                { value: 'val', label: 'Val' },
                { value: 'test', label: 'Test' },
              ]}
            />
          </Filter>

          <Filter label="Stage">
            <Select
              value={stageFilter}
              onChange={(v) => setStageFilter(v as 'all' | 'dataset' | 'staging')}
              options={[
                { value: 'all', label: 'All' },
                { value: 'dataset', label: 'In dataset' },
                { value: 'staging', label: 'Staging' },
              ]}
            />
          </Filter>

          <Filter label="Class">
            <Select
              value={String(classFilter)}
              onChange={(v) => setClassFilter(v === 'all' ? 'all' : Number(v))}
              options={[
                { value: 'all', label: 'All' },
                ...classes.map((c) => ({ value: String(c.id), label: c.name })),
              ]}
            />
          </Filter>

          <Filter label="Size">
            <input
              type="range"
              min={140}
              max={420}
              step={20}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
              className="w-24 accent-accent-600"
            />
          </Filter>

          <span className="ml-auto text-xs tabular-nums text-gray-500">
            {visible.length} image{visible.length === 1 ? '' : 's'} · {totalBoxes} box
            {totalBoxes === 1 ? '' : 'es'}
          </span>
        </div>

        {/* Legend — without it, coloured boxes are just coloured boxes. */}
        {classes.length > 0 && (
          <div className="mb-3 flex flex-wrap items-center gap-3">
            {classes.map((c) => (
              <span key={c.id} className="flex items-center gap-1.5 text-xs text-gray-600">
                <span
                  className="h-2.5 w-2.5 rounded-sm border border-black/10"
                  style={{ backgroundColor: c.color }}
                />
                {c.name}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-xs text-gray-400">
              <svg width="16" height="8" aria-hidden>
                <line
                  x1="0"
                  y1="4"
                  x2="16"
                  y2="4"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                />
              </svg>
              unreviewed
            </span>
          </div>
        )}

        {visible.length === 0 ? (
          <div className="card border-dashed px-4 py-8">
            <h3 className="text-sm font-medium text-gray-900">Nothing to show</h3>
            <p className="mt-1 text-xs text-gray-500">
              {images.length === 0
                ? 'Upload images to this project first.'
                : 'No images match these filters.'}
            </p>
          </div>
        ) : (
          <div
            className="grid gap-3"
            // auto-fill + the size slider: the grid reflows to the window AND
            // the user controls density. Auditing wants many small tiles;
            // checking a specific box wants few large ones.
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` }}
          >
            {visible.map((img) => (
              <AnnotatedTile
                key={img.id}
                image={img}
                boxes={boxes[img.id] ?? []}
                classes={classes}
                projectId={projectId}
              />
            ))}
          </div>
        )}
      </PageBody>

      <CommitDialog
        open={showCommit}
        projectId={projectId}
        onClose={() => setShowCommit(false)}
        onCommitted={() => void load()}
      />
    </>
  )
}

function Filter({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-xs text-gray-500">{label}</span>
      {children}
    </label>
  )
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-gray-300 bg-white px-1.5 py-0.5 text-xs focus:border-accent-500 focus:outline-none"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

const SPLIT_STYLE: Record<string, string> = {
  train: 'bg-gray-700',
  val: 'bg-accent-600',
  test: 'bg-amber-600',
}

function AnnotatedTile({
  image,
  boxes,
  classes,
  projectId,
}: {
  image: DatasetImage
  boxes: Annotation[]
  classes: ProjectClass[]
  projectId: number
}) {
  const colorOf = (id: number) => classes.find((c) => c.id === id)?.color ?? '#71717a'

  return (
    <Link
      to={`/projects/${projectId}/review/${image.id}`}
      className="group block overflow-hidden rounded border border-gray-200 bg-white transition-colors hover:border-accent-400"
      title={`${image.original_filename} — click to edit`}
    >
      <div className="relative bg-gray-100">
        <img src={image.url} alt="" loading="lazy" className="block w-full" />

        {/* Same viewBox trick as the editor: user units == image pixels, so
            stored coordinates draw correctly at whatever size the tile ends up.
            pointer-events-none so the whole tile stays one link target. */}
        <svg
          viewBox={`0 0 ${image.width} ${image.height}`}
          className="pointer-events-none absolute inset-0 h-full w-full"
        >
          {boxes.map((b) => (
            <rect
              key={b.id}
              x={b.x}
              y={b.y}
              width={b.width}
              height={b.height}
              fill={colorOf(b.category_id)}
              fillOpacity={0.12}
              stroke={colorOf(b.category_id)}
              // Without this, stroke width scales with the viewBox: a 4000px
              // image would draw hairlines and a 300px one would draw slabs.
              vectorEffect="non-scaling-stroke"
              strokeWidth={1.5}
              strokeDasharray={b.reviewed ? undefined : '4 3'}
            />
          ))}
        </svg>

        {/* Split chip. Only meaningful once an image is in the dataset — a
            staging image's split hasn't been decided yet. */}
        {image.in_dataset && (
          <span
            className={`absolute left-1 top-1 rounded px-1 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white ${
              SPLIT_STYLE[image.split] ?? 'bg-gray-500'
            }`}
          >
            {image.split}
          </span>
        )}
        {!image.in_dataset && (
          <span
            className="absolute left-1 top-1 rounded bg-white/90 px-1 py-0.5 text-[10px] font-medium text-gray-600"
            title="Not in the trainable dataset yet — approve its boxes, then use Add to dataset"
          >
            staging
          </span>
        )}

        <span className="absolute right-1 top-1 rounded bg-black/60 px-1 py-0.5 text-[10px] tabular-nums text-white">
          {boxes.length}
        </span>
      </div>

      <div className="px-1.5 py-1">
        <p className="truncate text-[11px] text-gray-700">{image.original_filename}</p>
        <p className="text-[10px] tabular-nums text-gray-400">
          {image.width}×{image.height}
        </p>
      </div>
    </Link>
  )
}
