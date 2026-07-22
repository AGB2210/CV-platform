import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ChevronLeft, ChevronRight, SquarePen } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { PageSizeSelect } from '@/components/PageSizeSelect'
import {
  getDatasetStats,
  listAnnotations,
  listClasses,
  listImagePage,
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
  /** image_id -> boxes, for the LOADED PAGE only. */
  const [boxes, setBoxes] = useState<Record<number, Annotation[]>>({})
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters — applied SERVER-SIDE, so they search the whole dataset. Filtering
  // the loaded page client-side shipped a real contradiction: the stats banner
  // said "1 image has no boxes" (whole-dataset) while the No-boxes filter found
  // nothing, because that image sat beyond the first page.
  const [splitFilter, setSplitFilter] = useState<'all' | Split>('all')
  const [stateFilter, setStateFilter] = useState<'all' | 'annotated' | 'empty' | 'pending'>(
    'all',
  )
  const [classFilter, setClassFilter] = useState<number | 'all'>('all')
  const [size, setSize] = useState(220)

  // Paging. Visualize used to silently show the server's default first page —
  // 200 images presented as if they were the dataset.
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(200)
  const [total, setTotal] = useState(0)

  const load = useCallback(async () => {
    try {
      const [pageResult, cls, s] = await Promise.all([
        listImagePage(projectId, pageSize, page * pageSize, {
          split: splitFilter === 'all' ? undefined : splitFilter,
          state:
            stateFilter === 'all'
              ? undefined
              : stateFilter === 'empty'
                ? 'unannotated'
                : stateFilter,
          categoryId: classFilter === 'all' ? undefined : classFilter,
        }),
        listClasses(projectId),
        getDatasetStats(projectId),
      ])
      setImages(pageResult.images)
      setTotal(pageResult.total)
      setClasses(cls)
      setStats(s)

      // Boxes for THIS PAGE's images, in parallel. Bounded by the page size, so
      // a 5,000-image project doesn't fire 5,000 requests.
      const pairs = await Promise.all(
        pageResult.images.map(async (i) => [i.id, await listAnnotations(i.id)] as const),
      )
      setBoxes(Object.fromEntries(pairs))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [projectId, page, pageSize, splitFilter, stateFilter, classFilter])

  useEffect(() => {
    void load()
  }, [load])

  // A filter change makes the current page number meaningless.
  const withPageReset = <T,>(set: (v: T) => void) => (v: T) => {
    set(v)
    setPage(0)
  }

  const visible = images
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
          <Link to={`/projects/${projectId}`} className="btn-secondary">
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

        {/* A pending batch is the one thing worth interrupting for here: those
            boxes aren't in the dataset and won't export until you decide. */}
        {stats && stats.proposed_boxes > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-accent-200 bg-accent-50 px-3 py-2 text-xs">
            <span className="text-accent-900">
              <span className="font-medium">
                {stats.proposed_boxes} model proposal
                {stats.proposed_boxes === 1 ? '' : 's'} across {stats.proposed_images}{' '}
                image{stats.proposed_images === 1 ? '' : 's'}
              </span>{' '}
              are pending — not part of your dataset, and not exported, until you accept
              them.
            </span>
            <Link
              to={`/projects/${projectId}/review`}
              className="ml-auto inline-flex items-center gap-1 font-medium text-accent-900 underline underline-offset-2"
            >
              <SquarePen size={12} />
              Review them
            </Link>
          </div>
        )}

        {/* Unannotated images are worth flagging now that nothing holds them
            back: with staging gone they export as negative examples, which is
            usually right but should never be a surprise. */}
        {stats && stats.unannotated_images > 0 && (
          <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
            {/* Agreement runs through the whole sentence, not just the noun.
                Pluralising "image" alone left "1 image have no boxes … delete
                them", which reads as broken software in a warning the user is
                meant to trust. */}
            {stats.unannotated_images === 1 ? (
              <>
                <span className="font-medium">1 image has no boxes.</span> It'll
                export as a negative example (a scene containing none of your
                classes). Annotate or delete it if that isn't what you want.
              </>
            ) : (
              <>
                <span className="font-medium">
                  {stats.unannotated_images} images have no boxes.
                </span>{' '}
                They'll export as negative examples (scenes containing none of
                your classes). Annotate or delete them if that isn't what you
                want.
              </>
            )}
          </div>
        )}

        {/* Filter bar. A dense row of small controls, not a settings panel —
            these get toggled constantly while auditing a dataset. */}
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-md border border-gray-200 bg-white px-3 py-2">
          <Filter label="Split">
            <Select
              value={splitFilter}
              onChange={withPageReset((v) => setSplitFilter(v as 'all' | Split))}
              options={[
                { value: 'all', label: 'All' },
                { value: 'train', label: 'Train' },
                { value: 'val', label: 'Val' },
                { value: 'test', label: 'Test' },
              ]}
            />
          </Filter>

          <Filter label="State">
            <Select
              value={stateFilter}
              onChange={withPageReset((v) =>
                setStateFilter(v as 'all' | 'annotated' | 'empty' | 'pending'),
              )}
              options={[
                { value: 'all', label: 'All' },
                { value: 'annotated', label: 'Annotated' },
                { value: 'empty', label: 'No boxes' },
                { value: 'pending', label: 'Pending review' },
              ]}
            />
          </Filter>

          <Filter label="Class">
            <Select
              value={String(classFilter)}
              onChange={withPageReset((v) => setClassFilter(v === 'all' ? 'all' : Number(v)))}
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
            {/* The MATCH count is dataset-wide; boxes are counted for the page
                actually on screen, and say so when those differ. */}
            {total} image{total === 1 ? '' : 's'}
            {total > visible.length ? ` · showing ${visible.length}` : ''} · {totalBoxes} box
            {totalBoxes === 1 ? '' : 'es'}
            {total > visible.length ? ' on this page' : ''}
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
              {(stats?.total_images ?? 0) === 0
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

        {/* Pager + page size — same bar as the Dataset grid, so the two views
            page identically. */}
        {total > 0 && (
          <div className="mt-4 flex items-center justify-center gap-2 text-xs">
            {total > pageSize && (
              <>
                <button
                  onClick={() => setPage(page - 1)}
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
                  onClick={() => setPage(page + 1)}
                  disabled={(page + 1) * pageSize >= total}
                  className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Next
                  <ChevronRight size={11} />
                </button>
              </>
            )}
            <PageSizeSelect
              value={pageSize}
              onChange={(n) => {
                setPageSize(n)
                setPage(0)
              }}
            />
          </div>
        )}
      </PageBody>
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
        {/* Thumbnail, not the original: these are grid cells, and decoding
            200 full-size images while scrolling is the lag the thumbs fix.
            The box overlay is unaffected — its viewBox is the image's natural
            size and scales with the container either way. */}
        <img src={image.thumb_url} alt="" loading="lazy" className="block w-full" />

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

        {/* Split chip. Every image is a dataset image now, so this always
            applies — there is no staging state to be in instead. */}
        <span
          className={`absolute left-1 top-1 rounded px-1 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white ${
            SPLIT_STYLE[image.split] ?? 'bg-gray-500'
          }`}
        >
          {image.split}
        </span>

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
