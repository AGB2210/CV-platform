import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Check, ChevronLeft, ChevronRight, Database, Trash2, X } from 'lucide-react'
import { AnnotationCanvas } from '@/components/AnnotationCanvas'
import { CommitDialog } from '@/components/CommitDialog'
import { ProposalBar } from '@/components/ProposalBar'
import {
  acceptImageProposals,
  rejectImageProposals,
  createAnnotation,
  deleteAnnotation,
  getDatasetStats,
  listAnnotations,
  listClasses,
  listImages,
  updateAnnotation,
  type Annotation,
  type DatasetImage,
  type DatasetStats,
  type ProjectClass,
} from '@/lib/api'

/**
 * Annotation review — where model output becomes trusted data.
 *
 * This is the screen that makes auto-annotation useful. Without it the model's
 * boxes are a number in a summary card; you have to trust them blind, or export
 * and open the dataset in some other tool to find out what actually happened.
 *
 * Layout is deliberate: filmstrip on the left for navigation and progress,
 * canvas in the middle at maximum size, class picker on the right. Review is a
 * repetitive task, so the whole design optimises for "look, judge, approve,
 * next" without moving the mouse much.
 */
export function Review() {
  const { id, imageId } = useParams<{ id: string; imageId: string }>()
  const projectId = Number(id)
  const navigate = useNavigate()

  const [images, setImages] = useState<DatasetImage[]>([])
  const [classes, setClasses] = useState<ProjectClass[]>([])
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [activeClassId, setActiveClassId] = useState<number | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [showCommit, setShowCommit] = useState(false)

  const current = useMemo(
    () => images.find((i) => i.id === Number(imageId)) ?? null,
    [images, imageId],
  )
  const index = useMemo(
    () => images.findIndex((i) => i.id === Number(imageId)),
    [images, imageId],
  )

  // --- data --------------------------------------------------------------

  const loadShell = useCallback(async () => {
    try {
      const [imgs, cls, s] = await Promise.all([
        listImages(projectId),
        listClasses(projectId),
        getDatasetStats(projectId),
      ])
      setImages(imgs)
      setClasses(cls)
      setStats(s)
      setActiveClassId((c) => c ?? cls[0]?.id ?? null)
      // No image in the URL: land on the first one rather than an empty screen.
      if (!imageId && imgs.length) {
        navigate(`/projects/${projectId}/review/${imgs[0].id}`, { replace: true })
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [projectId, imageId, navigate])

  useEffect(() => {
    void loadShell()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  // Reload boxes whenever the image changes.
  useEffect(() => {
    if (!imageId) return
    let cancelled = false
    listAnnotations(Number(imageId))
      .then((a) => !cancelled && setAnnotations(a))
      .catch((e: Error) => !cancelled && setError(e.message))
    setSelectedId(null)
    return () => {
      cancelled = true
    }
  }, [imageId])

  /** Refresh the filmstrip counts and dataset stats without refetching classes. */
  const refreshCounts = useCallback(async () => {
    try {
      const [imgs, s] = await Promise.all([listImages(projectId), getDatasetStats(projectId)])
      setImages(imgs)
      setStats(s)
    } catch {
      /* non-fatal: the counts are a nicety, not the work */
    }
  }, [projectId])


  // --- mutations ---------------------------------------------------------
  //
  // Each of these applies the change to local state immediately and then
  // reconciles with the server's response. Waiting for a round-trip before the
  // box moves would make dragging feel broken on a task you repeat 500 times.

  async function handleCreate(
    rect: { x: number; y: number; width: number; height: number },
    categoryId: number,
  ) {
    if (!current) return
    try {
      const created = await createAnnotation(current.id, {
        category_id: categoryId,
        ...rect,
      })
      setAnnotations((a) => [...a, created])
      setSelectedId(created.id)
      void refreshCounts()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function handleUpdate(
    annId: number,
    rect: { x: number; y: number; width: number; height: number },
  ) {
    // Optimistic: move it now, ask the server after.
    setAnnotations((a) => a.map((x) => (x.id === annId ? { ...x, ...rect } : x)))
    try {
      const updated = await updateAnnotation(annId, rect)
      // Reconcile — the server may have clamped it, and it flips source to
      // "manual", which changes the box from dashed to solid.
      setAnnotations((a) => a.map((x) => (x.id === annId ? updated : x)))
      void refreshCounts()
    } catch (e) {
      setError((e as Error).message)
      // Roll back by refetching the truth rather than guessing.
      if (current) setAnnotations(await listAnnotations(current.id))
    }
  }

  async function handleRelabel(annId: number, categoryId: number) {
    setAnnotations((a) =>
      a.map((x) => (x.id === annId ? { ...x, category_id: categoryId } : x)),
    )
    try {
      const updated = await updateAnnotation(annId, { category_id: categoryId })
      setAnnotations((a) => a.map((x) => (x.id === annId ? updated : x)))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function handleDelete(annId: number) {
    const backup = annotations
    setAnnotations((a) => a.filter((x) => x.id !== annId))
    setSelectedId(null)
    try {
      await deleteAnnotation(annId)
      void refreshCounts()
    } catch (e) {
      setError((e as Error).message)
      setAnnotations(backup)
    }
  }

  /** Accept THIS image's proposals: the model's boxes replace this image's. */
  async function handleAcceptImage() {
    if (!current) return
    try {
      await acceptImageProposals(current.id)
      setAnnotations(await listAnnotations(current.id))
      await refreshCounts()
      advanceIfMoreProposals()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** Reject THIS image's proposals: its own boxes stay exactly as they were. */
  async function handleRejectImage() {
    if (!current) return
    try {
      await rejectImageProposals(current.id)
      setAnnotations(await listAnnotations(current.id))
      await refreshCounts()
      advanceIfMoreProposals()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  /** After deciding on an image, jump to the next one still awaiting a
   *  decision — not merely the next image. Judging a batch means visiting the
   *  images the model actually touched, and stepping through untouched ones in
   *  between is friction for nothing. */
  function advanceIfMoreProposals() {
    const next = images.find((i, idx) => idx > index && i.proposed_count > 0)
    if (next) navigate(`/projects/${projectId}/review/${next.id}`)
  }

  const reloadCurrent = useCallback(async () => {
    if (!current) return
    setAnnotations(await listAnnotations(current.id))
    await refreshCounts()
  }, [current, refreshCounts])

  const goTo = useCallback(
    (delta: number) => {
      const next = images[index + delta]
      if (next) navigate(`/projects/${projectId}/review/${next.id}`)
    },
    [images, index, navigate, projectId],
  )

  // --- keyboard ----------------------------------------------------------
  // Judging a batch is hundreds of repetitions. Shortcuts aren't a power-user
  // luxury here, they're the difference between a usable tool and a chore.

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = document.activeElement
      if (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return

      if (e.key === 'ArrowRight' || e.key === 'd') goTo(1)
      if (e.key === 'ArrowLeft' || e.key === 'a') goTo(-1)
      // Enter used to approve. It now accepts THIS image's model boxes, which
      // is the same gesture ("yes, this image is right, next") aimed at the
      // decision that still exists. It does nothing when no batch is pending —
      // better than a key that silently no-ops on every screen.
      if (e.key === 'Enter' && proposals.length > 0) void handleAcceptImage()

      // Number keys pick the active class — 1..9 map to the class list. Faster
      // than reaching for the mouse on every box.
      const n = Number(e.key)
      if (!Number.isNaN(n) && n >= 1 && n <= classes.length) {
        const cls = classes[n - 1]
        setActiveClassId(cls.id)
        // With a box selected, the number key relabels it instead.
        if (selectedId !== null) void handleRelabel(selectedId, cls.id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [goTo, classes, selectedId, current, annotations])

  // --- render ------------------------------------------------------------

  if (loading) {
    return <div className="p-6 text-sm text-gray-500">Loading…</div>
  }

  if (!images.length) {
    return (
      <div className="p-6">
        <p className="text-sm text-gray-600">This project has no images yet.</p>
        <Link to={`/projects/${projectId}`} className="btn-secondary mt-3">
          Go to dataset
        </Link>
      </div>
    )
  }

  // Proposals aren't annotations, so they're counted apart from everything else
  // on this screen.
  const proposals = annotations.filter((a) => a.proposed)
  const accepted = annotations.filter((a) => !a.proposed)
  const unreviewed = accepted.filter((a) => !a.reviewed).length

  // ONE set of boxes on screen, never two.
  //
  // While a batch is pending you see the model's output ALONE — that's the
  // thing you're judging, and overlaying it on your own boxes made the two
  // indistinguishable on the same objects. Reject and your boxes reappear
  // untouched; accept and they become your boxes.
  const reviewingBatch = proposals.length > 0
  const visibleBoxes = reviewingBatch ? proposals : accepted

  return (
    <div className="flex h-full min-h-0 flex-1">
      {/* --- Filmstrip --- */}
      <aside className="flex w-40 shrink-0 flex-col border-r border-gray-200 bg-white">
        <div className="border-b border-gray-200 px-3 py-2">
          <p className="label-eyebrow">Images</p>
          <p className="text-xs tabular-nums text-gray-500">
            {images.filter((i) => i.annotation_count > 0 && i.reviewed_count === i.annotation_count)
              .length}{' '}
            / {images.length} done
          </p>
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2">
          {images.map((img, i) => {
            const isCurrent = img.id === Number(imageId)
            const done = img.annotation_count > 0 && img.reviewed_count === img.annotation_count
            return (
              <button
                key={img.id}
                onClick={() => navigate(`/projects/${projectId}/review/${img.id}`)}
                className={[
                  'relative block w-full overflow-hidden rounded border text-left transition-colors',
                  isCurrent
                    ? 'border-accent-600 ring-1 ring-accent-600'
                    : 'border-gray-200 hover:border-gray-300',
                ].join(' ')}
              >
                <img
                  src={img.url}
                  alt=""
                  loading="lazy"
                  className="aspect-video w-full object-cover"
                />
                <div className="flex items-center justify-between px-1.5 py-1">
                  <span className="text-[10px] tabular-nums text-gray-500">#{i + 1}</span>
                  <span
                    className={`text-[10px] tabular-nums ${
                      done ? 'text-status-good' : 'text-gray-400'
                    }`}
                  >
                    {done && <Check size={9} className="mr-0.5 inline" />}
                    {img.annotation_count}
                  </span>
                </div>
              </button>
            )
          })}
        </div>
      </aside>

      {/* --- Canvas ---
          <section>, not <main>: AppShell already renders the page's single
          <main> and this component is rendered inside it. Nested <main> is
          invalid HTML and makes screen readers announce two main landmarks. */}
      <section className="flex min-w-0 flex-1 flex-col bg-gray-100">
        {/* The pending batch sits above the canvas — it's the thing you'd act
            on first, and it disappears once applied or discarded. */}
        {stats && (
          <ProposalBar
            projectId={projectId}
            proposedBoxes={stats.proposed_boxes}
            onChanged={() => void reloadCurrent()}
          />
        )}
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-4">
          <div className="flex items-center gap-2">
            <button
              className="btn-secondary px-1.5"
              onClick={() => goTo(-1)}
              disabled={index <= 0}
              aria-label="Previous image"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              className="btn-secondary px-1.5"
              onClick={() => goTo(1)}
              disabled={index >= images.length - 1}
              aria-label="Next image"
            >
              <ChevronRight size={14} />
            </button>
            <div className="ml-2 min-w-0">
              <p className="truncate text-sm font-medium text-gray-900">
                {current?.original_filename}
              </p>
              <p className="text-xs tabular-nums text-gray-500">
                {index + 1} of {images.length} · {current?.width}×{current?.height} ·{' '}
                {accepted.length} box{accepted.length === 1 ? '' : 'es'}
                {unreviewed > 0 && (
                  <span className="text-status-busy"> · {unreviewed} unreviewed</span>
                )}
                {proposals.length > 0 && (
                  <span className="text-accent-700"> · {proposals.length} proposed</span>
                )}
              </p>
            </div>
          </div>

          {/* ONLY image-level actions live here. Project-level ones (approve
              all, add to dataset) are in the right panel — mixing the two
              scopes in one toolbar both muddled the meaning and overflowed the
              header at narrow widths. */}
          <div className="flex shrink-0 items-center gap-2">
            {/* While judging a batch the ONLY actions are accept/reject for
                this image. Approve is meaningless here — you can't approve
                boxes that aren't yours yet. */}
            {reviewingBatch ? (
              <>
                {/* Scoped to the CURRENT image. The bar above does the whole
                    batch and says "all" — this pair says "this image", because
                    two identical "Reject" buttons with different blast radii is
                    a trap. */}
                <span className="hidden text-[11px] uppercase tracking-wide text-gray-400 sm:inline">
                  This image
                </span>
                {/* Red = reject, blue = accept, identically to the batch bar
                    above. The colour has to mean the same thing in both places
                    or it means nothing. */}
                <button
                  className="btn bg-red-600 text-white hover:bg-red-700"
                  onClick={() => void handleRejectImage()}
                  title="Discard THIS image's model boxes and keep your own. Other images are unaffected."
                >
                  <X size={14} />
                  Reject image
                </button>
                <button
                  className="btn-primary"
                  onClick={() => void handleAcceptImage()}
                  title="Keep THIS image's model boxes, replacing your own. Other images are unaffected."
                >
                  <Check size={14} />
                  Accept image ({proposals.length})
                </button>
              </>
            ) : null}
          </div>
        </header>

        {error && (
          <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-800">
            {error}
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-auto p-6">
          {current && (
            <div className="mx-auto w-fit shadow-sm ring-1 ring-gray-300">
              <AnnotationCanvas
                imageUrl={current.url}
                imageWidth={current.width}
                imageHeight={current.height}
                annotations={visibleBoxes}
                classes={classes}
                activeClassId={activeClassId}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onCreate={(r, c) => void handleCreate(r, c)}
                onUpdate={(i, r) => void handleUpdate(i, r)}
                onDelete={(i) => void handleDelete(i)}
                // Editing is disabled while judging a batch: these aren't your
                // boxes yet. Accept them first, then edit freely.
                readOnly={reviewingBatch}
              />
            </div>
          )}
        </div>
      </section>

      {/* --- Classes + boxes --- */}
      <aside className="flex w-60 shrink-0 flex-col border-l border-gray-200 bg-white">
        {/* Project-scoped block, kept visually distinct from the per-image
            controls below it. This is where you leave the annotation loop.

            The "Progress N/M" bar and "Approve all" button used to live here.
            Both are gone: with the proposal model, accepting IS the
            confirmation, so every accepted box is reviewed by construction. The
            bar sat permanently at 100% and the button permanently read "All
            approved" — furniture that could never change state, next to a click
            that could never do anything. */}
        {stats && (
          <div className="border-b border-gray-200 bg-gray-50 p-3">
            <p className="label-eyebrow mb-1.5">Dataset</p>
            <dl className="space-y-1 text-xs">
              <div className="flex justify-between">
                <dt className="text-gray-500">Annotated, staged</dt>
                <dd className="font-mono tabular-nums text-gray-900">
                  {stats.staging_approved}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">In dataset</dt>
                <dd className="font-mono tabular-nums text-gray-900">
                  {stats.dataset_total}
                </dd>
              </div>
            </dl>

            {stats.staging_approved > 0 ? (
              <button
                className="btn-primary mt-2 w-full"
                onClick={() => setShowCommit(true)}
              >
                <Database size={13} />
                Add {stats.staging_approved} to dataset
              </button>
            ) : (
              <p className="mt-2 text-[11px] text-gray-400">
                {stats.staging_total > 0
                  ? `${stats.staging_total} staged image(s) have no boxes yet. Annotate them to add them to the dataset.`
                  : 'Nothing staged.'}
              </p>
            )}
          </div>
        )}

        <div className="border-b border-gray-200 px-3 py-2">
          <p className="label-eyebrow">Draw as</p>
        </div>
        <ul className="border-b border-gray-200 p-2">
          {classes.map((c, i) => (
            <li key={c.id}>
              <button
                onClick={() => setActiveClassId(c.id)}
                className={[
                  'flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors',
                  activeClassId === c.id
                    ? 'bg-accent-50 font-medium text-accent-800'
                    : 'text-gray-700 hover:bg-gray-50',
                ].join(' ')}
              >
                <span
                  className="h-3 w-3 shrink-0 rounded-sm border border-black/10"
                  style={{ backgroundColor: c.color }}
                />
                <span className="flex-1 truncate">{c.name}</span>
                {/* Show the shortcut on the affordance itself — a shortcut
                    nobody discovers may as well not exist. */}
                <kbd className="rounded border border-gray-300 px-1 font-mono text-[10px] text-gray-500">
                  {i + 1}
                </kbd>
              </button>
            </li>
          ))}
        </ul>

        <div className="border-b border-gray-200 px-3 py-2">
          <p className="label-eyebrow">
            {reviewingBatch ? "Model's boxes" : 'Boxes on this image'}
          </p>
          {reviewingBatch && (
            <p className="text-[11px] text-gray-400">Your boxes are hidden while you decide</p>
          )}
        </div>
        <ul className="min-h-0 flex-1 divide-y divide-gray-100 overflow-y-auto">
          {annotations.length === 0 && (
            <li className="px-3 py-4 text-xs text-gray-400">
              No boxes. Drag on the image to draw one.
            </li>
          )}
          {/* Lists whatever the canvas is showing — the model's boxes while a
              batch is pending, yours otherwise. Never a mix. */}
          {visibleBoxes.map((a) => {
            const cls = classes.find((c) => c.id === a.category_id)
            return (
              <li
                key={a.id}
                onMouseEnter={() => setSelectedId(a.id)}
                className={`group flex items-center gap-2 px-3 py-1.5 text-xs ${
                  selectedId === a.id ? 'bg-accent-50' : ''
                }`}
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-sm border border-black/10"
                  style={{ backgroundColor: cls?.color }}
                />
                <span className="flex-1 truncate text-gray-800">{cls?.name}</span>
                {a.confidence !== null && (
                  <span className="font-mono tabular-nums text-gray-400">
                    {a.confidence.toFixed(2)}
                  </span>
                )}
                {!a.proposed && !a.reviewed && (
                  <span
                    className="h-1.5 w-1.5 rounded-full bg-status-busy"
                    title="Unreviewed"
                  />
                )}
                {/* No per-box delete while judging: the decision is accept or
                    reject the batch, not curate it box by box. */}
                {!reviewingBatch && (
                  <button
                    onClick={() => void handleDelete(a.id)}
                    className="rounded p-0.5 text-gray-400 opacity-0 hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
                    aria-label="Delete box"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </li>
            )
          })}
        </ul>

        <div className="space-y-0.5 border-t border-gray-200 p-3 text-[11px] text-gray-400">
          <p>
            <Kbd>drag</Kbd> draw · <Kbd>1-9</Kbd> class
          </p>
          <p>
            <Kbd>Del</Kbd> delete · <Kbd>Esc</Kbd> deselect
          </p>
          <p>
            <Kbd>←</Kbd> <Kbd>→</Kbd> navigate
            {reviewingBatch && (
              <>
                {' · '}
                <Kbd>Enter</Kbd> accept image
              </>
            )}
          </p>
          {/* The legend only earns its space while a batch is pending — that's
              the only time dashed boxes are on screen. Otherwise every box is
              yours and solid, and a legend explaining one state is noise. */}
          {reviewingBatch && (
            <div className="space-y-0.5 pt-1.5">
              <LegendRow dash="6 4" label="the model's — not yours until accepted" />
            </div>
          )}
        </div>
      </aside>

      <CommitDialog
        open={showCommit}
        projectId={projectId}
        onClose={() => setShowCommit(false)}
        onCommitted={() => void refreshCounts()}
      />

    </div>
  )
}

function LegendRow({ dash, label }: { dash: string | undefined; label: string }) {
  return (
    <p className="flex items-center gap-1.5">
      <svg width="18" height="8" aria-hidden className="shrink-0">
        <line
          x1="0"
          y1="4"
          x2="18"
          y2="4"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeDasharray={dash}
        />
      </svg>
      {label}
    </p>
  )
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-gray-300 bg-gray-50 px-1 font-mono text-[10px] text-gray-600">
      {children}
    </kbd>
  )
}
