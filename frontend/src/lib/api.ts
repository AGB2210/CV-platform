/**
 * Typed API client.
 *
 * Every backend call in the app goes through here rather than calling fetch()
 * directly in components. That gives us one place to handle error shape, base
 * URL, and (later) auth headers — instead of 40 slightly different fetch calls
 * with 40 slightly different error handling bugs.
 *
 * Note the relative '/api' base: in dev, Vite proxies it to :8000 (see
 * vite.config.ts); in production, FastAPI serves the built frontend from the
 * same origin. Same code path either way, no environment switch.
 */

const BASE_URL = '/api'

/** Error carrying the HTTP status, so callers can branch on 404 vs 500. */
export class ApiError extends Error {
  // Declared and assigned explicitly rather than using a TypeScript parameter
  // property (`constructor(public status: number)`). This project builds with
  // `erasableSyntaxOnly`, which only permits TS syntax that can be erased to
  // plain JS — parameter properties generate real assignments, so they're out.
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * Turn a failed Response into a message worth reading.
 *
 * FastAPI reports errors as {"detail": ...} by convention, but `detail` is not
 * always a string: a 422 carries an ARRAY of per-field validation objects, and
 * rendering that with template interpolation produced the useless
 * "[object Object]". Each shape is unpacked to something a person can act on.
 */
async function describeFailure(res: Response): Promise<string> {
  let body: unknown
  try {
    body = await res.json()
  } catch {
    // Not JSON at all — a proxy error page, or a dead backend. The status line
    // is all there is, and it beats a JSON parse crash.
    return `${res.status} ${res.statusText}`
  }

  const detail = (body as { detail?: unknown })?.detail
  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    // Pydantic validation errors: [{loc: [...], msg: "...", type: "..."}]
    return detail
      .map((e) => {
        const item = e as { loc?: unknown[]; msg?: string }
        // Drop the leading "body"/"query" frame — it names the HTTP envelope,
        // not the field the user got wrong.
        const where = (item.loc ?? []).slice(1).join('.')
        return where ? `${where}: ${item.msg}` : item.msg
      })
      .filter(Boolean)
      .join('; ')
  }

  if (detail) return JSON.stringify(detail)
  return `${res.status} ${res.statusText}`
}

/**
 * A fetch() rejection, explained.
 *
 * fetch only rejects for network-level failures, and the browser deliberately
 * gives one opaque message ("Failed to fetch" / "NetworkError") for all of them
 * so a page can't probe the network. That message alone is useless to someone
 * staring at an upload that didn't work, so we say what it can actually mean.
 */
function describeNetworkFailure(err: unknown, context: string): ApiError {
  const raw = err instanceof Error ? err.message : String(err)
  return new ApiError(
    `Could not reach the server (${raw}).\n\n` +
      `This is a connection-level failure, so there is no server message to ` +
      `show. Usual causes, most likely first:\n` +
      `  • ${context}\n` +
      `  • the backend stopped — check the window running start.bat\n` +
      `  • the request was refused before it finished sending`,
    0,
  )
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options?.headers },
      ...options,
    })
  } catch (err) {
    throw describeNetworkFailure(err, 'the backend is not running')
  }

  if (!res.ok) throw new ApiError(await describeFailure(res), res.status)

  // 204 No Content has no body to parse.
  if (res.status === 204) return undefined as T

  return res.json() as Promise<T>
}

/**
 * Upload files as multipart/form-data.
 *
 * Separate from request() because it must NOT set Content-Type. That looks like
 * an omission but is load-bearing: the browser has to generate the header
 * itself so it can append the `boundary=...` token that delimits the parts.
 * Setting 'multipart/form-data' by hand omits the boundary, and the server
 * fails to parse the body with a confusing 422.
 */
async function upload<T>(path: string, files: File[], importId?: string): Promise<T> {
  const form = new FormData()
  // Field name must be 'files' — it matches the `files: list[UploadFile]`
  // parameter in the FastAPI endpoint.
  for (const file of files) form.append('files', file)

  // Shared by every batch of ONE upload, so the whole import can be undone as a
  // unit even though a large folder arrives as dozens of separate requests.
  if (importId) form.append('import_id', importId)

  // A folder upload also sends each file's path RELATIVE to the chosen folder,
  // in the same order, because the multipart filename carries only the
  // basename — so "train/a.png" and "val/a.png" would arrive indistinguishable
  // and the split information the user's own layout encodes would be lost.
  //
  // webkitRelativePath is '' for ordinary file selections, so this appends
  // nothing there and the server takes the plain-upload path.
  const relative = files.map((f) => f.webkitRelativePath || '')
  if (relative.some(Boolean)) {
    for (const p of relative) form.append('paths', p)
  }

  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, { method: 'POST', body: form })
  } catch (err) {
    throw describeNetworkFailure(
      err,
      `the batch was too large for one request (${files.length} files, ` +
        `${(files.reduce((n, f) => n + f.size, 0) / 1048576).toFixed(0)} MB)`,
    )
  }

  if (!res.ok) throw new ApiError(await describeFailure(res), res.status)
  return res.json() as Promise<T>
}

// --- Batching a large upload ------------------------------------------------
//
// Starlette refuses a multipart request carrying more than 1000 files, and it
// refuses it partway through RECEIVING the body. The browser is still uploading
// when the connection closes, so it never gets to read the 400 — it reports the
// opaque "Failed to fetch" and the real reason ("Too many files. Maximum number
// of files is 1000.") is never seen. A 5,000-image dataset hit this every time.
//
// So the client sends batches. Deliberately well under the limit, because each
// batch also carries its annotation sidecars.

/** Files per request. 1000 is the wall; this leaves room for sidecars. */
const BATCH_MAX_FILES = 400
/** Bytes per request. Bounds how much the server buffers at once, and keeps a
 *  single failure from costing a large re-send. Generous because this is
 *  loopback, not a network. */
const BATCH_MAX_BYTES = 128 * 1024 * 1024

/** Files that describe other files rather than being dataset content. */
const SIDECAR_RE = /\.(json|ya?ml|txt|names)$/i

const isSidecar = (f: File) => SIDECAR_RE.test(f.name)

/** The top-level folder a file sits in — "train" for "train/images/a.jpg".
 *  This is the scope the importer resolves annotations within, so it is also
 *  the scope a batch's sidecars have to be chosen by. */
function topFolder(f: File): string {
  const rel = f.webkitRelativePath || ''
  const parts = rel.split('/').filter(Boolean)
  // parts[0] is the picked folder itself; parts[1] is the split.
  return parts.length > 2 ? parts[1] : ''
}

/**
 * Split a selection into batches the server will accept.
 *
 * Every batch carries the sidecars for its own top-level folder, plus any
 * root-level ones (a YOLO data.yaml). That repetition is the price of a
 * stateless endpoint: the server resolves each batch independently, so the
 * annotations describing a batch's images must arrive WITH them. Scoping by
 * folder keeps the cost bounded — a 25 MB train_coco.json rides along with
 * train's batches only, never with test's.
 */
export function planUploadBatches(files: File[]): File[][] {
  const images = files.filter((f) => !isSidecar(f))
  const sidecars = files.filter(isSidecar)

  // No folder structure and few enough files: one request, no repetition.
  if (images.length + sidecars.length <= BATCH_MAX_FILES) {
    const bytes = files.reduce((n, f) => n + f.size, 0)
    if (bytes <= BATCH_MAX_BYTES) return files.length ? [files] : []
  }

  const rootSidecars = sidecars.filter((f) => topFolder(f) === '')
  const sidecarsFor = (folder: string) => [
    ...sidecars.filter((f) => f !== undefined && topFolder(f) === folder && folder !== ''),
    ...rootSidecars,
  ]

  // Group by folder so each batch has one sidecar set, then fill batches.
  const byFolder = new Map<string, File[]>()
  for (const f of images) {
    const key = topFolder(f)
    const list = byFolder.get(key)
    if (list) list.push(f)
    else byFolder.set(key, [f])
  }

  const batches: File[][] = []
  for (const [folder, group] of byFolder) {
    const extras = sidecarsFor(folder)
    const extraBytes = extras.reduce((n, f) => n + f.size, 0)
    let current: File[] = []
    let bytes = 0
    for (const f of group) {
      const wouldOverflow =
        current.length + extras.length + 1 > BATCH_MAX_FILES ||
        (current.length > 0 && bytes + extraBytes + f.size > BATCH_MAX_BYTES)
      if (wouldOverflow) {
        batches.push([...current, ...extras])
        current = []
        bytes = 0
      }
      current.push(f)
      bytes += f.size
    }
    if (current.length) batches.push([...current, ...extras])
  }

  // A selection of nothing but sidecars still deserves to reach the server, so
  // it can answer "no images found in that folder" rather than silently doing
  // nothing.
  if (!batches.length && sidecars.length) batches.push(sidecars)
  return batches
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body) }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PUT', body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
  upload,
}

// --- Response types -------------------------------------------------------
// Hand-written to mirror the backend's Pydantic schemas. They're the contract
// between the two halves of the app; if they drift, TypeScript won't catch it
// (the backend isn't type-checked from here). Worth knowing: FastAPI publishes
// an OpenAPI spec at /openapi.json, so these can be code-generated later if the
// surface grows enough to make hand-maintenance annoying.

export interface HealthResponse {
  status: string
  app: string
  database: string
  storage_dir: string
}

export interface Project {
  id: number
  name: string
  description: string | null
  task_type: string
  created_at: string
  updated_at: string
  image_count: number
  class_count: number
  /** When anything in the project last changed — an upload, a saved dataset
   *  version, a training run, or an edit to the project itself. NOT
   *  `updated_at`, which only moves when the project ROW is edited. */
  last_activity_at: string | null
}

export interface ProjectClass {
  id: number
  project_id: number
  name: string
  /** Boxes using this class. Deleting the class cascades to all of them, so
   *  the confirm dialog can state exactly what is lost. */
  annotation_count: number
  color: string
  created_at: string
}

export interface DatasetImage {
  id: number
  project_id: number
  filename: string
  original_filename: string
  width: number
  height: number
  size_bytes: number
  created_at: string
  /** Relative path, e.g. /static/images/1/abc.jpg — usable directly in <img src>. */
  url: string
  /** Small cached JPEG for GRIDS and filmstrips — the scroll-lag fix. Use
   *  `url` only where one image is being worked on at full size. */
  thumb_url: string
  /** ACCEPTED boxes only — a proposal isn't an annotation. */
  annotation_count: number
  reviewed_count: number
  /** Pending model suggestions awaiting accept/reject. */
  proposed_count: number
  split: Split
}

export interface UploadResult {
  uploaded: DatasetImage[]
  skipped: string[]
  uploaded_count: number
  skipped_count: number

  // Populated when an uploaded zip turned out to be an annotated dataset.
  annotations_imported: number
  classes_created: string[]
  /** split name -> image count, e.g. {train: 700, val: 200, test: 100} */
  splits: Record<string, number>
  /** True when the archive used train/valid/test folders — i.e. the split came
   *  from the user's own data rather than being defaulted by us. */
  has_split_folders: boolean
  notes: string[]
  /** Imported train data with no validation set — prompt for a percentage. */
  needs_val_split: boolean
  /** Images already in the project byte-for-byte, so not added again.
   *  Re-uploading a folder used to silently double the dataset. */
  duplicates_skipped: number
  /** Groups every image from ONE upload across all its batches, so a partly
   *  failed import can be undone as a unit. */
  import_id: string | null
  /** Boxes written as PROPOSALS because their image was already in the project
   *  — a corrected export, or a second annotator's pass. They await
   *  Accept/Reject rather than overwriting existing work. */
  proposals_created: number
  reannotated_images: number
}

// --- Endpoints ------------------------------------------------------------
// Thin named wrappers rather than components calling api.get('/projects')
// inline. One place to change if a path moves, and the return types are stated
// once instead of at every call site.

export const health = () => api.get<HealthResponse>('/health')

export const listProjects = () => api.get<Project[]>('/projects')
export const getProject = (id: number) => api.get<Project>(`/projects/${id}`)
export const createProject = (body: { name: string; description?: string }) =>
  api.post<Project>('/projects', body)
/** Rename a project (or edit its description). 409 on a duplicate name. */
export const updateProject = (projectId: number, patch: { name?: string; description?: string | null }) =>
  api.patch<Project>(`/projects/${projectId}`, patch)
export const deleteProject = (id: number) => api.delete<void>(`/projects/${id}`)

export const listClasses = (projectId: number) =>
  api.get<ProjectClass[]>(`/projects/${projectId}/classes`)
export const createClass = (projectId: number, name: string) =>
  api.post<ProjectClass>(`/projects/${projectId}/classes`, { name })
export const deleteClass = (classId: number) => api.delete<void>(`/classes/${classId}`)

export const listImages = (projectId: number) =>
  api.get<DatasetImage[]>(`/projects/${projectId}/images`)

/** One page of images, plus how many there are in total.
 *
 *  The plain `listImages` above returns whatever the server's default page is,
 *  which is fine for callers that just want "some images" (Review, Visualize)
 *  and wrong for the grid, which is meant to show the dataset. */
/** Filters run SERVER-SIDE so they see the whole dataset, not the loaded page —
 *  a page-local filter once contradicted the whole-dataset stats banner. */
export interface ImageFilters {
  split?: 'train' | 'val' | 'test'
  state?: 'annotated' | 'unannotated' | 'pending'
  categoryId?: number
}

export async function listImagePage(
  projectId: number,
  limit: number,
  offset: number,
  filters: ImageFilters = {},
): Promise<{ images: DatasetImage[]; total: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (filters.split) params.set('split', filters.split)
  if (filters.state) params.set('state', filters.state)
  if (filters.categoryId != null) params.set('category_id', String(filters.categoryId))
  let res: Response
  try {
    res = await fetch(`${BASE_URL}/projects/${projectId}/images?${params}`)
  } catch (err) {
    throw describeNetworkFailure(err, 'the backend is not running')
  }
  if (!res.ok) throw new ApiError(await describeFailure(res), res.status)

  const images = (await res.json()) as DatasetImage[]
  // Absent header (an old backend, or a proxy that stripped it) degrades to
  // "what we can see", which is the previous behaviour rather than a crash.
  const total = Number(res.headers.get('X-Total-Count') ?? images.length)
  return { images, total: Number.isFinite(total) ? total : images.length }
}
/** Progress of a batched upload, for the panel's status line. */
export interface UploadProgress {
  batch: number
  batches: number
  filesSent: number
  filesTotal: number
}

/**
 * Upload a selection, in as many requests as it takes.
 *
 * Sequential, not parallel: the server writes every batch into the same project
 * and merges classes by name as it goes, so overlapping requests would race on
 * creating the same class. It also keeps the progress number honest and stops a
 * huge folder from opening six connections that each buffer 128 MB.
 *
 * A failing batch aborts the rest rather than pressing on. Continuing would
 * leave a half-imported dataset whose splits and class list are silently
 * incomplete, which is harder to notice — and to undo — than a clear failure
 * partway through, which the reported counts describe exactly.
 */
export async function uploadImages(
  projectId: number,
  files: File[],
  onProgress?: (p: UploadProgress) => void,
): Promise<UploadResult> {
  const batches = planUploadBatches(files)
  const imagesTotal = files.filter((f) => !SIDECAR_RE.test(f.name)).length
  // One id for the whole upload, generated up front so every batch shares it.
  // This is what makes a 27-batch folder undoable as a single action after one
  // of those batches fails.
  const importId = crypto.randomUUID().replace(/-/g, '').slice(0, 32)

  let merged: UploadResult | null = null
  let sent = 0

  for (let i = 0; i < batches.length; i++) {
    onProgress?.({
      batch: i + 1,
      batches: batches.length,
      filesSent: sent,
      filesTotal: imagesTotal,
    })
    const result = await api.upload<UploadResult>(
      `/projects/${projectId}/images`,
      batches[i],
      importId,
    )
    sent += batches[i].filter((f) => !SIDECAR_RE.test(f.name)).length
    merged = merged ? mergeUploadResults(merged, result) : result
  }

  onProgress?.({
    batch: batches.length,
    batches: batches.length,
    filesSent: sent,
    filesTotal: imagesTotal,
  })

  return (
    merged ?? {
      uploaded: [],
      skipped: [],
      uploaded_count: 0,
      skipped_count: 0,
      annotations_imported: 0,
      classes_created: [],
      splits: {},
      has_split_folders: false,
      notes: [],
      needs_val_split: false,
      duplicates_skipped: 0,
      import_id: importId,
      proposals_created: 0,
      reannotated_images: 0,
    }
  )
}

/** Fold one batch's result into the running total.
 *
 *  `uploaded` keeps only a bounded sample: a 5,000-image import would otherwise
 *  hold every row in memory to render a panel that shows a count. */
function mergeUploadResults(a: UploadResult, b: UploadResult): UploadResult {
  const splits = { ...a.splits }
  for (const [k, v] of Object.entries(b.splits)) splits[k] = (splits[k] ?? 0) + v
  return {
    uploaded: [...a.uploaded, ...b.uploaded].slice(0, 200),
    skipped: [...a.skipped, ...b.skipped],
    uploaded_count: a.uploaded_count + b.uploaded_count,
    skipped_count: a.skipped_count + b.skipped_count,
    annotations_imported: a.annotations_imported + b.annotations_imported,
    // Classes are created once and reused by later batches, so the union is the
    // set created across the whole import — not a per-batch tally.
    classes_created: [...new Set([...a.classes_created, ...b.classes_created])],
    splits,
    has_split_folders: a.has_split_folders || b.has_split_folders,
    notes: [...new Set([...a.notes, ...b.notes])],
    // Recomputed from the merged totals: a batch of only-train images says
    // "needs val" even when a later batch supplies one.
    needs_val_split: (splits.train ?? 0) > 0 && (splits.val ?? 0) === 0,
    duplicates_skipped: a.duplicates_skipped + b.duplicates_skipped,
    // Same across every batch by construction — see uploadImages.
    import_id: a.import_id ?? b.import_id,
    proposals_created: a.proposals_created + b.proposals_created,
    reannotated_images: a.reannotated_images + b.reannotated_images,
  }
}
export const deleteImage = (imageId: number) => api.delete<void>(`/images/${imageId}`)

/** What a project is holding on disk. Three "not needed" states, deliberately
 *  counted apart — see backend services/storage_audit.py. */
export interface StorageReport {
  total_images: number
  /** Live images in no saved version. NOT waste — this is where the whole
   *  upload → annotate → save workflow lives. The user decides. */
  unsaved_images: number
  /** Files nothing can reach. Safe to delete. */
  orphan_files: number
  orphan_bytes: number
  /** Files with no live row, kept because a version needs them. Deleting these
   *  would break restore. Shown so the disk usage is explicable. */
  retained_files: number
  retained_bytes: number
  /** Non-empty means the orphan figure is incomplete on purpose. */
  unreadable_versions: string[]
}

export const getStorageReport = (projectId: number) =>
  api.get<StorageReport>(`/projects/${projectId}/storage`)
export const reclaimStorage = (projectId: number) =>
  api.post<{ files_removed: number; bytes_freed: number }>(
    `/projects/${projectId}/storage/reclaim`,
  )
export const discardUnsavedImages = (projectId: number) =>
  api.post<{ deleted: number; bytes_freed: number }>(
    `/projects/${projectId}/storage/discard-unsaved`,
  )
/** Remove every image one upload added, however many batches it took. */
export const undoImport = (projectId: number, importId: string) =>
  api.post<{ deleted: number; kept_in_versions: number; bytes_freed: number }>(
    `/projects/${projectId}/imports/${importId}/undo`,
  )
/** Delete several images — also how "delete all" is sent, so selecting
 *  everything takes the same path as selecting one. `recoverable` is true when
 *  the project has a saved version, meaning the files were kept on disk. */
export const bulkDeleteImages = (projectId: number, imageIds: number[]) =>
  api.post<{ deleted: number; not_found: number[]; recoverable: boolean }>(
    `/projects/${projectId}/images/bulk-delete`,
    { image_ids: imageIds },
  )

// --- Auto-annotation (Phase 2) --------------------------------------------

export interface AnnotatorInfo {
  key: string
  display_name: string
  /** Architecture family ("Grounding DINO", "YOLO-World") and size within
   *  it — the picker's two axes, same as trainers. */
  family: string
  variant: string
  description: string
  approx_vram_gb: number
}

export interface DeviceInfo {
  available: boolean
  device: string
  name: string
  total_vram_gb: number | null
  compute_capability: string | null
  note: string | null
}

export interface AnnotationJob {
  id: number
  project_id: number
  model_key: string
  status: 'queued' | 'running' | 'done' | 'failed'
  /** Why a queued job hasn't started — the live "waiting for GPU" reason
   *  from the admission loop. Null once running. */
  status_detail: string | null
  total_images: number
  processed_images: number
  boxes_created: number
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress_pct: number
}

export interface Annotation {
  id: number
  image_id: number
  category_id: number
  x: number
  y: number
  width: number
  height: number
  confidence: number | null
  source: string
  reviewed: boolean
  /** True = a model suggestion awaiting accept/reject. Not a real annotation. */
  proposed: boolean
  job_id: number | null
}

/** A pending model batch. Accept or reject — there are no modes.
 *  (append/merge/replace still exist for the staging -> dataset commit, which
 *  is a different decision about images.) */
export interface ProposalPreview {
  proposed_boxes: number
  proposed_images: number
  /** Your boxes on the images this run covered — exactly what Accept deletes. */
  existing_on_proposed_images: number
  /** Your boxes on images the run never touched. Accept leaves these alone. */
  existing_elsewhere: number
}

export const getProposalCount = (projectId: number) =>
  api.get<{ proposed_boxes: number }>(`/projects/${projectId}/proposals/count`)
export const getProposalPreview = (projectId: number) =>
  api.get<ProposalPreview>(`/projects/${projectId}/proposals/preview`)
/** The model's boxes replace yours on the images it covered. */
export const acceptProposals = (projectId: number) =>
  api.post<{ accepted: number; deleted_existing: number }>(
    `/projects/${projectId}/proposals/accept`,
  )
/** Discard the batch; your boxes are untouched. */
export const rejectProposals = (projectId: number) =>
  api.delete<void>(`/projects/${projectId}/proposals`)
export const acceptImageProposals = (imageId: number) =>
  api.post<{ accepted: number; deleted_existing: number }>(
    `/images/${imageId}/proposals/accept`,
  )
export const rejectImageProposals = (imageId: number) =>
  api.delete<void>(`/images/${imageId}/proposals`)

export interface AnnotationSummary {
  total_images: number
  annotated_images: number
  unannotated_images: number
  /** ACCEPTED boxes only — proposals are counted separately. */
  total_boxes: number
  auto_boxes: number
  manual_boxes: number
  imported_boxes: number
  proposed_boxes: number
}

export interface ExportFormatInfo {
  key: string
  display_name: string
  description: string
}

/** The model list is fetched, never hardcoded — adding an annotator on the
 *  backend makes it appear here with no frontend change. */
export const listAnnotators = () => api.get<AnnotatorInfo[]>('/annotators')
export const getDevice = () => api.get<DeviceInfo>('/device')
export const listExportFormats = () => api.get<ExportFormatInfo[]>('/export-formats')

// --- ML setup (installing torch/transformers/ultralytics on demand) --------

export interface MlStatus {
  /** Whether the heavy ML stack is importable right now. */
  installed: boolean
  install: {
    status: 'idle' | 'running' | 'done' | 'failed'
    phase: string
    error: string | null
    /** Tail of pip's output, so a stuck install is visible. */
    log_tail: string[]
  }
  /** What an install WOULD download for this machine — the GPU-detected build. */
  plan: {
    gpu_detected: boolean
    driver_cuda: string | null
    torch_build: string
    torch_index_url: string
  }
}

export const getMlStatus = () => api.get<MlStatus>('/ml/status')
export const startMlInstall = () => api.post<MlStatus>('/ml/install')

// --- Inference playground (Deploy) -----------------------------------------

export interface DeployableModel {
  job_id: number
  trainer_key: string
  version: number
  label: string
  best_map: number | null
}

export interface PredictionBox {
  label: string
  confidence: number
  /** COCO-style absolute pixels: top-left + size. */
  x: number
  y: number
  width: number
  height: number
}

export interface PredictionResult {
  image_width: number
  image_height: number
  boxes: PredictionBox[]
}

export const listModels = (projectId: number) =>
  api.get<DeployableModel[]>(`/projects/${projectId}/models`)

// --- Evaluation (test-split mAP) -------------------------------------------

export interface PerClassAP {
  name: string
  ap: number | null
}

export interface EvaluationJob {
  id: number
  project_id: number
  training_job_id: number
  dataset_version_id: number
  split: string
  status: 'queued' | 'running' | 'done' | 'failed'
  num_images: number
  map_50_95: number | null
  map_50: number | null
  map_75: number | null
  per_class: PerClassAP[]
  /** Diagnostics beyond the headline. Null while running or for evaluations
   *  from before the field existed. */
  details: EvaluationDetails | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface EvaluationDetails {
  /** Per-class precision over recall at IoU 0.50 — COCOeval's own sweep. */
  pr_curves: { name: string; recall: number[]; precision: number[] }[]
  /** matrix[predicted][actual]; last index on both axes is "background"
   *  (a missed object or an invented one). */
  confusion: { classes: string[]; matrix: number[][] }
  /** Test images ranked by errors at conf 0.25 / IoU 0.45 — the ones to look at. */
  worst: {
    image_id: number
    filename: string
    original_filename: string
    fp: number
    fn: number
  }[]
}

export const listEvaluations = (projectId: number) =>
  api.get<EvaluationJob[]>(`/projects/${projectId}/evaluations`)

export const getEvaluation = (jobId: number) =>
  api.get<EvaluationJob>(`/evaluation-jobs/${jobId}`)

export const startEvaluation = (
  projectId: number,
  body: { training_job_id: number; dataset_version_id: number; split?: string },
) => api.post<EvaluationJob>(`/projects/${projectId}/evaluate`, body)

/** Run a model on one uploaded image. Nothing is stored — this is read-only. */
export async function predictImage(
  jobId: number,
  file: File,
  confThreshold: number,
): Promise<PredictionResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('conf_threshold', String(confThreshold))
  let res: Response
  try {
    res = await fetch(`${BASE_URL}/models/${jobId}/predict`, { method: 'POST', body: form })
  } catch (err) {
    throw describeNetworkFailure(err, 'the model may still be loading, or the backend stopped')
  }
  if (!res.ok) throw new ApiError(await describeFailure(res), res.status)
  return res.json() as Promise<PredictionResult>
}

/** Used only when no explicit image selection is given. */
export type JobScope = 'unannotated' | 'all'

export interface AnnotatePreview {
  auto_boxes: number
  manual_boxes: number
  imported_boxes: number
  scope_counts: Record<JobScope, number>
}

/** What a run would destroy. Auto-annotation is not additive. */
export const getAnnotatePreview = (projectId: number) =>
  api.get<AnnotatePreview>(`/projects/${projectId}/annotate/preview`)

export const startAnnotation = (
  projectId: number,
  body: {
    model_key: string
    box_threshold?: number
    text_threshold?: number
    prompts?: Record<string, string>
    /** Annotate exactly these. Takes precedence over `scope`. */
    image_ids?: number[]
    scope?: JobScope
  },
) => api.post<AnnotationJob>(`/projects/${projectId}/annotate`, body)

/** Cancel a queued/running annotation run — DISCARDS everything it produced.
 *  The job row itself is deleted, so the poller sees 404 as the expected end. */
export const cancelAnnotationJob = (jobId: number) =>
  api.post<{ status: string }>(`/jobs/${jobId}/cancel`)

export const getJob = (jobId: number) => api.get<AnnotationJob>(`/jobs/${jobId}`)
export const listJobs = (projectId: number) =>
  api.get<AnnotationJob[]>(`/projects/${projectId}/jobs`)

export const listAnnotations = (imageId: number) =>
  api.get<Annotation[]>(`/images/${imageId}/annotations`)
export const createAnnotation = (
  imageId: number,
  body: { category_id: number; x: number; y: number; width: number; height: number },
) => api.post<Annotation>(`/images/${imageId}/annotations`, body)
export const updateAnnotation = (
  id: number,
  body: Partial<{
    category_id: number
    x: number
    y: number
    width: number
    height: number
    reviewed: boolean
  }>,
) => api.patch<Annotation>(`/annotations/${id}`, body)
export const deleteAnnotation = (id: number) => api.delete<void>(`/annotations/${id}`)

/** One box in a bulk save. id present = existing box (maybe edited);
 *  id absent = drawn since the last save. */
export interface AnnotationBulkItem {
  id?: number
  category_id: number
  x: number
  y: number
  width: number
  height: number
}

/** The review page's Save: atomically set an image's accepted boxes to exactly
 *  this list. Boxes absent from it are deleted; proposals are never touched.
 *  Returns ALL of the image's boxes afterwards (including proposals), so the
 *  caller can rebuild its state from one response. 409 = the draft went stale
 *  (something else changed the image) — reload and re-apply. */
export const replaceAnnotations = (imageId: number, annotations: AnnotationBulkItem[]) =>
  api.put<Annotation[]>(`/images/${imageId}/annotations`, { annotations })

// --- Dataset health ---------------------------------------------------------

export interface DatasetHealth {
  total_images: number
  annotated_images: number
  total_boxes: number
  classes: { id: number; name: string; color: string; boxes: number; images: number }[]
  box_sizes: {
    /** COCO absolute-area buckets. */
    small: number
    medium: number
    large: number
    /** Boxes spanning under ~3% of the image's width — hard to learn. */
    tiny: number
    /** 10 bins of sqrt(box_area / image_area) in [0, 1]. */
    relative_hist: number[]
  }
  warnings: string[]
}

/** The dataset's SHAPE — class balance, box sizes, and named warnings.
 *  The answer to "why is my mAP low?" (it's usually the data). */
export const getDatasetHealth = (projectId: number) =>
  api.get<DatasetHealth>(`/projects/${projectId}/dataset/health`)

// --- Dataset stats & splits -----------------------------------------------
//
// The staging -> dataset commit is gone: accepting IS the commit, so there's no
// CommitMode, no preview, and no dialog asking a question whose answer was
// always yes. Every image is a dataset image; `split` is just a property.

export type Split = 'train' | 'val' | 'test'

export interface SplitCounts {
  train: number
  val: number
  test: number
}

export interface DatasetStats {
  total_images: number
  annotated_images: number
  unannotated_images: number
  splits: SplitCounts
  /** ACCEPTED boxes. Proposals are reported separately, not folded in. */
  total_boxes: number
  proposed_boxes: number
  proposed_images: number
}

export const getDatasetStats = (projectId: number) =>
  api.get<DatasetStats>(`/projects/${projectId}/dataset/stats`)

export const resplitDataset = (
  projectId: number,
  body: { train_pct: number; val_pct: number; test_pct: number; only_train?: boolean },
) => api.post<SplitCounts>(`/projects/${projectId}/dataset/split`, body)

export const setImageSplit = (imageId: number, split: Split) =>
  api.patch<{ id: number; split: string }>(`/images/${imageId}/split?split=${split}`)

/** Move a chosen set of images to one split — the manual counterpart to the
 *  percentage control, for when you know these specific images belong in val. */
export const setSplitForImages = (projectId: number, imageIds: number[], split: Split) =>
  api.post<SplitCounts>(`/projects/${projectId}/dataset/split-selected`, {
    image_ids: imageIds,
    split,
  })

// --- Training (Phase 4) ---------------------------------------------------
// Mirrors the auto-annotation shape: fetch the trainer list (never hardcode
// it), POST to start a background job, poll it while it runs. Same StatusBadge
// vocabulary as AnnotationJob, on purpose.

export interface TrainerInfo {
  key: string
  display_name: string
  /** Architecture family ("YOLO12", "RT-DETR") and size within it ("nano",
   *  "L") — the picker groups by these two axes. */
  family: string
  variant: string
  description: string
  approx_vram_gb: number
  export_format: string
  default_epochs: number
  default_batch_size: number
  default_image_size: number
}

/** One epoch on the loss/mAP curve. Any metric may be null for an epoch that
 *  didn't measure it — that's "not measured", not zero. */
export interface EpochPoint {
  epoch: number
  train_loss: number | null
  val_map: number | null
  val_map50: number | null
}

export interface TrainingJob {
  id: number
  project_id: number
  trainer_key: string
  /** 1-based version within this project + model — what the UI shows, rather
   *  than the global row `id`. */
  version: number
  /** User-given name; null means it displays as "v{version}". */
  name: string | null
  status: 'queued' | 'running' | 'done' | 'failed'
  /** Why a queued job hasn't started — the live "waiting for GPU" reason
   *  from the admission loop. Null once running. */
  status_detail: string | null
  epochs: number
  batch_size: number
  image_size: number
  learning_rate: number | null
  train_images: number
  val_images: number
  num_classes: number
  /** Set when this run continued another run's checkpoint (finetune). */
  init_from_job_id: number | null
  /** The saved dataset version this run trained on. */
  dataset_version_id: number | null
  /** "stop" | "cancel" once requested — the run is winding down. */
  control: string | null
  /** True when the run ended because the user stopped it short. */
  stopped_early: boolean
  current_epoch: number
  total_epochs: number
  train_loss: number | null
  /** Latest validation mAP@.50:.95. */
  val_map: number | null
  /** Best mAP across all epochs — the checkpoint we keep. */
  best_map: number | null
  checkpoint_path: string | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress_pct: number
  /** Per-epoch history for the curve. Always present (possibly empty). */
  metrics: EpochPoint[]
}

/** Dataset readiness — answered before the click so a doomed run is never
 *  launched. */
export interface TrainPreview {
  num_classes: number
  splits: Record<Split, { images: number; boxes: number }>
  /** Training is gated on the dataset having been saved at least once. */
  has_saved_version: boolean
  latest_version: number | null
  latest_version_id: number | null
  /** The version the live dataset matches; null = unsaved changes. */
  current_version: number | null
  current_version_id: number | null
  has_unsaved_changes: boolean
  can_train: boolean
  warnings: string[]
}

/** Fetched, never hardcoded — registering a trainer on the backend makes it
 *  appear here. Empty until the Phase 4 training deps are installed. */
export const listTrainers = () => api.get<TrainerInfo[]>('/trainers')
export const getTrainPreview = (projectId: number) =>
  api.get<TrainPreview>(`/projects/${projectId}/train/preview`)
export const startTraining = (
  projectId: number,
  body: {
    trainer_key: string
    epochs?: number
    batch_size?: number
    image_size?: number
    learning_rate?: number | null
    /** Continue/finetune from this completed run's checkpoint. */
    init_from_job_id?: number | null
    /** Which saved dataset version to train. Omit for the latest save. */
    dataset_version_id?: number | null
  },
) => api.post<TrainingJob>(`/projects/${projectId}/train`, body)
export const getTrainingJob = (jobId: number) =>
  api.get<TrainingJob>(`/training-jobs/${jobId}`)
export const listTrainingJobs = (projectId: number) =>
  api.get<TrainingJob[]>(`/projects/${projectId}/training-jobs`)
/** Live log tail for a run — the framework's narration plus one line per
 *  epoch. In-memory on the server: gone after a restart, unlike metrics. */
export const getTrainingLogs = (jobId: number) =>
  api.get<{ lines: string[] }>(`/training-jobs/${jobId}/logs`)

// --- Dataset versions -----------------------------------------------------
// Save points for the dataset. Created only by "Save dataset", which is also the
// gate into training — you train a saved version, so a run stays reproducible.

export interface DatasetVersion {
  id: number
  project_id: number
  /** 1-based per project. */
  version: number
  /** User-given name; null means it displays as "v{version}". */
  name: string | null
  note: string | null
  total_images: number
  train_images: number
  val_images: number
  test_images: number
  total_boxes: number
  /** Boxes in the train split — what a run actually learns from. */
  train_boxes: number
  num_classes: number
  created_at: string
  /** True for the version the LIVE dataset matches — not always the newest.
   *  After restoring an older version, that older one is current. */
  is_current: boolean
}

/** What a restore actually did. `missing_files` non-empty = partial restore. */
export interface RestoreResult {
  restored_version: number
  images_restored: number
  boxes_restored: number
  images_removed: number
  missing_files: string[]
  /** Classes that existed only after the restored version, removed to rewind the
   *  class list. Non-empty means pending proposals using them went too. */
  classes_removed: string[]
}

export const listDatasetVersions = (projectId: number) =>
  api.get<DatasetVersion[]>(`/projects/${projectId}/dataset/versions`)
export const saveDatasetVersion = (projectId: number, note?: string) =>
  api.post<DatasetVersion>(`/projects/${projectId}/dataset/versions`, { note: note ?? null })
export const restoreDatasetVersion = (projectId: number, versionId: number) =>
  api.post<RestoreResult>(`/projects/${projectId}/dataset/versions/${versionId}/restore`)
/** Blank/undefined clears the name, reverting to "v{n}". 409 on a duplicate. */
export const renameDatasetVersion = (projectId: number, versionId: number, name: string | null) =>
  api.patch<DatasetVersion>(`/projects/${projectId}/dataset/versions/${versionId}`, { name })
export const deleteDatasetVersion = (projectId: number, versionId: number) =>
  api.delete<void>(`/projects/${projectId}/dataset/versions/${versionId}`)
/** Also how "delete all" is sent — same path, every id selected. */
export const bulkDeleteDatasetVersions = (projectId: number, versionIds: number[]) =>
  api.post<{ deleted: number; not_found: number[] }>(
    `/projects/${projectId}/dataset/versions/bulk-delete`,
    { version_ids: versionIds },
  )

/** Stop early, keeping the model trained so far. The epoch in flight finishes. */
export const stopTrainingJob = (jobId: number) =>
  api.post<TrainingJob>(`/training-jobs/${jobId}/stop`)
/** Cancel outright — no version kept, output discarded. The row disappears. */
export const cancelTrainingJob = (jobId: number) =>
  api.post<void>(`/training-jobs/${jobId}/cancel`)

/** Model-version housekeeping. Uniqueness is per project + trainer. */
export const renameTrainingJob = (jobId: number, name: string | null) =>
  api.patch<TrainingJob>(`/training-jobs/${jobId}`, { name })
export const deleteTrainingJob = (jobId: number) => api.delete<void>(`/training-jobs/${jobId}`)
export const bulkDeleteTrainingJobs = (projectId: number, jobIds: number[]) =>
  api.post<{ deleted: number; not_found: number[]; skipped: Record<string, string> }>(
    `/projects/${projectId}/training-jobs/bulk-delete`,
    { job_ids: jobIds },
  )

/** How a version presents in a list: its name, else its number. */
export const versionLabel = (v: { name: string | null; version: number }) =>
  v.name ?? `v${v.version}`

export const bulkDeleteProjects = (projectIds: number[]) =>
  api.post<{ deleted: number; not_found: number[] }>('/projects/bulk-delete', {
    project_ids: projectIds,
  })
export const getAnnotationSummary = (projectId: number) =>
  api.get<AnnotationSummary>(`/projects/${projectId}/annotations/summary`)

/** Export is a plain link, not a fetch: letting the browser navigate to the URL
 *  gets the native download UI and streaming for free, where fetch would buffer
 *  the whole zip into memory first. */
export const exportUrl = (
  projectId: number,
  format: string,
  includeUnreviewed = true,
  content: 'full' | 'annotations' | 'images' = 'full',
) =>
  `/api/projects/${projectId}/export?format=${format}&include_unreviewed=${includeUnreviewed}&content=${content}`

/** Trained weights (.pt) download — also a plain link, same reasoning. */
export const weightsUrl = (jobId: number) => `/api/models/${jobId}/weights`
/** ONNX export. NOT a plain link like the .pt: the first request per
 *  checkpoint CONVERTS (up to a minute), so the button needs a busy state
 *  and an error surface — hence a fetch, blob and a client-side download. */
export const onnxUrl = (jobId: number) => `/api/models/${jobId}/onnx`
