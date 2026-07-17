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

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })

  if (!res.ok) {
    // FastAPI reports errors as {"detail": "..."} by convention. Fall back to
    // the status text when the body isn't JSON (e.g. a proxy error page), so a
    // dead backend produces a readable message instead of a JSON parse crash.
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* non-JSON body — keep statusText */
    }
    throw new ApiError(detail, res.status)
  }

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
async function upload<T>(path: string, files: File[]): Promise<T> {
  const form = new FormData()
  // Field name must be 'files' — it matches the `files: list[UploadFile]`
  // parameter in the FastAPI endpoint.
  for (const file of files) form.append('files', file)

  const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', body: form })

  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* non-JSON body */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body) }),
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
}

export interface ProjectClass {
  id: number
  project_id: number
  name: string
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
  annotation_count: number
  reviewed_count: number
}

export interface UploadResult {
  uploaded: DatasetImage[]
  skipped: string[]
  uploaded_count: number
  skipped_count: number
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
export const deleteProject = (id: number) => api.delete<void>(`/projects/${id}`)

export const listClasses = (projectId: number) =>
  api.get<ProjectClass[]>(`/projects/${projectId}/classes`)
export const createClass = (projectId: number, name: string) =>
  api.post<ProjectClass>(`/projects/${projectId}/classes`, { name })
export const deleteClass = (classId: number) => api.delete<void>(`/classes/${classId}`)

export const listImages = (projectId: number) =>
  api.get<DatasetImage[]>(`/projects/${projectId}/images`)
export const uploadImages = (projectId: number, files: File[]) =>
  api.upload<UploadResult>(`/projects/${projectId}/images`, files)
export const deleteImage = (imageId: number) => api.delete<void>(`/images/${imageId}`)

// --- Auto-annotation (Phase 2) --------------------------------------------

export interface AnnotatorInfo {
  key: string
  display_name: string
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
}

export interface AnnotationSummary {
  total_images: number
  annotated_images: number
  unannotated_images: number
  total_boxes: number
  auto_boxes: number
  manual_boxes: number
  reviewed_boxes: number
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

export const startAnnotation = (
  projectId: number,
  body: {
    model_key: string
    box_threshold?: number
    text_threshold?: number
    prompts?: Record<string, string>
  },
) => api.post<AnnotationJob>(`/projects/${projectId}/annotate`, body)

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
export const approveImage = (imageId: number) =>
  api.post<Annotation[]>(`/images/${imageId}/annotations/approve`)
export const getAnnotationSummary = (projectId: number) =>
  api.get<AnnotationSummary>(`/projects/${projectId}/annotations/summary`)

/** Export is a plain link, not a fetch: letting the browser navigate to the URL
 *  gets the native download UI and streaming for free, where fetch would buffer
 *  the whole zip into memory first. */
export const exportUrl = (projectId: number, format: string, includeUnreviewed = true) =>
  `/api/projects/${projectId}/export?format=${format}&include_unreviewed=${includeUnreviewed}`
