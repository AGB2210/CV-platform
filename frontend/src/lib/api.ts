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

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
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

export const health = () => api.get<HealthResponse>('/health')
