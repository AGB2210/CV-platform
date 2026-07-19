import { useEffect, useState, type ReactNode } from 'react'
import { Outlet } from 'react-router-dom'
import { Minimize2 } from 'lucide-react'
import { Sidebar } from './Sidebar'

/**
 * Below this viewport width the dense multi-column layouts can't fit without
 * overlapping — the worst case is the Review screen, where the sidebar,
 * filmstrip, right panel and a ~340px-wide action header together need this
 * much room before the accept/reject buttons stop colliding with the right
 * panel. Measured empirically: the Review header clears the panel at ~1216px,
 * so 1280 gives a little margin and is a conventional desktop minimum.
 *
 * It's a single knob on purpose. Lower it and you trade a little button overlap
 * on Review for supporting narrower windows; the guard below reads it live.
 */
const MIN_SUPPORTED_WIDTH = 1280

/** Live viewport width. SPA only (no SSR), so reading `window` at init is safe. */
function useViewportWidth() {
  const [width, setWidth] = useState(() => window.innerWidth)
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return width
}

/**
 * The persistent application frame: sidebar + page header + scrollable content.
 *
 * Router pages render into <Outlet />, so navigating swaps only the content
 * region — the sidebar never unmounts or flickers. Every page gets the same
 * chrome for free.
 */
export function AppShell() {
  const width = useViewportWidth()
  const tooNarrow = width < MIN_SUPPORTED_WIDTH

  return (
    // h-screen + overflow-hidden on the frame, overflow-auto on the content
    // region: the sidebar and header stay pinned while only the page body
    // scrolls. Standard app-shell layout, and it avoids the whole-window scroll
    // that makes a dense tool feel like a document.
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        <Outlet />
      </main>
      {/* The guard is an OVERLAY, not a replacement: the app underneath stays
          mounted, so nothing loses its state. Widen the window back and the
          overlay simply unmounts, revealing the exact place you left off — a
          half-drawn box, the image you were on, your scroll position. */}
      {tooNarrow && <ViewportGuard width={width} min={MIN_SUPPORTED_WIDTH} />}
    </div>
  )
}

/**
 * Shown while the window is narrower than {@link MIN_SUPPORTED_WIDTH}. Covers
 * everything (fixed inset-0) so the cramped layout behind it is never seen; it
 * doesn't unmount the app, so resizing back is instant and lossless.
 */
function ViewportGuard({ width, min }: { width: number; min: number }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-white p-6">
      <div className="max-w-sm text-center">
        <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-gray-100">
          <Minimize2 size={20} className="text-gray-500" />
        </div>
        <h1 className="text-sm font-semibold text-gray-900">Window too narrow</h1>
        <p className="mt-1.5 text-sm text-gray-600">
          This tool lays out several panels side by side and needs at least{' '}
          <span className="font-medium text-gray-900">{min}px</span> of width to
          do it without overlap. Widen or maximise the window to continue — your
          place is kept.
        </p>
        <p className="mt-4 font-mono text-xs tabular-nums text-gray-400">
          {width}px / {min}px needed
        </p>
      </div>
    </div>
  )
}

/**
 * Page header. Each page renders one so the title bar height and typography
 * stay identical across the app.
 */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: string
  /** Right-aligned slot for primary actions, e.g. a "New project" button. */
  actions?: ReactNode
}) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-6">
      <div>
        <h1 className="text-sm font-semibold text-gray-900">{title}</h1>
        {description && <p className="text-xs text-gray-500">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}

/** Scrollable content region below the page header. */
export function PageBody({ children }: { children: ReactNode }) {
  return <div className="flex-1 overflow-auto p-6">{children}</div>
}
