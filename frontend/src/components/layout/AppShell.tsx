import type { ReactNode } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

/**
 * The persistent application frame: sidebar + page header + scrollable content.
 *
 * Router pages render into <Outlet />, so navigating swaps only the content
 * region — the sidebar never unmounts or flickers. Every page gets the same
 * chrome for free.
 */
export function AppShell() {
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
