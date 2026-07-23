/**
 * Status indicator.
 *
 * Built in Phase 0 despite only having one caller, because from Phase 2 onward
 * every job (auto-annotation, training, evaluation) reports the same lifecycle.
 * Defining the vocabulary once means "running" looks identical everywhere
 * instead of each page inventing its own amber.
 */

export type Status = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'unknown'

// Map lifecycle state -> presentation. Adding a state means adding one entry
// here, not editing JSX in five files.
const STYLES: Record<Status, { dot: string; text: string; label: string }> = {
  queued: { dot: 'bg-status-idle', text: 'text-gray-600', label: 'Queued' },
  running: { dot: 'bg-status-busy', text: 'text-amber-700', label: 'Running' },
  done: { dot: 'bg-status-good', text: 'text-green-700', label: 'Done' },
  failed: { dot: 'bg-status-bad', text: 'text-red-700', label: 'Failed' },
  // Neutral, not red: the user asked for this outcome. Red would read as
  // "something went wrong" — the exact confusion this state exists to end.
  cancelled: { dot: 'bg-status-idle', text: 'text-gray-500', label: 'Cancelled' },
  unknown: { dot: 'bg-gray-300', text: 'text-gray-400', label: 'Unknown' },
}

export function StatusBadge({ status, label }: { status: Status; label?: string }) {
  const style = STYLES[status]
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${style.text}`}>
      <span
        // A small dot rather than a filled pill: at this size a pill per row
        // adds visual weight without adding information.
        className={`h-1.5 w-1.5 shrink-0 rounded-full ${style.dot} ${
          status === 'running' ? 'animate-pulse' : ''
        }`}
      />
      {label ?? style.label}
    </span>
  )
}
