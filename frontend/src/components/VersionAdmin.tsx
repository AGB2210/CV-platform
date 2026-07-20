import { useState, type ReactNode } from 'react'
import { Check, Trash2 } from 'lucide-react'
import { Modal } from '@/components/ui/Modal'

/**
 * Shared housekeeping for the two version lists (datasets and models).
 *
 * The rows themselves are genuinely different — one offers Restore, the other a
 * status badge and click-to-inspect — so this deliberately does NOT try to be a
 * single generic list component. It shares the parts that would otherwise be
 * copied and drift: the selection toolbar, the row checkbox/actions, and the
 * rename dialog with its duplicate-name handling. The selection STATE lives in
 * lib/useVersionSelection so this file exports only components (Fast Refresh).
 */

/**
 * Appears only once something is selected — a permanently visible "Delete 0"
 * is noise, and a delete button sitting next to a list you haven't chosen from
 * invites misclicks.
 */
export function SelectionToolbar({
  count,
  total,
  onToggleAll,
  onDelete,
  busy,
}: {
  count: number
  total: number
  onToggleAll: () => void
  onDelete: () => void
  busy?: boolean
}) {
  if (count === 0) return null
  return (
    <div className="flex items-center justify-between gap-2 border-b border-accent-200 bg-accent-50 px-3 py-1.5 text-xs">
      <span className="text-accent-900">
        <span className="font-medium tabular-nums">{count}</span> selected
      </span>
      <span className="flex items-center gap-2">
        <button onClick={onToggleAll} className="text-accent-800 hover:underline">
          {count === total ? 'Clear' : `Select all ${total}`}
        </button>
        <button
          onClick={onDelete}
          disabled={busy}
          className="flex items-center gap-1 rounded bg-red-600 px-1.5 py-0.5 font-medium text-white hover:bg-red-700 disabled:opacity-60"
        >
          <Trash2 size={11} />
          {busy ? 'Deleting…' : 'Delete'}
        </button>
      </span>
    </div>
  )
}

/** A row's checkbox. Stops propagation so ticking it never also triggers a
 *  row-level click (the model list selects a version for the detail panel). */
export function RowCheckbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: () => void
  label: string
}) {
  return (
    <input
      type="checkbox"
      checked={checked}
      aria-label={label}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => {
        e.stopPropagation()
        onChange()
      }}
      className="mt-0.5 shrink-0 accent-accent-600"
    />
  )
}

/**
 * Rename dialog.
 *
 * Owns its own error state so a rejected name (the server refuses duplicates)
 * is shown against the field being edited, rather than as a page-level banner
 * far from the input that caused it. Submitting an empty value clears the name,
 * which is how a rename is undone — stated in the hint rather than hidden.
 */
export function RenameDialog({
  open,
  currentName,
  fallbackLabel,
  onClose,
  onSave,
}: {
  open: boolean
  currentName: string | null
  /** What it displays as when unnamed, e.g. "v3". */
  fallbackLabel: string
  onClose: () => void
  onSave: (name: string | null) => Promise<void>
}) {
  const [value, setValue] = useState(currentName ?? '')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Re-seed when the dialog opens for a different row.
  const [seededFor, setSeededFor] = useState<string | null>(null)
  const seed = `${fallbackLabel}:${currentName ?? ''}`
  if (open && seededFor !== seed) {
    setSeededFor(seed)
    setValue(currentName ?? '')
    setError(null)
  }

  async function save() {
    setBusy(true)
    setError(null)
    try {
      await onSave(value.trim() || null)
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={`Rename ${fallbackLabel}`}
      footer={
        <>
          <button className="btn-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn-primary" onClick={() => void save()} disabled={busy}>
            <Check size={13} />
            {busy ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      <label className="mb-1 block text-xs font-medium text-gray-700" htmlFor="rename-input">
        Name
      </label>
      <input
        id="rename-input"
        value={value}
        autoFocus
        maxLength={120}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') void save()
        }}
        placeholder={fallbackLabel}
        className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm placeholder:text-gray-300 focus:border-accent-500 focus:outline-none"
      />
      <p className="mt-1 text-xs text-gray-400">
        Leave empty to go back to <span className="font-mono">{fallbackLabel}</span>. Names
        must be unique.
      </p>
      {error && (
        <p className="mt-2 rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-800">
          {error}
        </p>
      )}
    </Modal>
  )
}

/** Small icon button used for per-row actions (rename, delete, restore). */
export function RowAction({
  onClick,
  title,
  children,
  danger,
}: {
  onClick: () => void
  title: string
  children: ReactNode
  danger?: boolean
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      title={title}
      aria-label={title}
      className={`rounded p-0.5 ${
        danger
          ? 'text-gray-400 hover:bg-red-50 hover:text-red-600'
          : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'
      }`}
    >
      {children}
    </button>
  )
}
