import { useState } from 'react'

/**
 * "Images per page" selector: fixed sensible steps plus a custom entry.
 *
 * The backend clamps limit to 1000, so custom values are clamped here too —
 * offering an input that silently does less than it says would be worse than
 * not offering it.
 */
const CHOICES = [50, 100, 200, 500, 1000]
const MAX = 1000

export function PageSizeSelect({
  value,
  onChange,
}: {
  value: number
  onChange: (size: number) => void
}) {
  const [customOpen, setCustomOpen] = useState(false)
  const [draft, setDraft] = useState(String(value))

  const commitCustom = () => {
    const n = Math.max(1, Math.min(MAX, Math.floor(Number(draft)) || value))
    setCustomOpen(false)
    if (n !== value) onChange(n)
  }

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
      per page:
      {customOpen ? (
        <input
          autoFocus
          type="number"
          min={1}
          max={MAX}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitCustom}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitCustom()
            if (e.key === 'Escape') setCustomOpen(false)
          }}
          className="w-16 rounded border border-gray-300 px-1.5 py-1 tabular-nums"
        />
      ) : (
        <select
          value={CHOICES.includes(value) ? String(value) : 'custom'}
          onChange={(e) => {
            if (e.target.value === 'custom') {
              setDraft(String(value))
              setCustomOpen(true)
            } else {
              onChange(Number(e.target.value))
            }
          }}
          className="rounded border border-gray-300 bg-white px-1.5 py-1 text-gray-700"
        >
          {CHOICES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
          {/* A non-standard current value (set via custom) still shows itself. */}
          {!CHOICES.includes(value) && <option value="custom">{value}</option>}
          <option value="custom">custom…</option>
        </select>
      )}
    </span>
  )
}
