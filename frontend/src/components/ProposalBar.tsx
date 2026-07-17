import { useCallback, useEffect, useState } from 'react'
import { Check, Sparkles, Trash2, X } from 'lucide-react'
import {
  applyProposals,
  discardProposals,
  getProposalPreview,
  type ApplyMode,
  type ProposalPreview,
} from '@/lib/api'

/**
 * The pending model batch, and the decision about it.
 *
 * Auto-annotation doesn't change your annotations — it proposes. This bar is
 * where a batch of proposals becomes (or doesn't become) real data.
 *
 * It only exists while a batch is pending, so it's not permanent furniture:
 * apply or discard it and the bar goes away.
 */

const MODES: { value: ApplyMode; label: string; blurb: (p: ProposalPreview) => string }[] = [
  {
    value: 'append',
    label: 'Append',
      blurb: (p) =>
      `Accept all ${p.proposed_boxes} proposal${
        p.proposed_boxes === 1 ? '' : 's'
      } alongside your boxes.`,
  },
  {
    value: 'merge',
    label: 'Merge',
    blurb: (p) =>
      p.conflicting_images === 0
        ? 'Only fill in empty images. None of these images have boxes yet, so this behaves like Append.'
        : `Only fill in empty images. ${p.conflicting_images} image${
            p.conflicting_images === 1 ? ' already has' : 's already have'
          } boxes — ${
            p.conflicting_images === 1 ? 'it keeps' : 'they keep'
          } yours and the proposals for ${
            p.conflicting_images === 1 ? 'it' : 'them'
          } are dropped.`,
  },
  {
    value: 'replace',
    label: 'Replace',
    // Uses existing_on_proposed_images, NOT would_delete_existing: the latter
    // is computed for whichever mode is SELECTED, so reading it here made
    // Replace claim it deletes nothing whenever Merge happened to be ticked.
    // Each mode must describe itself, not the current selection.
    blurb: (p) =>
      p.existing_on_proposed_images === 0
        ? `Proposals win on the ${p.proposed_images} image${
            p.proposed_images === 1 ? '' : 's'
          } this run covered. You have no boxes on them, so nothing is deleted.`
        : `Proposals win on the ${p.proposed_images} image${
            p.proposed_images === 1 ? '' : 's'
          } this run covered. ${p.existing_on_proposed_images} of your box${
            p.existing_on_proposed_images === 1 ? '' : 'es'
          } on them will be deleted.`,
  },
]

export function ProposalBar({
  projectId,
  proposedBoxes,
  onChanged,
}: {
  projectId: number
  proposedBoxes: number
  onChanged: () => void
}) {
  const [mode, setMode] = useState<ApplyMode>('append')
  const [preview, setPreview] = useState<ProposalPreview | null>(null)
  const [busy, setBusy] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setPreview(await getProposalPreview(projectId, mode))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [projectId, mode])

  // Re-preview on mode change: replace's numbers are nothing like append's, and
  // stale counts next to a destructive button is how people lose work.
  useEffect(() => {
    if (proposedBoxes > 0) void load()
  }, [load, proposedBoxes])

  if (proposedBoxes === 0) return null

  /**
   * Apply the batch in `which` mode.
   *
   * The mode is a PARAMETER, not read from state. It used to read `mode` from
   * the closure, and "Accept all" did `setMode('append'); apply()` — but
   * setState is asynchronous, so apply() still saw the PREVIOUS mode. Select
   * Replace, click "Accept all", and it silently ran replace and deleted your
   * boxes under a button that said "Accept all".
   *
   * Passing it explicitly makes the bug structurally impossible: the caller
   * states which mode it means, and there is no stale value to read.
   */
  async function apply(which: ApplyMode) {
    setBusy(true)
    setError(null)
    try {
      await applyProposals(projectId, which)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function discard() {
    setBusy(true)
    try {
      await discardProposals(projectId)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const destructive = mode === 'replace' && (preview?.would_delete_existing ?? 0) > 0

  return (
    <div className="border-b border-accent-200 bg-accent-50">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-2">
        <Sparkles size={14} className="shrink-0 text-accent-700" />
        <span className="text-xs text-accent-900">
          <span className="font-medium">
            {proposedBoxes} model proposal{proposedBoxes === 1 ? '' : 's'}
          </span>
          {preview && (
            <span className="text-accent-800">
              {' '}
              across {preview.proposed_images} image
              {preview.proposed_images === 1 ? '' : 's'} — not part of your dataset
              until you accept them
              {/* Say up front that accepting KEEPS your boxes. "Accept all" on
                  its own reads like it might overwrite them, and the only way
                  to find out was to click it. */}
              {preview.existing_on_proposed_images > 0 && (
                <>
                  . Accepting keeps your {preview.existing_on_proposed_images} existing
                  box
                  {preview.existing_on_proposed_images === 1 ? '' : 'es'} — use{' '}
                  <span className="font-medium">Apply batch</span> to replace them instead
                </>
              )}
            </span>
          )}
        </span>

        <button
          className="text-xs font-medium text-accent-800 underline underline-offset-2"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? 'Hide options' : 'Apply batch…'}
        </button>

        <div className="ml-auto flex items-center gap-2">
          <button className="btn-secondary" onClick={() => void discard()} disabled={busy}>
            <Trash2 size={13} />
            Discard
          </button>
          <button
            className="btn-primary"
            // Always append, whatever radio happens to be selected below. The
            // shortcut names its own mode rather than inheriting one.
            onClick={() => void apply('append')}
            disabled={busy}
            title="Accept every proposal, keeping your existing boxes"
          >
            <Check size={13} />
            Accept all {proposedBoxes}
          </button>
        </div>
      </div>

      {expanded && preview && (
        <div className="border-t border-accent-200 px-4 py-3">
          <div className="grid gap-1.5 sm:grid-cols-3">
            {MODES.map((m) => (
              <label
                key={m.value}
                className={`flex cursor-pointer gap-2 rounded-md border bg-white p-2 transition-colors ${
                  mode === m.value
                    ? 'border-accent-500 ring-1 ring-accent-500'
                    : 'border-gray-200 hover:bg-gray-50'
                }`}
              >
                <input
                  type="radio"
                  name="apply-mode"
                  checked={mode === m.value}
                  onChange={() => setMode(m.value)}
                  className="mt-0.5 accent-accent-600"
                />
                <span className="min-w-0">
                  <span className="block text-sm font-medium text-gray-900">{m.label}</span>
                  <span className="block text-xs text-gray-500">{m.blurb(preview)}</span>
                </span>
              </label>
            ))}
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <Stat label="Accept" value={preview.would_accept} />
            {preview.would_discard > 0 && (
              <Stat label="Discard" value={preview.would_discard} muted />
            )}
            {preview.would_delete_existing > 0 && (
              <Stat label="Delete YOUR boxes" value={preview.would_delete_existing} danger />
            )}

            <button
              className={
                destructive
                  ? 'btn ml-auto bg-red-600 text-white hover:bg-red-700'
                  : 'btn-primary ml-auto'
              }
              onClick={() => void apply(mode)}
              disabled={busy}
            >
              {busy ? 'Applying…' : `Apply ${mode}`}
            </button>
          </div>

          {/* Replace deletes your work. Name the number — a generic warning
              trains people to click through it. */}
          {destructive && (
            <p className="mt-2 rounded-md border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs text-red-800">
              <span className="font-medium">
                {preview.would_delete_existing} of your box
                {preview.would_delete_existing === 1 ? '' : 'es'} will be permanently
                deleted
              </span>{' '}
              on the images this run covered. Images it didn't touch are unaffected.
            </p>
          )}
        </div>
      )}

      {error && (
        <p className="border-t border-red-200 bg-red-50 px-4 py-1.5 text-xs text-red-800">
          {error}
          <button className="ml-2 underline" onClick={() => setError(null)}>
            <X size={11} className="inline" />
          </button>
        </p>
      )}
    </div>
  )
}

function Stat({
  label,
  value,
  muted,
  danger,
}: {
  label: string
  value: number
  muted?: boolean
  danger?: boolean
}) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className={danger ? 'text-red-700' : muted ? 'text-gray-400' : 'text-gray-600'}>
        {label}
      </span>
      <span
        className={`font-mono font-medium tabular-nums ${
          danger ? 'text-red-700' : muted ? 'text-gray-400' : 'text-gray-900'
        }`}
      >
        {value}
      </span>
    </span>
  )
}
