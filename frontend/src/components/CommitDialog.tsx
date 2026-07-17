import { useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Modal } from '@/components/ui/Modal'
import {
  commitToDataset,
  getCommitPreview,
  type CommitMode,
  type CommitPreview,
} from '@/lib/api'

/**
 * "Add to Dataset" — the staging -> dataset commit step.
 *
 * Roboflow's model: annotated images sit in a staging batch until you
 * deliberately move them into the trainable dataset. The value is that
 * half-reviewed work can never drift into a training run by accident.
 *
 * Only APPROVED images move, which is why the preview separates approved from
 * unapproved rather than just showing a total — "12 of 30 will be added" is the
 * number you need before clicking, not "30 staged".
 */

const MODES: { value: CommitMode; label: string; blurb: string }[] = [
  {
    value: 'append',
    label: 'Append',
    blurb: 'Add these to the dataset. Existing images are untouched.',
  },
  {
    value: 'merge',
    label: 'Merge',
    blurb:
      'Add these, but if a filename already exists in the dataset, fold the boxes into that image instead of creating a duplicate.',
  },
  {
    value: 'replace',
    label: 'Replace',
    blurb: 'These become the entire dataset. Everything currently in it is deleted.',
  },
]

export function CommitDialog({
  open,
  projectId,
  onClose,
  onCommitted,
}: {
  open: boolean
  projectId: number
  onClose: () => void
  onCommitted: () => void
}) {
  const [mode, setMode] = useState<CommitMode>('append')
  const [preview, setPreview] = useState<CommitPreview | null>(null)
  const [assignSplits, setAssignSplits] = useState(true)
  const [trainPct, setTrainPct] = useState(80)
  const [valPct, setValPct] = useState(20)
  const [testPct, setTestPct] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Re-preview whenever the mode changes: replace's numbers are wildly
  // different from append's, and showing stale counts next to a destructive
  // button is how people delete things they didn't mean to.
  useEffect(() => {
    if (!open) return
    let cancelled = false
    getCommitPreview(projectId, mode)
      .then((p) => !cancelled && setPreview(p))
      .catch((e: Error) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [open, projectId, mode])

  const total = trainPct + valPct + testPct
  const pctValid = total === 100

  async function submit() {
    setBusy(true)
    setError(null)
    try {
      await commitToDataset(projectId, {
        mode,
        assign_splits: assignSplits,
        train_pct: trainPct / 100,
        val_pct: valPct / 100,
        test_pct: testPct / 100,
      })
      onCommitted()
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const nothingToAdd = preview !== null && preview.would_add === 0

  return (
    <Modal open={open} onClose={onClose} title="Add to dataset">
      <div className="space-y-4">
        {/* --- Preview --- */}
        {preview && (
          <div className="rounded-md border border-gray-200 bg-gray-50 p-3">
            <dl className="space-y-1 text-xs">
              <Row label="Approved, ready to add" value={preview.would_add} strong />
              {preview.staged_unapproved > 0 && (
                <Row
                  label="Staged but not approved (will stay)"
                  value={preview.staged_unapproved}
                  muted
                />
              )}
              <Row label="Currently in dataset" value={preview.dataset_current} />
              {preview.would_remove > 0 && (
                <Row label="Will be DELETED" value={preview.would_remove} danger />
              )}
              <div className="mt-1 border-t border-gray-200 pt-1">
                <Row label="Dataset after" value={preview.dataset_after} strong />
              </div>
            </dl>
          </div>
        )}

        {nothingToAdd && (
          <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
            No approved images to add. Review and approve images first — only approved
            images move into the dataset.
          </p>
        )}

        {/* --- Mode --- */}
        <fieldset>
          <legend className="mb-1.5 text-xs font-medium text-gray-700">Mode</legend>
          <div className="space-y-1.5">
            {MODES.map((m) => (
              <label
                key={m.value}
                className={`flex cursor-pointer gap-2 rounded-md border p-2 transition-colors ${
                  mode === m.value
                    ? 'border-accent-500 bg-accent-50'
                    : 'border-gray-200 hover:bg-gray-50'
                }`}
              >
                <input
                  type="radio"
                  name="mode"
                  value={m.value}
                  checked={mode === m.value}
                  onChange={() => setMode(m.value)}
                  className="mt-0.5 accent-accent-600"
                />
                <span className="min-w-0">
                  <span className="block text-sm font-medium text-gray-900">{m.label}</span>
                  <span className="block text-xs text-gray-500">{m.blurb}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        {/* Destructive mode gets an explicit, specific warning with the real
            number in it — not a generic "are you sure?". */}
        {mode === 'replace' && preview && preview.would_remove > 0 && (
          <div className="flex gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2">
            <AlertTriangle size={14} className="mt-0.5 shrink-0 text-red-600" />
            <p className="text-xs text-red-800">
              <span className="font-medium">
                {preview.would_remove} image{preview.would_remove === 1 ? '' : 's'} and their
                annotations will be permanently deleted
              </span>{' '}
              from this project. This cannot be undone.
            </p>
          </div>
        )}

        {/* --- Splits --- */}
        <div>
          <label className="flex items-center gap-2 text-xs font-medium text-gray-700">
            <input
              type="checkbox"
              checked={assignSplits}
              onChange={(e) => setAssignSplits(e.target.checked)}
              className="accent-accent-600"
            />
            Assign train / val / test split
          </label>
          <p className="mt-0.5 text-xs text-gray-400">
            {assignSplits
              ? 'Images are shuffled with a fixed seed, so the split is reproducible.'
              : 'Each image keeps the split it already has (e.g. from an imported dataset).'}
          </p>

          {assignSplits && (
            <>
              <div className="mt-2 grid grid-cols-3 gap-2">
                <PctInput label="Train" value={trainPct} onChange={setTrainPct} />
                <PctInput label="Val" value={valPct} onChange={setValPct} />
                <PctInput label="Test" value={testPct} onChange={setTestPct} />
              </div>
              {!pctValid && (
                <p className="mt-1 text-xs text-red-600">
                  Must sum to 100% — currently {total}%.
                </p>
              )}
              {valPct === 0 && pctValid && (
                <p className="mt-1 text-xs text-amber-700">
                  No validation set: you'll have no way to tell if the model is
                  overfitting.
                </p>
              )}
            </>
          )}
        </div>

        {error && (
          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            {error}
          </p>
        )}

        <div className="flex justify-end gap-2 border-t border-gray-200 pt-3">
          <button className="btn-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className={mode === 'replace' ? 'btn bg-red-600 text-white hover:bg-red-700' : 'btn-primary'}
            onClick={() => void submit()}
            disabled={busy || nothingToAdd || (assignSplits && !pctValid)}
          >
            {busy
              ? 'Working…'
              : mode === 'replace'
                ? `Replace dataset`
                : `Add ${preview?.would_add ?? 0} to dataset`}
          </button>
        </div>
      </div>
    </Modal>
  )
}

function Row({
  label,
  value,
  strong,
  muted,
  danger,
}: {
  label: string
  value: number
  strong?: boolean
  muted?: boolean
  danger?: boolean
}) {
  return (
    <div className="flex justify-between">
      <dt className={danger ? 'text-red-700' : muted ? 'text-gray-400' : 'text-gray-600'}>
        {label}
      </dt>
      <dd
        className={[
          'font-mono tabular-nums',
          danger ? 'text-red-700' : muted ? 'text-gray-400' : 'text-gray-900',
          strong ? 'font-semibold' : '',
        ].join(' ')}
      >
        {value}
      </dd>
    </div>
  )
}

function PctInput({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <label className="block">
      <span className="mb-0.5 block text-xs text-gray-500">{label}</span>
      <div className="flex items-center rounded-md border border-gray-300 focus-within:border-accent-500">
        <input
          type="number"
          min={0}
          max={100}
          value={value}
          onChange={(e) => onChange(Math.max(0, Math.min(100, Number(e.target.value))))}
          className="w-full rounded-md px-2 py-1 text-sm tabular-nums focus:outline-none"
        />
        <span className="pr-2 text-xs text-gray-400">%</span>
      </div>
    </label>
  )
}
