import { useCallback, useEffect, useState } from 'react'
import { Check, Sparkles, X } from 'lucide-react'
import {
  acceptProposals,
  getProposalPreview,
  rejectProposals,
  type ProposalPreview,
} from '@/lib/api'

/**
 * The pending model batch: accept it or reject it.
 *
 * While a batch is pending the canvas shows the MODEL'S boxes and nothing else.
 * This bar is the decision:
 *
 *   Accept  the model's boxes become your annotations; your previous boxes on
 *           the images it covered are deleted.
 *   Reject  the proposals are discarded and your boxes come straight back.
 *
 * It used to offer append/merge/replace, defaulting to append — so accepting
 * stacked the model's boxes on top of the last run's and four runs gave you the
 * same object boxed four times. Two buttons say everything the three modes did,
 * and can't manufacture duplicates.
 */
export function ProposalBar({
  projectId,
  proposedBoxes,
  onChanged,
}: {
  projectId: number
  proposedBoxes: number
  onChanged: () => void
}) {
  const [preview, setPreview] = useState<ProposalPreview | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setPreview(await getProposalPreview(projectId))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [projectId])

  useEffect(() => {
    if (proposedBoxes > 0) void load()
  }, [load, proposedBoxes])

  if (proposedBoxes === 0) return null

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setError(null)
    try {
      await fn()
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // Accepting deletes your boxes on the covered images. If there are none, it's
  // a harmless action and shouldn't wear a warning colour.
  const willDelete = preview?.existing_on_proposed_images ?? 0

  return (
    <div className="border-b border-accent-200 bg-accent-50">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-2">
        <Sparkles size={14} className="shrink-0 text-accent-700" />

        <div className="min-w-0 text-xs">
          <p className="text-accent-900">
            <span className="font-medium">
              Reviewing model output — {proposedBoxes} box
              {proposedBoxes === 1 ? '' : 'es'}
            </span>
            {preview && (
              <span className="text-accent-800">
                {' '}
                across {preview.proposed_images} image
                {preview.proposed_images === 1 ? '' : 's'}. Your own boxes are hidden
                while you decide.
              </span>
            )}
          </p>
          {/* State the consequence in the bar, not in a dialog after the fact.
              Accept DELETES — that has to be visible before the click. */}
          {preview && (
            <p className="text-accent-800">
              {willDelete > 0 ? (
                <>
                  <span className="font-medium">
                    Accepting replaces your {willDelete} existing box
                    {willDelete === 1 ? '' : 'es'}
                  </span>{' '}
                  on {preview.proposed_images === 1 ? 'this image' : 'these images'}.
                  Rejecting keeps them.
                </>
              ) : (
                <>You have no boxes on these images, so accepting deletes nothing.</>
              )}
              {preview.existing_elsewhere > 0 && (
                <>
                  {' '}
                  Your {preview.existing_elsewhere} box
                  {preview.existing_elsewhere === 1 ? '' : 'es'} on other images
                  {preview.existing_elsewhere === 1 ? ' is' : ' are'} unaffected either
                  way.
                </>
              )}
            </p>
          )}
        </div>

        <div className="ml-auto flex shrink-0 items-center gap-2">
          <button
            className="btn-secondary"
            onClick={() => void act(() => rejectProposals(projectId))}
            disabled={busy}
            title="Discard the model's boxes and keep yours"
          >
            <X size={13} />
            Reject
          </button>
          <button
            className={
              willDelete > 0
                ? 'btn bg-red-600 text-white hover:bg-red-700'
                : 'btn-primary'
            }
            onClick={() => void act(() => acceptProposals(projectId))}
            disabled={busy}
            title={
              willDelete > 0
                ? `Keep the model's boxes and delete your ${willDelete} existing`
                : "Keep the model's boxes"
            }
          >
            <Check size={13} />
            {busy
              ? 'Working…'
              : willDelete > 0
                ? `Accept, replace ${willDelete}`
                : `Accept ${proposedBoxes}`}
          </button>
        </div>
      </div>

      {error && (
        <p className="border-t border-red-200 bg-red-50 px-4 py-1.5 text-xs text-red-800">
          {error}
        </p>
      )}
    </div>
  )
}
