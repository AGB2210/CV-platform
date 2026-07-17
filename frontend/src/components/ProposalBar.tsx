import { Check, Sparkles, X } from 'lucide-react'
import { acceptProposals, rejectProposals, type ProposalPreview } from '@/lib/api'

/**
 * The pending model batch, split into two pieces that live in DIFFERENT PLACES.
 *
 * WHY SPLIT
 * ---------
 * The batch actions and the per-image actions used to sit in stacked toolbars:
 * "Reject all" 16px above "Reject image", with 62px of horizontal overlap. Two
 * red buttons, same word, and a small slip turned "discard this image's
 * suggestions" into "discard the entire batch". Adjacency was doing the exact
 * opposite of what the labels worked so hard to establish.
 *
 * So they're now separated by scope AND by geography:
 *   ProposalBanner   top strip — explains the mode, NO buttons
 *   ProposalActions  right panel, beside the other project-scoped action
 *                    ("Add to dataset") — the batch buttons
 * The per-image pair keeps the image header. Different columns, no overlap,
 * nothing to slip between.
 *
 * Both take `preview` as a prop rather than fetching it: they'd otherwise issue
 * the same request twice and could disagree with each other mid-flight.
 */

export function ProposalBanner({
  proposedBoxes,
  preview,
}: {
  proposedBoxes: number
  preview: ProposalPreview | null
}) {
  if (proposedBoxes === 0) return null
  const willDelete = preview?.existing_on_proposed_images ?? 0

  return (
    <div className="flex items-start gap-2 border-b border-accent-200 bg-accent-50 px-4 py-2">
      <Sparkles size={14} className="mt-0.5 shrink-0 text-accent-700" />
      <div className="min-w-0 text-xs">
        <p className="text-accent-900">
          <span className="font-medium">
            Reviewing model output — {proposedBoxes} box{proposedBoxes === 1 ? '' : 'es'}
          </span>
          {preview && (
            <span className="text-accent-800">
              {' '}
              across {preview.proposed_images} image
              {preview.proposed_images === 1 ? '' : 's'}. Your own boxes are hidden while
              you decide.
            </span>
          )}
        </p>
        {/* The consequence, stated before any click — not in a dialog after. */}
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
                {preview.existing_elsewhere === 1 ? ' is' : ' are'} unaffected either way.
              </>
            )}
          </p>
        )}
      </div>
    </div>
  )
}

/**
 * Batch actions. Lives in the right panel, deliberately far from the per-image
 * pair in the image header.
 */
export function ProposalActions({
  projectId,
  proposedBoxes,
  preview,
  busy,
  onBusy,
  onChanged,
  onError,
}: {
  projectId: number
  proposedBoxes: number
  preview: ProposalPreview | null
  busy: boolean
  onBusy: (v: boolean) => void
  onChanged: () => void
  onError: (msg: string) => void
}) {
  if (proposedBoxes === 0) return null
  const willDelete = preview?.existing_on_proposed_images ?? 0

  async function act(fn: () => Promise<unknown>) {
    onBusy(true)
    try {
      await fn()
      onChanged()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      onBusy(false)
    }
  }

  return (
    <div className="border-b border-accent-200 bg-accent-50 p-3">
      <p className="label-eyebrow text-accent-700">Model batch</p>
      <p className="mt-0.5 text-[11px] text-accent-900">
        {proposedBoxes} box{proposedBoxes === 1 ? '' : 'es'} across{' '}
        {preview?.proposed_images ?? 0} image
        {(preview?.proposed_images ?? 0) === 1 ? '' : 's'} — applies to{' '}
        <span className="font-medium">every image</span>, not just this one.
      </p>

      {/* OUTLINED, and stacked full-width — unlike the header's solid,
          side-by-side pair. Three differences at once (place, shape, fill) so
          the scope is obvious without reading a word. Outlined also suits the
          rarer action: batch decisions shouldn't shout as loudly as the
          per-image ones you click a hundred times. */}
      <button
        className="btn-accept-outline mt-2 w-full"
        onClick={() => void act(() => acceptProposals(projectId))}
        disabled={busy}
        title={
          willDelete > 0
            ? `Keep the model's boxes on EVERY image and delete your ${willDelete} existing`
            : "Keep the model's boxes on every image"
        }
      >
        <Check size={13} />
        {busy ? 'Working…' : willDelete > 0 ? `Accept all, replace ${willDelete}` : `Accept all ${proposedBoxes}`}
      </button>
      <button
        className="btn-reject-outline mt-1.5 w-full"
        onClick={() => void act(() => rejectProposals(projectId))}
        disabled={busy}
        title="Discard the model's boxes on EVERY image and keep yours"
      >
        <X size={13} />
        Reject all {proposedBoxes}
      </button>
    </div>
  )
}
