import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'

/**
 * Modal dialog built on the native <dialog> element.
 *
 * Using <dialog> rather than a hand-rolled div + portal gets several things for
 * free that are tedious and easy to get wrong by hand:
 *   - focus trapping (Tab stays inside the dialog)
 *   - Escape to close
 *   - the top layer, so it renders above everything without z-index roulette
 *   - inert background content for screen readers
 *
 * showModal() must be called imperatively — there's no `open` prop that
 * produces modal behaviour, since the `open` attribute alone renders a
 * *non-modal* dialog. Hence the effect below.
 */
export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
}) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    else if (!open && dialog.open) dialog.close()
  }, [open])

  return (
    <dialog
      ref={ref}
      // The browser fires 'close' for Escape too, so routing our own close
      // through the same handler keeps React state in sync however it happened.
      onClose={onClose}
      // Click-outside-to-dismiss. The backdrop isn't a separate element, so a
      // click on it reports the <dialog> itself as the target — comparing
      // e.target to the dialog is how you distinguish backdrop from content.
      onClick={(e) => {
        if (e.target === ref.current) onClose()
      }}
      // `m-auto` is not cosmetic padding — it restores centring. A native
      // <dialog> centres itself in the top layer via the UA stylesheet's
      // `margin: auto`, but Tailwind's preflight reset zeroes margins on every
      // element, which silently pins the dialog to the top-left corner.
      className="m-auto w-full max-w-md rounded-lg border border-gray-200 p-0 shadow-lg backdrop:bg-gray-900/25"
    >
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-gray-900">{title}</h2>
        <button
          onClick={onClose}
          className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          aria-label="Close"
        >
          <X size={16} />
        </button>
      </div>

      <div className="px-4 py-4">{children}</div>

      {footer && (
        <div className="flex justify-end gap-2 border-t border-gray-200 bg-gray-50 px-4 py-3">
          {footer}
        </div>
      )}
    </dialog>
  )
}

/**
 * Confirmation prompt for destructive actions.
 *
 * Deliberately spells out the consequence in `message` rather than asking a
 * generic "Are you sure?" — deleting a project takes its images and classes
 * with it, and that should be stated before the click, not discovered after.
 */
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  message,
  confirmLabel = 'Delete',
  busy = false,
}: {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmLabel?: string
  busy?: boolean
}) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <>
          <button className="btn-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            // The one place a red button is warranted: it marks an irreversible
            // action. Using the normal accent here would make deleting look
            // like any other primary action.
            className="btn bg-red-600 text-white hover:bg-red-700"
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Deleting…' : confirmLabel}
          </button>
        </>
      }
    >
      <p className="text-sm text-gray-600">{message}</p>
    </Modal>
  )
}
