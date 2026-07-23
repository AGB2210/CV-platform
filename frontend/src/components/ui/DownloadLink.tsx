import { useEffect, useRef, useState } from 'react'
import { Check, Download } from 'lucide-react'

/**
 * An `<a download>` that acknowledges the click.
 *
 * A native download link gives no feedback while the browser spins up the
 * transfer — for a multi-hundred-MB checkpoint or a large export zip that
 * dead moment reads as "the button didn't work", and the natural reaction
 * is a second click and a second copy. The label flips the instant the
 * click lands, then returns once the browser's own download UI has had
 * time to appear.
 *
 * Still an `<a>`, not a fetch: the browser streams straight to disk, shows
 * its own progress, and the download survives navigating away — none of
 * which a blob-in-memory fetch gives for big files.
 */
export function DownloadLink({
  href,
  className,
  children,
  startedLabel = 'Download started',
}: {
  href: string
  className?: string
  children: React.ReactNode
  startedLabel?: string
}) {
  const [started, setStarted] = useState(false)
  const timer = useRef<number | null>(null)
  // Clear the pending reset on unmount — a timeout firing setState against a
  // dead component is a console warning today and a leak pattern always.
  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current)
    },
    [],
  )
  return (
    <a
      href={href}
      download
      className={className}
      aria-live="polite"
      onClick={() => {
        setStarted(true)
        if (timer.current) window.clearTimeout(timer.current)
        timer.current = window.setTimeout(() => setStarted(false), 4000)
      }}
    >
      {started ? (
        <>
          <Check size={13} className="text-status-good" />
          {startedLabel}
        </>
      ) : (
        <>
          <Download size={13} />
          {children}
        </>
      )}
    </a>
  )
}
