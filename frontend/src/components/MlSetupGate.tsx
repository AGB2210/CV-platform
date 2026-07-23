import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Loader2, TriangleAlert, Cpu, MonitorCog } from 'lucide-react'
import { getMlStatus, startMlInstall, type MlStatus } from '@/lib/api'

/**
 * Gate a feature behind the on-demand ML install.
 *
 * Auto-annotate and Train need torch + transformers + ultralytics — several GB
 * that the app does NOT install up front (see backend services/ml_setup.py).
 * The first time someone opens one of those features, this shows an install
 * panel instead of the feature, runs the install with progress on the page, and
 * reveals the feature when it's ready. The user never touches a command line.
 *
 * While the stack is present (the common case after the first install) this is a
 * transparent pass-through: it renders its children and gets out of the way.
 */
export function MlSetupGate({ feature, children }: { feature: string; children: React.ReactNode }) {
  const [status, setStatus] = useState<MlStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // A ref, not state: the poll loop reads it without re-subscribing on change.
  const pollingRef = useRef<number | null>(null)

  const refresh = useCallback(async () => {
    try {
      setStatus(await getMlStatus())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Poll only while an install is running — the one time the state changes on
  // its own. Idle/installed/failed are stable until the user acts, so polling
  // them would be a request every couple of seconds for nothing.
  const installStatus = status?.install.status
  useEffect(() => {
    if (installStatus !== 'running') {
      if (pollingRef.current) window.clearInterval(pollingRef.current)
      pollingRef.current = null
      return
    }
    pollingRef.current = window.setInterval(refresh, 2000)
    return () => {
      if (pollingRef.current) window.clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }, [installStatus, refresh])

  // Keep the install log pinned to the newest line, terminal-style — unless
  // the user scrolls up to read something, in which case leave them be until
  // they scroll back to the bottom themselves. Same pattern as the training
  // page's LiveLogs. Refs, not state: scroll position isn't render data.
  const logRef = useRef<HTMLPreElement>(null)
  const logPinned = useRef(true)
  const logText = status?.install.log_tail.join('\n') ?? ''
  useEffect(() => {
    const el = logRef.current
    if (el && logPinned.current) el.scrollTop = el.scrollHeight
  }, [logText])

  const install = async () => {
    setStarting(true)
    setError(null)
    try {
      setStatus(await startMlInstall())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  // Still checking — say nothing rather than flash the feature and yank it back.
  if (!status) {
    return (
      <div className="flex items-center gap-2 p-6 text-sm text-gray-500">
        <Loader2 size={14} className="animate-spin" />
        Checking setup…
      </div>
    )
  }

  // Installed: the feature is available. Get out of the way.
  if (status.installed) return <>{children}</>

  const inst = status.install
  const running = inst.status === 'running'
  const failed = inst.status === 'failed'
  const plan = status.plan

  return (
    <div className="card max-w-2xl">
      <div className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <MonitorCog size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">
          {feature} needs a one-time setup
        </h2>
      </div>

      <div className="space-y-3 p-4 text-sm text-gray-600">
        {!running && !failed && (
          <>
            <p>
              This feature runs a machine-learning model, which needs PyTorch and
              a few companion libraries — several gigabytes, kept out of the base
              install so the app opens quickly. It installs once; after that this
              step never appears again.
            </p>
            <div className="flex items-center gap-2 rounded border border-gray-200 bg-gray-50 px-3 py-2 text-xs">
              <Cpu size={13} className="shrink-0 text-gray-400" />
              {plan.gpu_detected ? (
                <span>
                  GPU detected (driver CUDA {plan.driver_cuda}) — installing the{' '}
                  <span className="font-mono text-gray-800">{plan.torch_build}</span> build for
                  hardware acceleration.
                </span>
              ) : (
                <span>
                  No CUDA GPU detected — installing the{' '}
                  <span className="font-mono text-gray-800">CPU</span> build. It will work, but
                  model runs will be slow.
                </span>
              )}
            </div>
          </>
        )}

        {running && (
          <>
            <div className="flex items-center gap-2 text-gray-900">
              <Loader2 size={15} className="animate-spin text-accent-600" />
              <span className="font-medium">{inst.phase || 'Installing…'}</span>
            </div>
            {/* Indeterminate bar: pip gives no reliable percentage, and a fake
                one is worse than an honest "working". */}
            <div className="h-1.5 overflow-hidden rounded-full bg-gray-100">
              <div className="ml-setup-bar h-full w-1/3 rounded-full bg-accent-500" />
            </div>
            <p className="text-xs text-gray-500">
              This can take a few minutes on the first run — it's a large download. You can leave
              this page open; it will continue.
            </p>
          </>
        )}

        {failed && (
          <div className="flex items-start gap-2 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
            <TriangleAlert size={14} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">The install didn't finish.</p>
              <p className="mt-0.5">{inst.error}</p>
            </div>
          </div>
        )}

        {error && <p className="text-xs text-red-700">{error}</p>}

        {inst.log_tail.length > 0 && (
          <details
            className="text-xs"
            // Opening the panel lands at the newest line, not the top — that's
            // where the action is. Also re-pins after a scroll-up + close.
            onToggle={(e) => {
              if (!e.currentTarget.open) return
              const el = logRef.current
              if (el) {
                logPinned.current = true
                el.scrollTop = el.scrollHeight
              }
            }}
          >
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              Installation log
            </summary>
            <pre
              ref={logRef}
              onScroll={(e) => {
                const el = e.currentTarget
                logPinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
              }}
              className="mt-1 max-h-40 overflow-auto rounded border border-gray-200 bg-gray-50 p-2 text-[11px] leading-relaxed text-gray-700"
            >
              {inst.log_tail.join('\n')}
            </pre>
          </details>
        )}

        {!running && (
          <button
            type="button"
            onClick={install}
            disabled={starting}
            className="btn-primary inline-flex items-center gap-1.5"
          >
            <Download size={14} />
            {failed ? 'Try again' : starting ? 'Starting…' : 'Install and continue'}
          </button>
        )}
      </div>
    </div>
  )
}
