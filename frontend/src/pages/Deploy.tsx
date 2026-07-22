import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Cpu, Download, Upload, ScanSearch } from 'lucide-react'
import { PageBody, PageHeader } from '@/components/layout/AppShell'
import { MlSetupGate } from '@/components/MlSetupGate'
import {
  listModels,
  onnxUrl,
  predictImage,
  weightsUrl,
  type DeployableModel,
  type PredictionResult,
} from '@/lib/api'

/**
 * Deploy — the inference playground. Pick a trained model, upload an image, see
 * what it finds. Read-only: nothing here is stored (predictions are not
 * annotations), so there are no accept/reject verbs and no writes.
 *
 * The overlay follows the one load-bearing rule from the annotation canvas: the
 * SVG viewBox is the image's NATURAL pixel dimensions, so a box at x=437 draws
 * at x=437 with no scale factor, at any display size. It does NOT reuse
 * AnnotationCanvas itself — that component edits Annotation ROWS, and a
 * prediction is a transient label+box, not a row.
 */
export function Deploy() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)

  const [models, setModels] = useState<DeployableModel[] | null>(null)
  const [modelId, setModelId] = useState<number | null>(null)
  const [threshold, setThreshold] = useState(0.25)
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    listModels(projectId)
      .then((m) => {
        setModels(m)
        if (m.length) setModelId(m[0].job_id)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [projectId])

  // Object URLs must be revoked or they leak. One per selected file.
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    setResult(null)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const run = async () => {
    if (modelId == null || !file) return
    setRunning(true)
    setError(null)
    try {
      setResult(await predictImage(modelId, file, threshold))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setResult(null)
    } finally {
      setRunning(false)
    }
  }

  const onPick = (f: File | null | undefined) => {
    if (f) setFile(f)
  }

  return (
    <>
      <PageHeader
        title="Deploy"
        description="Run a trained model on a new image"
        actions={
          <Link to={`/projects/${projectId}`} className="btn-secondary">
            <ArrowLeft size={14} />
            Dataset
          </Link>
        }
      />
      <PageBody>
        <MlSetupGate feature="Deploy">
          {error && (
            <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              {error}
            </div>
          )}

          {models !== null && models.length === 0 ? (
            <NoModelsCard projectId={projectId} />
          ) : (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
              {/* --- Left: controls --- */}
              <div className="space-y-4">
                <div className="card p-4">
                  <label className="label-eyebrow mb-1 block">Model</label>
                  <select
                    value={modelId ?? ''}
                    onChange={(e) => setModelId(Number(e.target.value))}
                    className="w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm"
                  >
                    {(models ?? []).map((m) => (
                      <option key={m.job_id} value={m.job_id}>
                        {m.label}
                        {m.best_map != null ? ` · mAP ${m.best_map.toFixed(3)}` : ''}
                      </option>
                    ))}
                  </select>

                  <label className="label-eyebrow mb-1 mt-4 block">
                    Confidence ≥ {threshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                    className="w-full accent-accent-600"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Higher keeps only confident boxes; lower surfaces more, with more
                    false positives.
                  </p>

                  {/* The model as a file. An <a download> streaming the .pt
                      straight from storage — take it to a Pi, a server, a
                      colleague's ultralytics script. */}
                  {modelId != null && (
                    <a
                      href={weightsUrl(modelId)}
                      download
                      className="btn-secondary mt-4 w-full"
                      title="Download this run's checkpoint (.pt) for use outside the app"
                    >
                      <Download size={13} />
                      Download weights (.pt)
                    </a>
                  )}
                  {modelId != null && (
                    <OnnxButton key={modelId} modelId={modelId} />
                  )}
                </div>

                <div className="card p-4">
                  <label className="label-eyebrow mb-2 block">Image</label>
                  <button
                    type="button"
                    onClick={() => fileRef.current?.click()}
                    className="flex w-full flex-col items-center gap-1.5 rounded border border-dashed border-gray-300 px-3 py-6 text-sm text-gray-500 hover:border-accent-400 hover:text-accent-700"
                  >
                    <Upload size={18} />
                    {/* Truncated, not wrapped: camera exports have 60-char
                        names, and letting one grow the button three lines
                        tall pushes Run detection off screen. The full name
                        stays available as a tooltip. */}
                    <span
                      className="max-w-full truncate"
                      title={file ? file.name : undefined}
                    >
                      {file ? file.name : 'Choose an image'}
                    </span>
                  </button>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => onPick(e.target.files?.[0])}
                  />
                  <button
                    type="button"
                    onClick={run}
                    disabled={modelId == null || !file || running}
                    className="btn-primary mt-3 inline-flex w-full items-center justify-center gap-1.5"
                  >
                    <ScanSearch size={14} />
                    {running ? 'Running…' : 'Run detection'}
                  </button>
                </div>

                {/* The playground, scriptable. Same endpoint the Run button
                    calls — documenting it makes every trained model a local
                    REST service with zero extra setup. */}
                {modelId != null && <ApiUsageCard modelId={modelId} threshold={threshold} />}
              </div>

              {/* --- Right: image + overlay --- */}
              <div className="card p-4">
                {!imageUrl ? (
                  <div className="flex h-full min-h-64 items-center justify-center text-sm text-gray-400">
                    Choose an image and run detection to see predictions.
                  </div>
                ) : (
                  <div className="space-y-3">
                    <PredictionView imageUrl={imageUrl} result={result} />
                    {result && (
                      <DetectionList result={result} />
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </MlSetupGate>
      </PageBody>
    </>
  )
}

/**
 * ONNX download. A fetch with a busy state rather than an <a download>,
 * because the FIRST request per checkpoint converts the model (up to a
 * minute on CPU) — a link that silently hangs for a minute reads as broken.
 * Later requests stream the cached file immediately.
 */
function OnnxButton({ modelId }: { modelId: number }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const download = async () => {
    setBusy(true)
    setError(null)
    try {
      const resp = await fetch(onnxUrl(modelId))
      if (!resp.ok) {
        const body = await resp.json().catch(() => null)
        throw new Error(body?.detail ?? `Export failed (${resp.status})`)
      }
      const blob = await resp.blob()
      // The server names the file; recover it from the header.
      const dispo = resp.headers.get('content-disposition') ?? ''
      const name = /filename="?([^";]+)"?/.exec(dispo)?.[1] ?? `model_${modelId}.onnx`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = name
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => void download()}
        disabled={busy}
        className="btn-secondary mt-2 w-full"
        title="Convert and download as ONNX — runnable without PyTorch, on edge devices and other stacks"
      >
        <Download size={13} />
        {busy ? 'Converting to ONNX… (first time takes a minute)' : 'Download ONNX'}
      </button>
      {error && <p className="mt-1 text-xs text-red-700">{error}</p>}
    </>
  )
}

/**
 * "Use via API" — the selected model as a documented local REST endpoint.
 *
 * Nothing here adds a server: the snippets call the SAME endpoint the Run
 * button uses, so anything that can POST multipart — a script, another app on
 * the LAN, a cron job — can use the trained model while this app is running.
 * Snippets are generated for the selected model and current threshold, so
 * copy-paste works without editing placeholders.
 */
function ApiUsageCard({ modelId, threshold }: { modelId: number; threshold: number }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState<string | null>(null)

  // The address the BROWSER reached the app on — right for localhost and for
  // a LAN machine, where a hardcoded localhost would be wrong for callers.
  const base = window.location.origin
  const url = `${base}/api/models/${modelId}/predict`

  const curl = `curl -X POST "${url}" \\
  -F "file=@your_image.jpg" \\
  -F "conf_threshold=${threshold}"`

  const python = `import requests

with open("your_image.jpg", "rb") as f:
    r = requests.post(
        "${url}",
        files={"file": f},
        data={"conf_threshold": ${threshold}},
    )
r.raise_for_status()
for box in r.json()["boxes"]:
    print(box["label"], box["confidence"], box["x"], box["y"], box["width"], box["height"])`

  const copy = async (label: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(label)
      window.setTimeout(() => setCopied(null), 1500)
    } catch {
      /* clipboard can be denied; the text is still selectable */
    }
  }

  return (
    <div className="card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-gray-900 hover:bg-gray-50"
      >
        Use via API
        <span className="text-xs font-normal text-gray-400">{open ? 'hide' : 'show'}</span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-gray-200 p-4">
          <p className="text-xs text-gray-500">
            While this app is running, the selected model answers HTTP on this
            machine — no extra server. POST an image, get boxes back as JSON
            (pixel coordinates, top-left origin). Nothing is stored.
          </p>
          <Snippet
            label="curl"
            text={curl}
            copied={copied === 'curl'}
            onCopy={() => void copy('curl', curl)}
          />
          <Snippet
            label="Python"
            text={python}
            copied={copied === 'Python'}
            onCopy={() => void copy('Python', python)}
          />
          <p className="text-xs text-gray-400">
            Response: {'{'} image_width, image_height, boxes: [{'{'} label,
            confidence, x, y, width, height {'}'}] {'}'}. Everything in this app
            shares one GPU, so if a training or auto-annotation job happens to be
            running when a request arrives, the API answers 409 (busy) instead of
            fighting it for memory — retry once the job finishes.
          </p>
        </div>
      )}
    </div>
  )
}

function Snippet({
  label,
  text,
  copied,
  onCopy,
}: {
  label: string
  text: string
  copied: boolean
  onCopy: () => void
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="label-eyebrow">{label}</span>
        <button
          type="button"
          onClick={onCopy}
          className="text-[11px] font-medium text-accent-700 hover:underline"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto rounded border border-gray-200 bg-gray-950 p-2.5 text-[11px] leading-relaxed text-gray-200">
        {text}
      </pre>
    </div>
  )
}

// A stable colour per class label, so the same class is the same colour across
// images. Hash the label into the accent-neutral palette range.
const BOX_COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d',
]
function colorFor(label: string): string {
  let h = 0
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0
  return BOX_COLORS[h % BOX_COLORS.length]
}

function PredictionView({
  imageUrl,
  result,
}: {
  imageUrl: string
  result: PredictionResult | null
}) {
  return (
    <div className="relative inline-block max-w-full">
      <img src={imageUrl} alt="uploaded" className="block max-w-full rounded" />
      {result && (
        // viewBox = the image's NATURAL dimensions, so one SVG unit is one image
        // pixel and boxes need no scale factor at any display size.
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${result.image_width} ${result.image_height}`}
          preserveAspectRatio="none"
        >
          {result.boxes.map((b, i) => {
            const c = colorFor(b.label)
            return (
              <g key={i}>
                <rect
                  x={b.x}
                  y={b.y}
                  width={b.width}
                  height={b.height}
                  fill="none"
                  stroke={c}
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={b.x + 2}
                  y={b.y - 3}
                  fill={c}
                  fontSize={12}
                  style={{ paintOrder: 'stroke', stroke: 'white', strokeWidth: 3 }}
                >
                  {b.label} {(b.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

function DetectionList({ result }: { result: PredictionResult }) {
  const byClass = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const b of result.boxes) counts[b.label] = (counts[b.label] ?? 0) + 1
    return counts
  }, [result])

  if (result.boxes.length === 0) {
    return (
      <p className="rounded border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-500">
        No objects above the confidence threshold. Lower it to surface weaker detections.
      </p>
    )
  }
  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {Object.entries(byClass).map(([label, n]) => (
        <span
          key={label}
          className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 px-2 py-0.5"
        >
          <span className="h-2 w-2 rounded-full" style={{ background: colorFor(label) }} />
          {label} · {n}
        </span>
      ))}
    </div>
  )
}

function NoModelsCard({ projectId }: { projectId: number }) {
  return (
    <div className="card max-w-xl">
      <div className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <Cpu size={14} className="text-gray-400" />
        <h2 className="text-sm font-medium text-gray-900">No trained model yet</h2>
      </div>
      <div className="space-y-2 p-4 text-sm text-gray-600">
        <p>
          Deploy runs a model you have trained. Train one first — a finished run with a
          saved checkpoint appears here automatically.
        </p>
        <Link
          to={`/projects/${projectId}/train`}
          className="btn-primary inline-flex items-center gap-1.5"
        >
          Go to Train
        </Link>
      </div>
    </div>
  )
}
