import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { EpochPoint } from '@/lib/api'

/**
 * Epoch-vs-mAP chart for a training run.
 *
 * A real chart, not a sparkline: labelled axes, a marker at every epoch (so each
 * measured value is a distinct point you can read, not an anonymous line), a
 * hover tooltip with the exact numbers, and zoom/pan so a long run's late-epoch
 * plateau can be inspected up close.
 *
 * WHY CUSTOM SVG RATHER THAN A CHART LIBRARY
 * ------------------------------------------
 * The project keeps its dependency surface small and its look hand-tuned. A
 * chart lib would pull hundreds of KB and still need custom work for the zoom
 * behaviour and the design-token styling. This is ~1 file with no new deps.
 *
 * INTERACTION MODEL
 * -----------------
 *   wheel        zoom in/out around the cursor (both axes)
 *   drag         pan
 *   double-click reset to auto-fit
 * While auto-fit is active (the user hasn't zoomed), the visible range tracks
 * the data — so a live run's curve keeps filling the frame as epochs arrive.
 * The moment you zoom, it holds still until you reset.
 */

interface Domain {
  x0: number
  x1: number
  y0: number
  y1: number
}

const H = 220
const PAD = { l: 44, r: 14, t: 14, b: 30 }

/** "Nice" round tick values across [min, max]. Standard 1/2/5×10ⁿ stepping. */
function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min
  if (span <= 0) return [min]
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const start = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(Math.round(v * 1e6) / 1e6)
  return ticks
}

export function MetricsChart({ points }: { points: EpochPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [width, setWidth] = useState(600)
  // null = auto-fit to the data; a Domain = the user has zoomed/panned.
  const [view, setView] = useState<Domain | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const drag = useRef<{ px: number; py: number; dom: Domain } | null>(null)

  // Measure available width so 1 SVG unit == 1 CSS pixel, which keeps pointer
  // maths trivial (offsetX/offsetY map straight onto the coordinate system).
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const data = useMemo(() => points.filter((p) => p.val_map !== null), [points])

  const fullDomain = useMemo<Domain>(() => {
    if (!data.length) return { x0: 0, x1: 1, y0: 0, y1: 1 }
    const xs = data.map((d) => d.epoch)
    const maxMap = Math.max(...data.map((d) => Math.max(d.val_map ?? 0, d.val_map50 ?? 0)))
    return {
      x0: Math.min(...xs),
      x1: Math.max(...xs, Math.min(...xs) + 1),
      y0: 0,
      // Round the top up to a tenth so the axis has a clean ceiling and the
      // curve isn't glued to the frame edge.
      y1: Math.max(0.1, Math.ceil((maxMap * 1.08) * 10) / 10),
    }
  }, [data])

  const dom = view ?? fullDomain
  const W = width
  const plotW = Math.max(1, W - PAD.l - PAD.r)
  const plotH = H - PAD.t - PAD.b

  const sx = (e: number) => PAD.l + ((e - dom.x0) / (dom.x1 - dom.x0 || 1)) * plotW
  const sy = (v: number) => PAD.t + (1 - (v - dom.y0) / (dom.y1 - dom.y0 || 1)) * plotH
  const invX = (px: number) => dom.x0 + ((px - PAD.l) / plotW) * (dom.x1 - dom.x0)

  // Wheel zoom needs a non-passive listener to preventDefault the page scroll.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = svg.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const base = view ?? fullDomain
      const cx = base.x0 + ((px - PAD.l) / plotW) * (base.x1 - base.x0)
      const cy = base.y0 + (1 - (py - PAD.t) / plotH) * (base.y1 - base.y0)
      const k = e.deltaY > 0 ? 1.15 : 1 / 1.15 // >0 = scroll down = zoom out
      const nx0 = cx - (cx - base.x0) * k
      const nx1 = cx + (base.x1 - cx) * k
      const ny0 = cy - (cy - base.y0) * k
      const ny1 = cy + (base.y1 - cy) * k
      // Clamp: never invert, never zoom past a sliver.
      if (nx1 - nx0 < 0.5 || ny1 - ny0 < 0.01) return
      setView({ x0: nx0, x1: nx1, y0: ny0, y1: ny1 })
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [view, fullDomain, plotW, plotH])

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    drag.current = { px: e.nativeEvent.offsetX, py: e.nativeEvent.offsetY, dom }
  }
  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    const ox = e.nativeEvent.offsetX
    const oy = e.nativeEvent.offsetY
    if (drag.current) {
      const d = drag.current
      const dxData = ((ox - d.px) / plotW) * (d.dom.x1 - d.dom.x0)
      const dyData = ((oy - d.py) / plotH) * (d.dom.y1 - d.dom.y0)
      setView({
        x0: d.dom.x0 - dxData,
        x1: d.dom.x1 - dxData,
        y0: d.dom.y0 + dyData,
        y1: d.dom.y1 + dyData,
      })
      return
    }
    // Hover: snap to the nearest epoch that has a point.
    if (!data.length) return
    const targetEpoch = invX(ox)
    let nearest = data[0].epoch
    let best = Infinity
    for (const p of data) {
      const d = Math.abs(p.epoch - targetEpoch)
      if (d < best) {
        best = d
        nearest = p.epoch
      }
    }
    setHover(nearest)
  }
  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    drag.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)
  }

  if (!data.length) {
    return (
      <p className="py-6 text-center text-xs text-gray-400">
        No mAP measured yet — the curve appears after the first validated epoch.
      </p>
    )
  }

  const xTicks = niceTicks(dom.x0, dom.x1, 6).filter((t) => t >= dom.x0 - 1e-6 && t <= dom.x1 + 1e-6)
  const yTicks = niceTicks(dom.y0, dom.y1, 5).filter((t) => t >= dom.y0 - 1e-6 && t <= dom.y1 + 1e-6)
  const line = (key: 'val_map' | 'val_map50') =>
    data
      .filter((d) => d[key] !== null)
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${sx(d.epoch).toFixed(1)},${sy(d[key] as number).toFixed(1)}`)
      .join(' ')
  const hoverPt = hover !== null ? data.find((d) => d.epoch === hover) : null
  const zoomed = view !== null

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[10px] text-gray-400">
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-[1px] bg-accent-600" /> mAP@50-95
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-gray-400" /> mAP@50
          </span>
        </span>
        {/* Always rendered, not just once zoomed. A control that only appears
            after you've already zoomed is one you have to discover by accident —
            and the double-click shortcut is invisible either way. Disabled when
            there's nothing to reset, so it reads as "available, not needed". */}
        <span className="flex items-center gap-2">
          <span className="hidden text-gray-300 sm:inline">
            scroll to zoom · drag to pan
          </span>
          <button
            onClick={() => setView(null)}
            disabled={!zoomed}
            className="rounded border border-gray-200 px-1.5 py-px text-[10px] text-accent-700 enabled:hover:bg-accent-50 disabled:border-gray-100 disabled:text-gray-300"
          >
            Reset zoom
          </button>
        </span>
      </div>
      <div ref={containerRef} className="w-full select-none">
        <svg
          ref={svgRef}
          width={W}
          height={H}
          className="touch-none"
          style={{ cursor: drag.current ? 'grabbing' : 'crosshair' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => setHover(null)}
          onDoubleClick={() => setView(null)}
        >
          <defs>
            <clipPath id="plot-clip">
              <rect x={PAD.l} y={PAD.t} width={plotW} height={plotH} />
            </clipPath>
          </defs>

          {/* y grid + labels */}
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={PAD.l} y1={sy(t)} x2={W - PAD.r} y2={sy(t)} stroke="#f1f1f2" strokeWidth={1} />
              <text x={PAD.l - 6} y={sy(t)} dy="0.32em" textAnchor="end" className="fill-gray-400 text-[9px] tabular-nums">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {/* x grid + labels */}
          {xTicks.map((t) => (
            <g key={`x${t}`}>
              <line x1={sx(t)} y1={PAD.t} x2={sx(t)} y2={H - PAD.b} stroke="#f6f6f7" strokeWidth={1} />
              <text x={sx(t)} y={H - PAD.b + 14} textAnchor="middle" className="fill-gray-400 text-[9px] tabular-nums">
                {Math.round(t)}
              </text>
            </g>
          ))}
          {/* axis frame */}
          <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#d4d4d8" strokeWidth={1} />
          <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#d4d4d8" strokeWidth={1} />
          <text x={PAD.l + plotW / 2} y={H - 2} textAnchor="middle" className="fill-gray-500 text-[9px]">
            epoch
          </text>

          <g clipPath="url(#plot-clip)">
            {/* mAP@50 secondary line + round markers */}
            <path d={line('val_map50')} fill="none" stroke="#a1a1aa" strokeWidth={1.25} />
            {data.filter((d) => d.val_map50 !== null).map((d) => (
              <circle key={`c${d.epoch}`} cx={sx(d.epoch)} cy={sy(d.val_map50 as number)} r={2} fill="#a1a1aa" />
            ))}
            {/* mAP@50-95 primary line + SQUARE markers (the "boxes of values") */}
            <path d={line('val_map')} fill="none" stroke="var(--color-accent-600)" strokeWidth={1.75} />
            {data.map((d) => {
              const on = hover === d.epoch
              const s = on ? 8 : 5
              return (
                <rect
                  key={`s${d.epoch}`}
                  x={sx(d.epoch) - s / 2}
                  y={sy(d.val_map as number) - s / 2}
                  width={s}
                  height={s}
                  rx={1}
                  fill={on ? 'var(--color-accent-600)' : '#fff'}
                  stroke="var(--color-accent-600)"
                  strokeWidth={1.5}
                />
              )
            })}
            {/* hover guide */}
            {hoverPt && (
              <line x1={sx(hoverPt.epoch)} y1={PAD.t} x2={sx(hoverPt.epoch)} y2={H - PAD.b} stroke="var(--color-accent-400)" strokeWidth={1} strokeDasharray="3 3" />
            )}
          </g>

          {/* hover tooltip — drawn outside the clip so it's never cut off */}
          {hoverPt && (
            <HoverTooltip pt={hoverPt} x={sx(hoverPt.epoch)} chartW={W} />
          )}
        </svg>
      </div>
    </div>
  )
}

function HoverTooltip({ pt, x, chartW }: { pt: EpochPoint; x: number; chartW: number }) {
  const w = 132
  // Flip to the left of the cursor when near the right edge.
  const tx = x + w + 8 > chartW ? x - w - 8 : x + 8
  const ty = PAD.t + 4
  const rows: [string, string][] = [
    ['epoch', String(pt.epoch)],
    ['mAP@50-95', pt.val_map != null ? pt.val_map.toFixed(3) : '—'],
    ['mAP@50', pt.val_map50 != null ? pt.val_map50.toFixed(3) : '—'],
    ['loss', pt.train_loss != null ? pt.train_loss.toFixed(3) : '—'],
  ]
  return (
    <g pointerEvents="none">
      <rect x={tx} y={ty} width={w} height={rows.length * 15 + 8} rx={4} fill="#111" opacity={0.9} />
      {rows.map(([k, v], i) => (
        <text key={k} x={tx + 8} y={ty + 15 + i * 15} className="fill-white text-[9px]">
          <tspan className="fill-gray-400">{k}</tspan>
          <tspan x={tx + w - 8} textAnchor="end" className="fill-white tabular-nums">{v}</tspan>
        </text>
      ))}
    </g>
  )
}
