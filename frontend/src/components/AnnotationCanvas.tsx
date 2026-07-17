import { useCallback, useEffect, useRef, useState } from 'react'
import type { Annotation, ProjectClass } from '@/lib/api'

/**
 * Bounding box editor.
 *
 * WHY SVG AND NOT <canvas>
 * -----------------------
 * <canvas> means immediate-mode drawing: you repaint everything each frame and
 * implement hit-testing by hand (which box is under the cursor? which handle?).
 * SVG gives every box a real DOM node, so hit-testing IS the browser's event
 * system, hover/focus are CSS, and boxes are inspectable in devtools.
 *
 * Canvas wins beyond a few thousand shapes. An image never has thousands of
 * boxes, so that tradeoff never arrives. This is what CVAT and Label Studio do.
 *
 * THE COORDINATE TRICK — the single most important idea here
 * ---------------------------------------------------------
 * The SVG's viewBox is set to "0 0 imageWidth imageHeight", i.e. the image's
 * NATURAL pixel dimensions, while the SVG element itself is stretched to
 * whatever size the image displays at.
 *
 * The consequence: one SVG user unit === one image pixel, always. A box stored
 * as x=437.68 renders at x=437.68 with no conversion, at any zoom, in any
 * window size. No scale factors threaded through the component, no drift, no
 * "boxes are slightly off when the window is narrow" bug.
 *
 * The only conversion needed is the inverse — screen coords from a mouse event
 * back into image space — and getScreenCTM().inverse() does that exactly,
 * accounting for scaling, scrolling, and CSS transforms in one step.
 */

type Handle = 'nw' | 'ne' | 'sw' | 'se'
type Drag =
  | { kind: 'create'; startX: number; startY: number }
  | { kind: 'move'; id: number; offsetX: number; offsetY: number }
  | { kind: 'resize'; id: number; handle: Handle; anchorX: number; anchorY: number }

/** A box mid-edit, in image pixels. */
interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface CanvasProps {
  imageUrl: string
  imageWidth: number
  imageHeight: number
  annotations: Annotation[]
  classes: ProjectClass[]
  /** Class assigned to newly drawn boxes. */
  activeClassId: number | null
  selectedId: number | null
  onSelect: (id: number | null) => void
  onCreate: (rect: Rect, categoryId: number) => void
  onUpdate: (id: number, rect: Rect) => void
  onDelete: (id: number) => void
  /** Accept one model proposal. Absent = proposals aren't actionable here. */
  onAccept?: (id: number) => void
}

/** Below this many pixels a drag is a click, not a box. Prevents the classic
 *  "clicked to deselect, accidentally created a 2px box" annoyance. */
const MIN_DRAG = 4

export function AnnotationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  annotations,
  classes,
  activeClassId,
  selectedId,
  onSelect,
  onCreate,
  onUpdate,
  onDelete,
  onAccept,
}: CanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  /** The rect being dragged, rendered instead of the stored one. Local state
   *  so dragging is smooth — sending a PATCH per mousemove would be absurd. */
  const [preview, setPreview] = useState<Rect | null>(null)

  const colorOf = useCallback(
    (categoryId: number) => classes.find((c) => c.id === categoryId)?.color ?? '#71717a',
    [classes],
  )

  /**
   * Screen coordinates -> image pixels.
   *
   * getScreenCTM() is the SVG element's current transform to screen space.
   * Inverting it and applying to the mouse point undoes scaling, page scroll,
   * and any CSS transform in one operation. Doing this by hand with
   * getBoundingClientRect ratios is where "boxes drift when zoomed" comes from.
   */
  const toImageCoords = useCallback((e: { clientX: number; clientY: number }) => {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const ctm = svg.getScreenCTM()
    if (!ctm) return { x: 0, y: 0 }
    const p = pt.matrixTransform(ctm.inverse())
    return { x: p.x, y: p.y }
  }, [])

  const clampRect = useCallback(
    (r: Rect): Rect => {
      // Keep boxes inside the image. A box half outside is unexportable and
      // trains on pixels that don't exist.
      const x = Math.max(0, Math.min(r.x, imageWidth))
      const y = Math.max(0, Math.min(r.y, imageHeight))
      return {
        x,
        y,
        width: Math.max(1, Math.min(r.width, imageWidth - x)),
        height: Math.max(1, Math.min(r.height, imageHeight - y)),
      }
    },
    [imageWidth, imageHeight],
  )

  // --- pointer handling --------------------------------------------------

  function onPointerDownBackground(e: React.PointerEvent) {
    // Only the background starts a create-drag; boxes stop propagation.
    if (e.button !== 0 || activeClassId === null) return
    const { x, y } = toImageCoords(e)
    onSelect(null)
    setDrag({ kind: 'create', startX: x, startY: y })
    setPreview({ x, y, width: 0, height: 0 })
    // Capture means we keep receiving move/up even if the cursor leaves the
    // SVG — without it, dragging off the image strands the box mid-edit.
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }

  function onPointerDownBox(e: React.PointerEvent, ann: Annotation) {
    if (e.button !== 0) return
    e.stopPropagation()
    onSelect(ann.id)
    const { x, y } = toImageCoords(e)
    setDrag({ kind: 'move', id: ann.id, offsetX: x - ann.x, offsetY: y - ann.y })
    setPreview({ x: ann.x, y: ann.y, width: ann.width, height: ann.height })
    ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
  }

  function onPointerDownHandle(e: React.PointerEvent, ann: Annotation, handle: Handle) {
    if (e.button !== 0) return
    e.stopPropagation()
    onSelect(ann.id)
    // Anchor = the corner OPPOSITE the one being dragged. Resizing is then just
    // "rectangle between anchor and cursor", which handles dragging past the
    // anchor (inverting the box) for free — no special cases.
    const anchorX = handle === 'nw' || handle === 'sw' ? ann.x + ann.width : ann.x
    const anchorY = handle === 'nw' || handle === 'ne' ? ann.y + ann.height : ann.y
    setDrag({ kind: 'resize', id: ann.id, handle, anchorX, anchorY })
    setPreview({ x: ann.x, y: ann.y, width: ann.width, height: ann.height })
    ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!drag) return
    const { x, y } = toImageCoords(e)

    if (drag.kind === 'create') {
      // Math.abs + Math.min handles dragging in any direction: drawing
      // right-to-left or bottom-to-top produces a normal box, not a negative one.
      setPreview(
        clampRect({
          x: Math.min(drag.startX, x),
          y: Math.min(drag.startY, y),
          width: Math.abs(x - drag.startX),
          height: Math.abs(y - drag.startY),
        }),
      )
    } else if (drag.kind === 'move') {
      const ann = annotations.find((a) => a.id === drag.id)
      if (!ann) return
      setPreview(
        clampRect({
          x: x - drag.offsetX,
          y: y - drag.offsetY,
          width: ann.width,
          height: ann.height,
        }),
      )
    } else {
      setPreview(
        clampRect({
          x: Math.min(drag.anchorX, x),
          y: Math.min(drag.anchorY, y),
          width: Math.abs(x - drag.anchorX),
          height: Math.abs(y - drag.anchorY),
        }),
      )
    }
  }

  function onPointerUp() {
    if (!drag || !preview) {
      setDrag(null)
      setPreview(null)
      return
    }

    if (drag.kind === 'create') {
      // Only commit if it's a real drag. A click (or a 2px twitch) shouldn't
      // litter the dataset with degenerate boxes — the backend rejects them
      // anyway, so this avoids a pointless failed request.
      if (preview.width >= MIN_DRAG && preview.height >= MIN_DRAG && activeClassId !== null) {
        onCreate(preview, activeClassId)
      }
    } else if (preview.width >= 1 && preview.height >= 1) {
      // Only PATCH if the box ACTUALLY moved.
      //
      // Selecting a box is a pointerdown+pointerup inside it, which is
      // indistinguishable from a zero-distance move-drag. Without this guard,
      // merely CLICKING a box to look at it issued a PATCH with its existing
      // coordinates — and because the backend treats any geometry write as a
      // human edit, that silently:
      //   - flipped source "auto" -> "manual", fabricating provenance
      //   - set reviewed=True, so a box nobody approved counted as verified
      // i.e. clicking around the dataset quietly marked it reviewed. Exactly
      // the kind of corruption that doesn't error, it just makes your training
      // data a lie.
      const original = annotations.find((a) => a.id === drag.id)
      const moved =
        !original ||
        Math.abs(preview.x - original.x) > 0.01 ||
        Math.abs(preview.y - original.y) > 0.01 ||
        Math.abs(preview.width - original.width) > 0.01 ||
        Math.abs(preview.height - original.height) > 0.01

      if (moved) {
        // One PATCH on pointerup, not per mousemove. Dragging is local state;
        // the server hears about it once, when you let go.
        onUpdate(drag.id, preview)
      }
    }

    setDrag(null)
    setPreview(null)
  }

  // --- keyboard ----------------------------------------------------------

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Ignore keys while typing in a field, or every shortcut fires while
      // you're renaming something.
      const el = document.activeElement
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT'))
        return

      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId !== null) {
        e.preventDefault()
        onDelete(selectedId)
      }
      if (e.key === 'Escape') onSelect(null)

      // 'y' accepts the selected proposal. Deliberately NOT Enter — that
      // approves the whole image, and having one key mean two different
      // commitments depending on what's selected is how you accept things you
      // didn't mean to.
      if (e.key === 'y' && selectedId !== null && onAccept) {
        const sel = annotations.find((a) => a.id === selectedId)
        if (sel?.proposed) {
          e.preventDefault()
          onAccept(selectedId)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, onDelete, onSelect, onAccept, annotations])

  // --- render ------------------------------------------------------------

  const dragId = drag && 'id' in drag ? drag.id : null

  return (
    <div className="relative inline-block max-w-full">
      {/* The image is a plain <img> UNDER the SVG, not painted into it. The
          browser handles decoding, caching and scaling; the SVG only draws
          boxes. Both are sized identically so they stay registered. */}
      <img
        src={imageUrl}
        alt=""
        // block removes the inline-element baseline gap that would offset the
        // SVG by a few pixels — a real and maddening source of misalignment.
        className="block max-w-full select-none"
        draggable={false}
      />

      <svg
        ref={svgRef}
        // THE KEY LINE: user units == image pixels, at any display size.
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        className="absolute inset-0 h-full w-full"
        style={{ cursor: activeClassId !== null ? 'crosshair' : 'default' }}
        onPointerDown={onPointerDownBackground}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        {/* Accepted boxes first, proposals after, so suggestions always draw ON
            TOP of your work rather than hiding underneath it. */}
        {[...annotations]
          .sort((a, b) => Number(a.proposed) - Number(b.proposed))
          .map((ann) => {
          const isSelected = ann.id === selectedId
          const isDragging = ann.id === dragId
          const rect: Rect = isDragging && preview ? preview : ann
          const color = colorOf(ann.category_id)

          return (
            <g key={ann.id}>
              <rect
                x={rect.x}
                y={rect.y}
                width={rect.width}
                height={rect.height}
                fill={color}
                // Nearly transparent fill, not fill="none": it gives the box an
                // interior to click for moving, while leaving the image visible.
                // Proposals get almost none — they're suggestions layered over
                // your work, and they shouldn't tint the image you're judging.
                fillOpacity={ann.proposed ? 0.04 : isSelected ? 0.18 : 0.08}
                stroke={color}
                // vectorEffect keeps the stroke 2 CSS px regardless of the
                // viewBox scale. Without it, a 4000px-wide image would render
                // hairline borders and a 200px one would render fat ones.
                vectorEffect="non-scaling-stroke"
                strokeWidth={isSelected ? 3 : 2}
                // THREE visual states, and the distinction is the whole point
                // of this screen:
                //   solid       your annotation, accepted
                //   long dash   accepted model output, not yet reviewed
                //   dotted      a PROPOSAL — not an annotation at all yet
                // Dotted vs dashed reads as "lighter, provisional" at a glance,
                // which is exactly what a proposal is.
                strokeDasharray={
                  ann.proposed ? '2 3' : ann.reviewed ? undefined : '6 4'
                }
                strokeOpacity={ann.proposed ? 0.9 : 1}
                onPointerDown={(e) => onPointerDownBox(e, ann)}
                className="cursor-move"
              />

              {/* Label chip. Rendered in a foreignObject-free way (plain SVG
                  text on a rect) so it doesn't inherit page CSS quirks. */}
              <g transform={`translate(${rect.x}, ${rect.y})`}>
                <rect
                  x={0}
                  y={-18}
                  width={Math.max(38, labelWidth(ann, classes))}
                  height={17}
                  fill={color}
                  // A hollow chip for proposals: same colour, but not the solid
                  // block an accepted box gets.
                  fillOpacity={ann.proposed ? 0.35 : 1}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={4}
                  y={-6}
                  fill="white"
                  // Font size in user units would scale with the image; this
                  // keeps labels legible on a 4000px image.
                  style={{ fontSize: 11, fontFamily: 'system-ui', userSelect: 'none' }}
                >
                  {labelText(ann, classes)}
                </text>
              </g>

              {/* Accept / reject affordances, on the selected proposal only.
                  Putting them on every proposal would bury the image under
                  buttons on a busy scene. */}
              {ann.proposed && isSelected && onAccept && (
                <g transform={`translate(${rect.x + rect.width}, ${rect.y})`}>
                  <g
                    transform="translate(-40, 2)"
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      onAccept(ann.id)
                    }}
                    className="cursor-pointer"
                  >
                    <rect width={18} height={16} rx={2} fill="#15803d" />
                    <text x={5} y={12} fill="white" style={{ fontSize: 11 }}>
                      ✓
                    </text>
                  </g>
                  <g
                    transform="translate(-20, 2)"
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      onDelete(ann.id)
                    }}
                    className="cursor-pointer"
                  >
                    <rect width={18} height={16} rx={2} fill="#b91c1c" />
                    <text x={6} y={12} fill="white" style={{ fontSize: 11 }}>
                      ✕
                    </text>
                  </g>
                </g>
              )}

              {/* Resize handles, only on the selected box — showing them on
                  every box turns a busy image into confetti. */}
              {isSelected &&
                (['nw', 'ne', 'sw', 'se'] as Handle[]).map((h) => {
                  const hx = h === 'nw' || h === 'sw' ? rect.x : rect.x + rect.width
                  const hy = h === 'nw' || h === 'ne' ? rect.y : rect.y + rect.height
                  return (
                    <circle
                      key={h}
                      cx={hx}
                      cy={hy}
                      // r in user units WOULD scale with image size, so a handle
                      // on a big image would be a dot. Radius is set via CSS
                      // pixels using vector-effect on the stroke and a fixed r
                      // scaled by the viewBox ratio.
                      r={handleRadius(imageWidth)}
                      fill="white"
                      stroke={color}
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                      onPointerDown={(e) => onPointerDownHandle(e, ann, h)}
                      style={{ cursor: `${h}-resize` }}
                    />
                  )
                })}
            </g>
          )
        })}

        {/* The box being drawn right now. */}
        {drag?.kind === 'create' && preview && activeClassId !== null && (
          <rect
            x={preview.x}
            y={preview.y}
            width={preview.width}
            height={preview.height}
            fill={colorOf(activeClassId)}
            fillOpacity={0.2}
            stroke={colorOf(activeClassId)}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            strokeDasharray="4 3"
            pointerEvents="none"
          />
        )}
      </svg>
    </div>
  )
}

function labelText(ann: Annotation, classes: ProjectClass[]): string {
  const name = classes.find((c) => c.id === ann.category_id)?.name ?? '?'
  // Show confidence only for model output. A manual box has none, and "1.00"
  // would be a fabrication.
  return ann.confidence !== null ? `${name} ${ann.confidence.toFixed(2)}` : name
}

function labelWidth(ann: Annotation, classes: ProjectClass[]): number {
  // Rough advance-width estimate. Measuring text properly needs a canvas
  // context or a DOM round-trip; ~6px/char at 11px system-ui is close enough
  // for a background chip, and being a few px wide is invisible.
  return labelText(ann, classes).length * 6 + 8
}

function handleRadius(imageWidth: number): number {
  // Handles are drawn in user units, so their apparent size shrinks as the
  // image gets bigger. Scale r with the image so it stays roughly 5 CSS px.
  return Math.max(3, imageWidth / 160)
}
