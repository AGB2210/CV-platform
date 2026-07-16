"""
ML integrations package.

Deliberately isolated from the web layer. Nothing in here should import FastAPI,
and nothing here should know what a request is. That boundary is what keeps the
heavy, slow, GPU-bound code swappable.

Planned shape (kept here as a map of where this is going):

    ml/
      annotators/          # Phase 2 — AutoAnnotator implementations
        base.py            #   the AutoAnnotator protocol + a model registry
        grounding_dino.py  #   text prompt -> boxes
        sam.py             #   point/box prompt -> mask -> box (box-assist)
        grounded_sam.py    #   DINO boxes -> SAM refinement -> tighter boxes
      trainers/            # Phase 4 — Trainer implementations
        base.py            #   the Trainer protocol + a registry
        rf_detr.py         #   default
        rt_detr.py
        yolo.py
      inference/           # Phase 5 — architecture-agnostic predict interface

Empty in Phase 0.
"""
