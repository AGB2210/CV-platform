"""
Business logic layer.

Route handlers should stay thin: parse the request, call a service, shape the
response. The actual work — writing files, running a model, building COCO JSON —
lives here. That separation is what makes the ML code testable without spinning
up an HTTP server, and what will let a background worker call the exact same
function an endpoint calls.

Empty in Phase 0.
"""
