"""
Business logic layer.

Route handlers stay thin: parse the request, call a service, shape the response.
The actual work — writing files, running a model, building COCO JSON — lives
here. That separation is what makes the logic testable without spinning up an
HTTP server, and what lets a background job call the same function an endpoint
calls.

The rule that keeps this honest: nothing in here raises HTTPException. Services
raise their own domain errors (e.g. `storage.ImageRejected`) and the route
decides what HTTP status that deserves. A service that knows about status codes
can only ever be called from a web request.

  storage.py  — image validation, disk layout, zip extraction
"""
