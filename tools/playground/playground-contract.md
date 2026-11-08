# Mora-3 local playground serving contract

The launcher listens only on `127.0.0.1`. It accepts the exact `Host` it prints and requires that origin on POST. Browser requests use same-origin relative URLs with credentials omitted, redirects rejected, and no CORS. The launcher reads checkpoint directories only from explicitly supplied paths. The directory picker lists opaque IDs and folder names. It neither uploads weights nor accepts arbitrary paths from a request.

- `GET /api/runtime` returns the `mora3-playground-v1` protocol, `state`, `detail`, `can_select`, `checkpoints` (entries `{id, label}`), and `active`. States are `preview`, `unavailable`, `select`, and `ready`. An active entry includes `id`, `revision`, `model`, `checkpoint`, and `outputs`.
- `POST /api/checkpoint` accepts JSON `{ "id": "c1" }` using an ID from the status response. The adapter verifies and loads the directory. Success returns the updated runtime state. A failed selection preserves any previously loaded checkpoint.
- `POST /api/infer` accepts a request matching the schema below. Send the active `revision` in `X-Mora3-Revision` to prevent a stale tab from sending input to another checkpoint. An unsupported capability or invalid input returns a JSON `{ "error": "…" }` response with a non-success HTTP status. Synthetic examples are separate browser actions.

Use `--adapter module:factory` to select trusted local Python code. The factory accepts a resolved `pathlib.Path` and returns an object with `capabilities()` and `infer(request)`. Capabilities declare `model`, `checkpoint` and `outputs`, where each output has `id`, `tasks` and optional `notes`. The browser enables model execution only for an advertised task and output. The adapter owns artifact identity, safe loading, route-specific validation, and result provenance.

## Request

Use protocol `mora3-playground-v1`. Supply `task` as `predict` or `generate`, the `output` ID, ordered `messages` with `role` and `content`, and `inputs`. Each input contains `modality`, original `name`, `bytes`, `encoding` set to `base64`, and `data`. The launcher accepts up to 20 turns, 8 files, 6 MiB per file, and 12 MiB total file bytes. Adapters can impose smaller limits.

The current native foreground adapter accepts `task: "predict"`, `output: "image"`, `messages: []`, and exactly one `modality: "image"` input. The input must be an 8-bit grayscale PNG at most 64 × 64 pixels and 64 KiB. It returns foreground probabilities rounded to 0–255 in the original pixel frame. This diagnostic route does not establish full-model image generation or trained biological performance.

## Result

Return `protocol`, the requested `output`, a `summary`, human-readable `provenance`, and the matching `format`. Include the actual model and checkpoint identity in provenance. Results must fit the 1 MiB display limit. **Open result JSON** accepts this format without model serving.

| Output IDs | Format | Data fields |
|---|---|---|
| affinity, annotation, expression, bioelectric, reaction | table | `columns`, rectangular `rows` of scalar cells |
| answer, sequence, molecule | text | `text` with FASTA for sequence or SMILES rows for molecule |
| structure | coordinates | `atoms` as `[element, x, y, z]` in Å, with frame in the summary |
| dynamics | series | `xLabel`, `yLabel`, numeric `points` as `[x, y]`, with time or step semantics |
| image | raster | Rectangular `pixels` of integers from 0 to 255, up to 64 × 64, with channel or label semantics |
