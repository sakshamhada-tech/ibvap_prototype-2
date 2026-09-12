# Controlled long-range evaluation

This directory documents the reproducible, metadata-only evaluation workflow for the
Astra I Smart India Hackathon demo. It does **not** contain performance claims or
representative border footage.

## What the harness measures

On the same labelled source frames, `scripts/evaluate_long_range.py` reports:

- person/vehicle recall by source target-height band and daylight/low-light label;
- unmatched detections per minute when every frame in the interval is labelled;
- recall from all detector proposals separately from authoritative full-frame ByteTrack recall;
- target and observation track-acquisition rates;
- longest uninterrupted track duration and ID switches per labelled object;
- median, low-percentile, and wall-clock processing FPS;
- loitering/group event precision, recall, scenario breakdown, and alert latency when event windows are labelled; and
- wide-only versus focus-enabled deltas.

The fixed reporting bands are `<=24 px`, `25-48 px`, `49-96 px`, and `>96 px`.
They describe this dataset; they are not universal operating-range claims.

## 1. Prepare consented footage and labels

Copy `annotation_template.json` outside Git or into `evaluation/results/`, then replace
its placeholder values. `source_id` is a non-sensitive name shared by annotations and
both traces. Never place a camera URL, credentials, faces, number plates, or personal
names in it. The report fails closed if decoded dimensions, frame numbers, one-way
source-frame SHA-256 fingerprints, or shared model/context/hardware/build settings
differ between runs.

With `exhaustive_frames: true`, annotations must include every compared trace frame
across one continuous interval. Only then can unmatched detections for the source clip
and per minute be calculated. Set it to `false` for sparse frame sampling; recall still
works, but false detections per minute are reported as unavailable.

Set `source_width` and `source_height` to the decoded frame dimensions. Bounding
boxes are source coordinates `[x1, y1, x2, y2]` and must remain inside that frame.
Keep the same `object_id`
for the same consented target within one source only. It is an annotation key—not a
biometric identity or a proposed runtime re-identification feature.

Optional event windows use this shape:

```json
{
  "event_id": "loiter-window-01",
  "event_type": "SUSPICIOUS_LOITERING",
  "expected": true,
  "eligible_frame": 450,
  "end_frame": 650,
  "scenario": "stationary-person"
}
```

For group testing, use separate labelled windows with scenarios such as `toward`,
`stationary`, `parallel`, and `away`; mark only the intended toward-fence windows as
`expected: true`. Set `events_exhaustive` to true only if all relevant alert windows are
labelled. Event matching is clip-window based and never claims identity.

## 2. Collect the wide-only baseline

Use the same machine, source, model hash, confidence, fence, and context settings in
both runs. The baseline deliberately disables both tiling and native focus:

```bash
IBVAP_ENABLE_TILED_INFERENCE=false \
IBVAP_ENABLE_NATIVE_FOCUS=false \
IBVAP_EVALUATION_BUILD_LABEL=git-commit-or-release \
IBVAP_EVALUATION_HARDWARE_LABEL=presentation-laptop-description \
python scripts/evaluate_long_range.py collect \
  --mode wide-only \
  --source /absolute/path/to/consented-sequence.mp4 \
  --source-id campus-sequence-01 \
  --output evaluation/results/campus-wide.json
```

## 3. Collect the focus-enabled run

Run the exact same source from frame one. Focus mode requires both the native-tile
proposal path and the bounded focus bridge:

```bash
IBVAP_ENABLE_TILED_INFERENCE=true \
IBVAP_ENABLE_NATIVE_FOCUS=true \
IBVAP_EVALUATION_BUILD_LABEL=git-commit-or-release \
IBVAP_EVALUATION_HARDWARE_LABEL=presentation-laptop-description \
python scripts/evaluate_long_range.py collect \
  --mode focus-enabled \
  --source /absolute/path/to/consented-sequence.mp4 \
  --source-id campus-sequence-01 \
  --output evaluation/results/campus-focus.json
```

The collector stores metadata only: source dimensions, frame numbers, one-way SHA-256
fingerprints of decoded source frames, boxes, confidence, short-lived runtime track
IDs, focus/context diagnostics, sanitized alert types, and processing times. Frame
fingerprints enforce exact input parity without retaining pixels; keep traces local
because they still describe a surveillance sequence. The collector does not store
frames, free-form alert details, camera URLs, or credentials. Generated traces/reports
under `evaluation/results/` are ignored by Git.

## 4. Produce JSON and presentation tables

```bash
python scripts/evaluate_long_range.py report \
  --annotations evaluation/results/campus-annotations.json \
  --wide-trace evaluation/results/campus-wide.json \
  --focus-trace evaluation/results/campus-focus.json \
  --json-output evaluation/results/campus-report.json \
  --markdown-output evaluation/results/campus-report.md
```

Add `--overwrite` only when intentionally replacing an existing trace or report.
Bounds, IoU matching, warmup frames, low-FPS percentile, and maximum JSON size are
validated `IBVAP_EVALUATION_*` settings documented in `.env.example`.

## Interpretation rules

- Detection recall includes matched full-frame and supplemental tile observations.
- Authoritative recall includes only full-frame detections carrying a ByteTrack ID.
- Target acquisition rate is the share of detected labelled targets that receive an
  authoritative ID; acquisition delay runs from first matched detection to first
  authoritative match.
- Continuity counts consecutive source frames with the same authoritative track ID;
  an interruption or changed ID starts a new segment.
- Processing FPS is per-frame pipeline time after initialization. Wall throughput also
  includes source reads, frame fingerprinting, and shutdown, but excludes model startup.
- Event latency runs from the labelled eligible frame to the first matching alert frame,
  divided by source FPS.
- A focus candidate episode is never counted as a persistent identity.
- Digital focus may preserve pixels discarded by whole-frame resizing; it cannot create
  sensor detail.
- Publish abstentions, false detections, ID switches, and runtime cost alongside gains.
- Record hardware, software commit, source resolution, lighting, weather, compression,
  and measured distance separately before using metres in a presentation.
- Use at least one empty negative segment and toward/parallel/away group scenarios.
- Repeat timed runs on presentation hardware; one controlled clip is not a field claim.
