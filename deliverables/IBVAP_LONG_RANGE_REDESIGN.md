# Astra I long-range redesign for a Smart India Hackathon demo

**Date:** 10 September 2026  
**Repository baseline:** `6332ed9`  
**Objective:** redesign the prototype around long-range person/vehicle triage from existing camera feeds while keeping every alert subject to human review.

## Executive decision

The current `yolov8n.pt` full-frame detector should not be presented as a dependable long-range border detector. Its context rules receive no evidence when the full-frame detector fails to create a stable track, which is exactly what the low-resolution CAVIAR trial exposed.

The proposed demo should instead present Astra I as a **vendor-neutral attention and evidence layer** for existing day/night CCTV and PTZ feeds:

1. ingest the camera's highest useful native-resolution stream;
2. scan the wide view cheaply;
3. identify candidate regions using motion and cadence-gated sliced inference;
4. reprocess native source pixels from those regions at higher model scale;
5. require temporal confirmation and an authoritative short-lived track;
6. display the wide view and a clearly labelled digital focus crop together; and
7. emit explainable review alerts without automated response.

This is a credible software-only improvement when a high-resolution source already contains useful pixels that normal full-frame inference discards by resizing. It is not a substitute for optical zoom, thermal sensing, or adequate pixel density.

## What publicly available Indian sources support

Public Government of India sources describe a heterogeneous surveillance stack rather than one dominant camera model:

- A March 2025 PIB response lists Hand Held Thermal Imagers, night-vision devices, UAVs, CCTV/PTZ cameras, IR sensors, and CIBMS on the Indo-Bangladesh border: <https://pib.gov.in/PressReleseDetailm.aspx?PRID=2110805>
- A July 2023 PIB response additionally names LORROS, battlefield-surveillance radar, CCTV/PTZ cameras, IR sensors, infrared alarms, and command-and-control integration: <https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1942053&reg=48&lang=2>
- A December 2025 PIB response describes drones, thermal imagers, night-vision devices, sensors, and real-time radar/optical systems under an integrated surveillance approach: <https://www.pib.gov.in/PressReleasePage.aspx?PRID=2200966&reg=6&lang=1>
- MHA's 2023–24 annual report describes CIBMS as integration of manpower, sensors, networks, intelligence, and command-and-control solutions: <https://www.mha.gov.in/sites/default/files/AnnualReport_27122024.pdf>
- BEL publicly lists BOSS, Prahari, and BELROS electro-optical systems. BELROS combines day, thermal, SWIR, laser range-finding, compass, GPS, and IRNSS-compatible components. These product pages are useful interoperability references, but they are not proof that a particular model dominates national deployment: <https://bel-india.in/product/be-long-range-observation-system-belros/> and <https://bel-india.in/product/prahari/>

Therefore the SIH presentation should say **"aligned with publicly documented Indian CCTV/PTZ, day/night, thermal, and CIBMS-style command feeds"**, not **"compatible with the camera used most by BSF"**. No reliable public inventory establishes a single most-used model.

## Can software zoom obtain more pixels?

### What it cannot do

Digital zoom after capture only crops and enlarges pixels already recorded. It cannot recover texture, edges, or class evidence that never reached the sensor. Upscaling a five-pixel-wide person or a few-pixel object may make it larger on screen, but not more informative.

Generative super-resolution must not be used as detection evidence. It can invent plausible detail and would make the result unsuitable for an accountable surveillance demo.

### What it can do

Software can avoid throwing away source pixels. A normal detector often resizes an entire 1920×1080 or 3840×2160 frame into a roughly 640-pixel model input. A native-resolution crop from a suspected area can occupy much more of the model input than it did during the wide pass.

This is the useful form of "software zoom":

- crop from the original native frame, never from the dashboard JPEG;
- include contextual padding around the candidate;
- resize that crop for a second detector pass;
- map the result back into original frame coordinates;
- keep the crop labelled **DIGITAL FOCUS — SOURCE PIXELS ONLY**; and
- compare wide-pass and focused-pass confidence transparently.

The existing tiled inference is an early form of this technique. The SAHI paper reports that sliced inference can improve small-object average precision on its evaluated aerial datasets, while also making clear that tiny objects contain limited detail and slicing increases computation: <https://arxiv.org/abs/2202.06934>. Those reported gains must not be copied as an Astra I performance claim; Astra I needs its own footage and measurements.

If a PTZ camera supports optical zoom, software can later request real optical magnification through a vendor adapter or ONVIF. ONVIF defines absolute, relative, and continuous pan/tilt/zoom operations: <https://www.onvif.org/specs/2306/ONVIF-PTZ-Service-Spec-v2306.pdf>. The SIH demo can simulate focus with a crop, but must distinguish that from optical PTZ control.

## Redesigned architecture

```text
High-resolution RTSP/file input
            │
            ├──────────────► low-bandwidth dashboard preview
            │
            ▼
Native Frame Buffer (bounded, newest frame wins)
            │
    ┌───────┴─────────────────────────────┐
    ▼                                     ▼
Wide detector + authoritative tracker     Candidate proposer
(pretrained person/vehicle classes)       motion + scheduled native tiles
    │                                     │
    │                            bounded candidate queue
    │                                     ▼
    │                              Native ROI focus pass
    │                         crop + context, no invented pixels
    │                                     │
    └────────────────┬────────────────────┘
                     ▼
       Source-coordinate evidence fusion
      class agreement + confidence + time
                     │
                     ▼
     Authoritative short-lived track state
                     │
        ┌────────────┼───────────────┐
        ▼            ▼               ▼
      fence       loitering       group approach
        └────────────┼───────────────┘
                     ▼
       explainable contextual review score
                     │
                     ▼
       dashboard + CSV + bounded alarm queue
```

### 1. Native frame intake

- Request the highest useful camera stream for analytics and use a separate low-resolution stream or encoded derivative for the dashboard.
- Keep only the newest frame in each expensive worker queue; stale frames are dropped rather than allowed to accumulate.
- Record source dimensions and compression metadata in diagnostics.

### 2. Wide detector

- Keep pretrained COCO person and vehicle classes as the no-training baseline.
- Benchmark `yolov8n.pt` against stronger checksum-pinned pretrained candidates rather than assuming that a larger model is automatically better.
- Make detector input size explicit. A 1280-pixel input can preserve more source detail than 640, but costs substantially more compute.
- Explicitly select ByteTrack instead of relying on Ultralytics' default tracker. Ultralytics documents BoT-SORT as the default and requires `tracker="bytetrack.yaml"` to choose ByteTrack: <https://docs.ultralytics.com/modes/track/>.
- Keep re-identification disabled, matching the project scope.

### 3. Candidate proposer

Candidate generation should not require a semantic person detection:

- frame differencing or background subtraction for a fixed camera;
- optical-flow clusters where camera motion is controlled;
- cadence-gated overlapping native-resolution tiles; and
- existing tracks that have become low-confidence.

Motion candidates mean only "region worth inspecting." They never independently generate a person, vehicle, loitering, or risk alert.

### 4. Native ROI focus pass

- Select at most a configurable small number of regions per cycle.
- Expand each candidate by configurable padding to retain body/vehicle context.
- Crop from the original frame and run a stronger or larger-input pretrained detector.
- Map boxes back to source coordinates and merge them with class-aware NMS.
- Publish wide-pass score, focus-pass score, source box size, and evidence age.
- Run in a bounded single-slot worker so capture and the dashboard cannot block.

### 5. Temporal evidence and tracking

- A focused detection must appear consistently across configurable frames before it becomes a confirmed review candidate.
- Focused detections must be acquired by an actual short-lived tracker before they enter loitering or group analytics; the system must not fabricate a persistent identity.
- Track diagnostics should expose tracked, untracked-wide, untracked-tile, ID-switch, and longest-continuous-track counts.
- Loitering diagnostics should expose observed duration, movement spread, scope state, and the exact blocking reason.

### 6. Optional PTZ adapter

For compatible existing PTZ cameras:

- discover capabilities without assuming ONVIF support;
- request optical zoom only after sustained candidate confirmation;
- rate-limit movement and enforce pan/tilt/zoom bounds;
- suspend geometry-based alerts while the camera moves;
- reacquire scene calibration at a preset before resuming fence analytics; and
- always retain operator override.

This phase is not required for the software-only SIH demo. The demo UI should show where an ONVIF adapter would replace the digital crop with true optical evidence.

### 7. Future thermal/radar adapter

India's publicly documented approach is multi-sensor. The interface should permit later ingestion of:

- thermal candidate boxes;
- radar range/azimuth tracks;
- IR or fence-sensor events; and
- UAV observations.

For the current demo these are typed mock inputs, clearly labelled **SIMULATED SENSOR EVENT**, not fabricated claims of deployed hardware integration.

## SIH controlled-demo design

### Demo footage

Capture a consented 4K or 1080p fixed-camera sequence on a campus or private ground:

1. empty-scene negative segment;
2. one distant person entering and then standing;
3. three people walking together toward a marked virtual fence;
4. a car entering and crossing the line; and
5. movement parallel to or away from the line as a group-approach negative.

Do not describe distance from visual estimation alone. Measure it on site if metres are shown, and always report source target height in pixels.

### Live comparison

The dashboard should show:

- **Wide pass:** original frame, configured model input size, candidate box and score;
- **Digital focus:** native crop, source-pixel dimensions, focused score and confirmation count;
- **Track health:** current ID, continuous duration and missed-frame count;
- **Context reasons:** fence distance, movement direction, loiter duration/spread, group membership and bounded score; and
- **Limit label:** digital focus does not add optical detail.

### Demo sequence

1. Run wide inference at the baseline model size.
2. Show a weak or missed distant candidate.
3. Enable candidate-guided native ROI focus.
4. Show whether confidence and tracking improve on the exact same source frames.
5. Hold the track long enough to generate a loitering review alert.
6. Run the three-person approach clip and show positive and away-from-fence negative cases.
7. Export the CSV evidence and the range/pixel-size evaluation table.

A failed focus pass should be shown as an abstention, not hidden.

## Evaluation required before the presentation

Create a small labelled test set with at least positive and negative sequences. Report:

| Metric | Required breakdown |
|---|---|
| Person/vehicle recall | target-height pixel bands and daylight/low-light |
| False detections | per minute and source clip |
| Track acquisition | percentage of detected targets that receive an ID |
| Track continuity | longest uninterrupted duration and ID switches |
| Loitering | event precision, recall and alert latency |
| Group approach | toward-fence positives versus stationary/parallel/away negatives |
| Runtime | median and low-percentile FPS for wide-only and focus-enabled modes |
| Focus value | wide-only versus focused recall on identical frames |

Recommended target-height bands for reporting are dataset bins, not universal capability claims: `<=24 px`, `25–48 px`, `49–96 px`, and `>96 px`.

## Implementation phases

### Phase 0 — stop misleading failures

- Explicitly configure ByteTrack.
- Add full-frame versus tile provenance to visible labels.
- Add track-acquisition and loiter-blocking diagnostics.
- Add configurable detector input size.
- Test with a consented high-resolution clip rather than CAVIAR.

### Phase 1 — software-only focused inference

- Add motion/native-tile candidate proposals.
- Add the bounded native ROI focus worker.
- Add temporal confirmation and source-coordinate fusion.
- Add the dashboard digital-focus panel.
- Keep all new work disabled by default.

### Phase 2 — benchmark pretrained candidates

- Acquire each model explicitly with source, licence and SHA-256 metadata.
- Compare speed and pixel-band recall on the same labelled frames.
- Select the smallest model meeting the demo acceptance criteria.
- Do not silently download weights at runtime.

### Phase 3 — existing-camera integration

- Add vendor-neutral RTSP profiles.
- Add an optional bounded ONVIF PTZ adapter.
- Add camera-movement state and calibration-preserving presets.

### Phase 4 — field-pilot path

- Add thermal/radar event schemas.
- Collect representative footage with authorization.
- Validate by terrain, weather, lighting, range and target pixel density.
- Conduct security, privacy, licensing and operational reviews.

## Presentation wording

### Safe claim

> Astra I is a vendor-neutral software attention layer aligned with publicly documented Indian CCTV/PTZ and CIBMS-style command architectures. It preserves native camera pixels through candidate-guided sliced and focused inference, confirms evidence over time, and provides explainable human-review alerts.

### Do not claim

- "AI creates extra pixels through zoom."
- "Reliable person identification at 100 metres from any CCTV camera."
- "Compatible with the camera most commonly used by BSF."
- "Works in all weather with RGB only."
- "Contextual risk predicts intent."
- "Tiling or super-resolution recovers missing optical evidence."
- "Field-ready border surveillance."

## Bottom line

A credible SIH innovation is not magical zoom. It is **candidate-guided preservation of native pixels, transparent temporal confirmation, and vendor-neutral integration with the kinds of CCTV/PTZ and multi-sensor command systems that official Indian sources publicly describe**.

The current repository already has pieces of this idea—full-frame tracking, tiled inference, source-coordinate remapping, NMS, context rules, human-review alerts, and bounded optional services—but it lacks the bridge from a tile candidate to an authoritative track, the focused evidence panel, and the diagnostics needed to prove why an alert did or did not occur. Those are the highest-value implementation targets for the redesigned demo.
