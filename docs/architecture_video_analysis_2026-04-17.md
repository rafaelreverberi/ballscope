# BallScope Video Analysis Architecture

Date: 2026-04-17

Status:
- Implemented as the active offline dual-camera pipeline in `ballscope/ai/offline_analysis.py` and integrated through `ballscope/web/app.py`.

## Goal
Build a post-analysis pipeline for left/right football-camera recordings that produces one seamless broadcast-style output.

The final render should:
- hide the fact that two source cameras are used
- follow the ball smoothly across the full field
- zoom automatically based on context, without a manual zoom input
- degrade gracefully when the ball is temporarily lost
- preserve one shared implementation for Apple Silicon macOS and NVIDIA Jetson

## Current Problems
- The existing offline analysis treats the result as a left-or-right stream selection problem instead of one shared field view.
- A single crop definition is applied to both cameras, which can exclude the relevant area on one side.
- Ball detection is too tightly coupled to the currently selected stream.
- Small-ball detection quality drops quickly when input resolution is reduced too early.
- The final render never forms one spatially continuous scene, so cross-midfield transitions are not broadcast-smooth.

## Target Architecture
The redesigned pipeline is a five-stage system:

1. Input normalization
- Load left and right videos as separate synchronized sources.
- Estimate stream timing alignment from metadata and refine with optional content-based sync later.
- Normalize working FPS for analysis while preserving original output timing metadata.

2. Field registration
- Map each source camera into one shared field/master-canvas coordinate system.
- Start with a practical overlap/blend model.
- Evolve toward homography-based field registration so the center overlap and sideline geometry align robustly.

3. Ball detection and fusion
- Run detection on original-resolution frames or focused ROIs, not on the final broadcast crop.
- Produce one per-camera ball hypothesis stream.
- Fuse left/right hypotheses into one master-canvas ball state using confidence, size, motion continuity, and source agreement.

4. Tracking and recovery
- Use model detections as the primary signal.
- Use a short-gap CPU tracker / motion prior / velocity model when detections drop out briefly.
- When uncertainty rises, widen the virtual camera instead of snapping to a wrong player cluster.

5. Virtual broadcast camera
- Render one virtual camera crop from the master canvas.
- Use automatic zoom rules based on ball speed, ball confidence, local player density, and tactical context.
- Apply deadzones, acceleration limits, and look-ahead to keep the motion TV-like and stable.

## Rendering Strategy
The output render is produced from a master canvas, not from a single raw stream.

### Phase 1
- Build a heuristic blended master canvas from left and right sources.
- Use a configurable overlap ratio and a broad feathered seam. The default blend is intentionally wider than a hard cut so players crossing the midfield overlap remain visible across the transition.
- Add mild seam exposure compensation so the overlap is less visibly split.
- Map detections from each source into master-canvas coordinates.
- Drive one virtual camera crop over that canvas.
- Expose a lightweight full-field debug preview with a red ball marker so stitching and fusion can be inspected during analysis.

### Phase 2
- Replace heuristic overlap with homography-based alignment from field lines / static calibration.
- Store per-camera registration parameters.
- Use field registration for more accurate fusion near the midfield seam.

## Detection Strategy
Detection must remain resolution-aware.

- Never rely only on a low-resolution whole-frame pass for small-ball analysis.
- Use a two-stage policy:
  - expensive global reacquire on original or near-original resolution
  - cheaper local ROI detection around the predicted ball location
- Keep the ROI detector active most of the time once a lock exists.

Recommended policy:
- full-frame reacquire scans on original-resolution source frames at lower cadence
- ROI-based detection around the predicted position at higher cadence
- short-gap CPU prediction when no confident model result is available

Implemented policy details:
- full-frame reacquire is per-camera and never uses aggressive downscaling
- ROI detection is the common path while locked
- full-frame cadence becomes more aggressive during `LOST_SHORT` and `LOST_LONG`
- left/right scans can alternate so the expensive path does not fire on every source every frame

## Fusion Strategy
Each frame produces zero or more hypotheses:
- left stream detection
- right stream detection
- predicted-only fallback hypothesis

The fused master state is chosen by:
- confidence
- continuity with previous position and velocity
- source image scale / object size
- consistency between adjacent frames
- source visibility quality in overlap regions

Current implementation:
- per-camera hypotheses are scored in master-canvas space with confidence, scale sanity, and continuity
- close left/right detections can be merged into one fused master hypothesis
- predicted hypotheses are allowed only for short gaps and are penalized relative to real detections

## Virtual Camera Strategy
The virtual camera should behave like a sports broadcast operator.

### Rules
- Keep the ball near center, but not with jitter.
- If confidence is high and the ball is small, zoom in more.
- If confidence drops, zoom out gradually.
- If the ball approaches the seam or is in transition play, keep a slightly wider tactical view.
- Avoid rapid left-right micro-corrections with a deadzone.
- Use velocity look-ahead so the crop arrives before the ball reaches the edge.

### Inputs
- fused ball position
- fused ball velocity
- confidence
- local player density around the ball
- uncertainty level

Current implementation:
- `TRACKED`: follow the fused ball with deadzones, acceleration limits, and mild look-ahead
- `HOLD_SHORT`: stay near the last safe region and widen slightly
- `LOST_SHORT`: keep bias toward the last safe region while blending toward a wider tactical view
- `LOST_LONG` / `UNKNOWN`: transition to a wide tactical master-canvas framing

## Implementation Phases
### Phase A: Foundation
- Add a dedicated master-canvas assembly module.
- Add coordinate mapping from source frames into master space.
- Enable left/right fusion mode by default for dual-video analysis.
- Render the final crop from the master canvas instead of a single chosen source.

### Phase B: Better detection control
- Separate per-stream crop hints.
- Add explicit reacquire cadence and ROI cadence.
- Add per-stream debug counters.

### Phase C: Registration upgrade
- Add calibration data structures.
- Add field-line-assisted registration / homography.
- Replace heuristic seam alignment with calibrated projection.

### Phase D: Broadcast polish
- Better zoom rules
- tactical wide view on uncertainty
- event-aware framing near goals / corners / throws

## Testing Requirements
For each substantial phase:
- syntax-check all edited modules
- verify `python main.py` startup assumptions still hold
- verify device resolution still respects `BALLSCOPE_AI_DEVICE=auto`
- test left/right dual-source analysis on a short clipped segment
- verify the final output stays continuous at midfield transitions
- verify the final output still works on Apple Silicon and does not introduce CUDA-only assumptions

## Risks
- Incorrect time sync between left/right videos will create false seam jumps.
- Over-aggressive downscaling will destroy small-ball detectability.
- Heuristic seam blending without calibration may still create visible parallax artifacts.
- Tracking fallback can drift onto players if not bounded by strong reacquire logic.

## Immediate Build Decision
Delivered baseline:
- heuristic blended master canvas
- mapped left/right detections
- one virtual camera crop from master space
- default fusion enabled for dual-video analysis
- fixed-layout seam calibration from multiple early video samples

Next iteration target:
- move from overlap heuristics to explicit calibrated field registration / homography when calibration data is available.
