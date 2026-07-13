# Shot Zones — Configurable Partitions

Shot location is stored as a continuous hoop-centred court position (metres). Zones are a
**view over that position**, not a measurement: `bball.lift.zones` turns any taxonomy into a
pure lookup, so changing taxonomies **re-buckets every historical shot instantly** — no video
reprocessing (`ZonePartition.rebucket`).

## Modes

| Mode | What it is | Construction |
|---|---|---|
| Presets | `basic3` (interior ≤ 7 ft arc / midrange / three) · `extended` (adds short-/long-mid split and **deep-three**) · `spots` (the classic chart: interior + corner/wing/top × mid/three per side) | `preset_basic3(court)`, `preset_extended(court)`, `preset_spots(court)` |
| Parametric | Every preset boundary is a number you can change (interior radius, mid split, deep-three offset, sector angles), serialized to YAML/JSON | `part.to_dict()` ↔ `from_dict(d)` |
| Freeform | Arbitrary polygons; screen-drawn shapes are lifted through the session homography once, then live in court space | `lift_screen_polyline(H, pts)` → `from_polygons(...)` |

## Design decisions worth knowing

- **Zones live in court coordinates, never screen coordinates.** A screen drawing is lifted
  via the calibration homography at creation time; after that the partition is
  camera-independent — move the camera, recalibrate, same zones, and stats stay comparable
  across sessions.
- **Deep-three is an offset of the true 3-point shape, not a bigger circle.** The 3PT line
  is an arc *plus straight corner segments*; a radial threshold misclassifies the corner
  (where `hypot(x, y)` never reaches the apex radius). `distance_beyond_three` and
  `offset_three_polyline` handle the shape correctly, including the recomputed
  corner-to-arc transition. The unit tests pin exactly the points where a radial rule and
  the offset rule disagree.
- **Every boundary carries the on-the-line band.** A shot within `band_m` (default 15 cm)
  of any boundary is flagged `on_line` with the boundary named — a category read inside the
  location-error bar is a guess, and the UI should say so rather than silently bucketing.
- **Boundary reliability from the error model.** `compose_with_error_map(partition, err_fn)`
  samples each boundary against a location-error field (the A7 Monte-Carlo P90 field for
  the current camera placement) and scores the fraction of the boundary where the error
  exceeds the band — powering warnings like *"your corner-three boundary is not
  trustworthy from this camera position."* This composes two things the project already
  measures (calibration error geography × user taxonomy) into a user-facing guarantee.

## Example

```python
from bball.lift.court_model import get_court
from bball.lift import zones

court = get_court("hs")                      # or "nba" / "fiba" / custom measurements
part = zones.preset_extended(court, deep_three_offset_m=0.75)
result = part.classify(x=6.9, y=0.5)         # corner three, radial rules would say midrange
relabeled = part.rebucket(stored_shots)      # retroactive view change, free
```
