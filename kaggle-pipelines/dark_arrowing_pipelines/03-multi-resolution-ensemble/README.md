Multi-Resolution Ensemble Pipeline (VERY IMPORTANT)

## Why This Pipeline Matters

EDA showed:
different taxa operate at different temporal scales.

Examples:
- insects → short repetitive chirps,
- birds → medium melodic phrases,
- frogs → sustained pulses.

A single window size loses information.

---

## Pipeline Structure

Train separate models on:
- 1-second crops,
- 5-second crops,
- 30-second crops.

Then ensemble predictions.

---

## Example

| Window | Captures |
|---|---|
| 1s | transient chirps |
| 5s | standard calls |
| 30s | habitat context + long phrases |

---

## Ensemble Methods
- mean probability,
- weighted averaging,
- stacking meta-model.

---

## Why It Fits the EDA

EDA strongly showed:
- acoustic heterogeneity,
- varying temporal dynamics,
- diverse taxa behavior.

This pipeline directly addresses those properties.

---

## Practical Benefit
This often improves:
- recall,
- robustness,
- rare species detection,
- noisy soundscape performance.

---