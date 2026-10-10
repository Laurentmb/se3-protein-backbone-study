# Attribution

The project implements a standard explicit Lie midpoint integrator and provides
wrappers, paired experiment orchestration, validation, scores and reports.
Its development used AI coding assistance. No new pretrained model or new
numerical integration method is claimed.

## Model sources

| Component | Source used | Role |
|---|---|---|
| FrameFlow | microsoft/protein-frame-flow, commit `f50d8dbbdae827be291e9f73d732b61b195f8816` | Fixed protein backbone generator |
| FrameDiff | jasonkyuyim/se3_diffusion, commit `53359d71cfabc819ffaa571abd2cef736c871a5d` | Initial generation example; bundled ProteinMPNN code |
| ProteinMPNN | ProteinMPNN directory of that FrameDiff commit, `v_48_020` | Sequence design |
| ESMFold | Public ESM Atlas API, response titles identify ESMFold V1 | Recorded structure predictions |
| TM-align | tmtools 0.2.0 | Structural similarity scores |

Original projects: [FrameFlow](https://github.com/microsoft/protein-frame-flow),
[FrameDiff](https://github.com/jasonkyuyim/se3_diffusion),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[ESM](https://github.com/facebookresearch/esm),
[tmtools](https://github.com/jvkersch/tmtools).

Upstream FrameFlow, FrameDiff and ProteinMPNN MIT notices are retained in
`licenses/`. Third-party libraries and downloaded checkpoints retain their
respective upstream terms. Their repositories and model weights are not vendored
in this public bundle. The small `.pt` files under `results/` contain frame
states, not pretrained model weights.

FrameDiff sampling methods are loaded from the pinned upstream training class
without changing their function bodies. FrameFlow's architecture and state
dictionary are unchanged. Index constants are cast to int64 for Windows
compatibility. The controlled sampler's self-conditioning and terminal-time
choices are documented in `docs/METHODS.md`.

The ESM Atlas service does not expose a checkpoint hash or its internal inference
settings. Saved responses reproduce the analysis; a fresh call may give different
results. Scientific model references are linked in the main README.
