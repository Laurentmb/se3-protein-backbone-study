# Project history

1. **Initial generation check:** one 100-residue backbone from FrameDiff and one
   from FrameFlow, using their published checkpoints. Both files passed parsing
   and basic geometry checks. Their unequal budgets make them exploratory examples.
2. **Pipeline and integrator verification:** design and refold two sequences for
   one FrameFlow backbone; verify scoring; test Lie midpoint against an exact
   analytic path; connect the integrators to the real FrameFlow network. The
   Euler adapter matched upstream output within 1.595e-6 A before PDB rounding.
3. **Paired pilot:** five exact initial states, two network budgets, two methods,
   two sequences per backbone, 40 saved ESMFold predictions. Analyse quality and,
   separately, numerical deviation against checked reference integrations.
4. **Repository preparation:** consolidate executable scripts, make paths portable,
   preserve experiment evidence, add CPU reproduction, tests and CI configuration.

The main research question evolved from a FrameDiff/FrameFlow comparison to an
integrator comparison with one fixed FrameFlow checkpoint. Midpoint is a standard
method; the contribution is its integration, validation and controlled evaluation.

`results/history/` preserves earlier data and source snapshots. Its archived
scripts describe those runs; the supported entry point is `scripts/project.py`.
Personal workspace prefixes in public text records were replaced by portable
locations. Raw PDB responses, frame tensors and measured numerical values were
preserved. `results/manifest.json` describes the public copy's hashes.

Open questions remain: diagnose the unconfirmed numerical reference, investigate
self-conditioning, and broaden the sample before claiming a general quality ranking.
