# SE(3) Protein Backbone Sampling

A small, reproducible study of **Euler versus Lie midpoint** sampling with a
fixed pretrained FrameFlow model.

**Question:** at the same number of neural-network evaluations, does midpoint
produce more self-consistent protein backbones?

## Main steps

1. **Check the integrators.** Use an analytic trajectory to verify convergence,
   valid rotations and behaviour under a global rigid transformation.
2. **Generate backbones.** Run Euler and midpoint from exactly the same initial
   positions and rotations, at equal network-evaluation budgets.
3. **Design sequences.** ProteinMPNN proposes two amino-acid sequences for each
   generated backbone. The input's placeholder polyalanine is excluded.
4. **Predict structures.** ESMFold predicts a structure for each designed
   sequence. The recorded experiment used the public ESM Atlas API.
5. **Evaluate and compare.** Align each prediction to its source backbone,
   calculate CA RMSD and TM-score, and compare matched initial states.

```text
Shared initial frames → FrameFlow + integrator → backbone
                        → ProteinMPNN → sequences → ESMFold
                        → alignment and self-consistency scores
```

For each backbone, **scRMSD** is the lowest RMSD among its two predictions.
Lower is better. The pilot's success criterion is **scRMSD < 2 Å**.
This measures computational self-consistency, not experimental folding or function.

## Recorded pilot

Five initial states, 100 residues, two methods and two budgets:
**20 backbones and 40 structure predictions**, all successfully evaluated.
One final clean prediction is included in each budget: Euler uses 20/100
integration steps, while midpoint uses 10/50 steps with two evaluations per step.

| Network evaluations | Method | Backbones with scRMSD < 2 Å | Median scRMSD (Å) |
|---:|---|---:|---:|
| 21 | Euler | 2/5 | 2.147 |
| 21 | Lie midpoint | 2/5 | 2.265 |
| 101 | Euler | 4/5 | 1.137 |
| 101 | Lie midpoint | 3/5 | 1.778 |

**Outcome:** midpoint improves some cases, but this pilot does not establish a
systematic quality improvement. At 101 evaluations, Euler has lower scRMSD in
four of the five paired cases. Five initial states are insufficient for a
general solver ranking.

![All paired quality results](results/pilot/paired_quality.png)

A separate numerical experiment disables self-conditioning and compares the
integrated frames to refined references. One reference passes our checks; the
other remains unconfirmed. Its deviations are explicitly provisional.
See [methods and limitations](docs/METHODS.md) for the full protocol and scores.

## Reproduce the analysis on CPU

Use Python **3.10** in a virtual environment; activation instructions are in
[setup](docs/SETUP.md). From this repository's root:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/project.py verify
python scripts/project.py test-metrics
python scripts/project.py analyze
```

The last command reads the included PDB and FASTA files, recomputes **all 40
candidate scores**, checks them against the recorded scores, and writes tables,
figures and `REPORT.md` to `runs/reproduced-pilot/`.
It needs **no GPU, model downloads or external API requests**.

To generate new data or run the analytic integrator tests, follow
[setup and commands](docs/SETUP.md). The generation path uses the recorded
upstream commits and verifies model hashes. Folding new sequences requires
the explicit `--allow-sequence-export` option.

## Repository map

| Location | Contents |
|---|---|
| `scripts/project.py` | Main command-line entry point |
| `scripts/` | Integrators, model adapters, sequence design, folding and scoring |
| `configs/` | Experiment protocol and pinned model sources |
| `tests/` | Alignment, paired-input and folding-transport checks |
| `results/pilot/` | Raw backbones, sequences, predictions, states and measured scores |
| `results/history/` | Initial FrameDiff/FrameFlow runs and pipeline verification |
| `docs/` | Setup, detailed methods, project history and publication instructions |
| `.github/workflows/` | CPU analysis and analytic tests configured for GitHub Actions |

Downloads and new runs go to ignored `external/` and `runs/` folders.
Model weights are downloaded separately; small saved frame tensors are included
to preserve the exact paired inputs.

## Contribution and sources

This project implements a standard Lie midpoint method, connects it to FrameFlow,
and builds a controlled evaluation pipeline. It uses published pretrained models;
no model training was performed. Implementation was developed with AI coding assistance.

- [FrameFlow code](https://github.com/microsoft/protein-frame-flow),
  [backbone-generation paper](https://arxiv.org/abs/2310.05297) and
  [motif-scaffolding paper](https://arxiv.org/abs/2401.04082).
- [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion): initial generation
  check and the bundled ProteinMPNN implementation used in the recorded runs.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN): backbone-to-sequence design.
- [ESM / ESMFold](https://github.com/facebookresearch/esm): structure prediction.

Project code is MIT licensed. Attribution and upstream license notices are in
[THIRD_PARTY.md](THIRD_PARTY.md) and `licenses/`.
