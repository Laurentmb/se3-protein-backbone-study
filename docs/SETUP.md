# Setup and commands

Run commands from the repository root. Use Python 3.10 for a new installation.
The original GPU experiments and the local packaging checks used Python 3.8.8;
GitHub Actions is configured to test Python 3.10 after publication.

## 1. CPU analysis: the shortest path

Create an isolated environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

If PowerShell activation is unavailable, use `.venv\Scripts\python.exe` in place
of `python` in every command. No activation is required in that case.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/project.py verify
python scripts/project.py test-metrics
python scripts/project.py analyze
```

Analysis uses the recorded ESMFold responses. Outputs are separated from the
bundled evidence. Repeating the command refreshes the same analysis folder.
Use `analyze --output runs/another-analysis` for a different destination.

## 2. Analytic integrator tests on CPU

In the analysis environment, add the recorded PyTorch version:

```bash
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python scripts/project.py test-integrators
```

The tests use an independent analytic trajectory and float64 arithmetic.
They check Euler's first-order and midpoint's second-order convergence,
rotation orthogonality, determinant +1, constant/zero fields, batching and
equivariance under a global rigid motion. Results go to
`runs/tests/integrator_tests.json`. The recorded test results are in
`results/history/verification/integrator_tests.json`.

## 3. Model inference on an NVIDIA GPU

Use a separate virtual environment, for example `.venv-gpu`, with Python 3.10.
The recorded hardware was a GTX 1650 Ti with 4 GB VRAM. Different GPUs can
produce slightly different outputs and timings.

Install the generation dependencies and matching CUDA binaries:

```bash
python -m pip install -r requirements-generation.txt
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
python scripts/project.py prepare
```

See [PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/)
for the CPU/CUDA wheel indexes. `prepare` needs Git and internet access. It fetches
the exact commits in `configs/sources.json`, downloads the published FrameFlow
weights archive (about 0.6 GB), and checks checkpoint/config hashes.
The pinned FrameDiff source supplies its checkpoint and the ProteinMPNN weights.
Upstream checkouts, checkpoints and the download archive remain in `external/`.

The wrappers import generation-only modules directly; training frameworks and a
local ESMFold installation are unnecessary for this workflow.

## 4. Run a new paired experiment

```bash
python scripts/project.py generate --run runs/new-pilot
python scripts/project.py design --run runs/new-pilot
python scripts/project.py fold --run runs/new-pilot --allow-sequence-export
python scripts/project.py analyze --input runs/new-pilot --output runs/new-pilot-analysis
```

- **Generate:** uses `configs/pilot.json` and reuses the included five initial
  states. Counts actual network calls, alternates execution order and warms up
  both methods. Generation requires an empty run folder.
- **Design:** uses equal ProteinMPNN temperature, seed and sequence count for
  every backbone. Original polyalanine entries are excluded.
- **Fold:** sends only this run's amino-acid sequences to the public ESM Atlas
  endpoint. The explicit flag enables that export. Saved successful responses
  are reused after verifying their sequence and PDB hashes; failures are recorded.
- **Analyze:** checks every request record, validates structure/sequence matches,
  calculates scores and generates tables, figures and a short report.

For an existing installation, `generate` and `numerical` accept `--repo PATH`.
`design` accepts `--mpnn-repo PATH` and `--weights PATH`; paths are explicit and
do not depend on the original development machine.

For a different protocol, copy and edit `configs/pilot.json`, then pass
`generate --protocol PATH`. Keep paired states, checkpoint, sequence count and
folding backend consistent. Supply `--initial-states PATH` to choose another
state folder; missing seed files are generated and saved. Remove the numerical
study's seeds from its list only when intentionally changing that experiment.
When changing protein length, use a new initial-state folder: saved tensors must
match the length specified in the protocol.

## 5. Optional numerical-reference study

```bash
python scripts/project.py numerical --run runs/new-pilot
python scripts/project.py analyze --input runs/new-pilot --output runs/new-pilot-analysis
```

This disables self-conditioning and integrates to t=0.99 before the final clean
prediction. It can require thousands of network evaluations per case. It refines
the reference and checks agreement with a second solver; it is not needed to
recompute the stored quality scores. An unconfirmed reference stays unconfirmed.

## Initial FrameDiff/FrameFlow check

These exploratory samples are preserved in `results/history/first-generation/`.
They used different evaluation budgets and are not the controlled solver benchmark.
To generate a single upstream sample after `prepare`:

```bash
python scripts/generate_backbone.py --model framediff --repo external/framediff --output runs/framediff-example --length 100 --seed 123
python scripts/generate_backbone.py --model frameflow --repo external/frameflow --output runs/frameflow-example --length 100 --seed 123
```

Run them in separate processes because both upstream projects expose packages
named `data`, `analysis` and `openfold`. The numerical benchmark changes only the
sampler adapter and documents its terminal-time and self-conditioning conventions.
