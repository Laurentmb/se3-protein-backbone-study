# Local repository checks

Checked on 7 October 2026, on Windows with Python 3.8.8.

| Check | Observed result |
|---|---|
| Fresh analysis environment | Created without system packages; PyTorch absent |
| Offline analysis | All 20 backbones and 40 predictions scored successfully |
| Recomputed scores | Every candidate's CA RMSD and TM-score matches the stored result within 1e-6 |
| Unit tests | Seven passed: alignment, reflection exclusion, deformation, paired inputs, API payload, cached response and HTTP failure |
| Analytic integration | All checks passed; observed orders approximately 1.004 for Euler and 2.006/2.010 for midpoint |
| Portable GPU generation | Both methods ran from the new entry point with exactly 21 network calls each |
| Portable sequence design | Two ProteinMPNN jobs produced four 100-residue candidates |
| Model assets | Pinned source commits, three checkpoint hashes and FrameFlow config hash verified against local assets |

The GPU checks reused the installed generation environment and verified local
model assets. The CPU analysis used a newly installed isolated environment.
The Python 3.10 GitHub Actions jobs are configured but will run only after
publication; they have not been observed on GitHub yet.

These checks validate the repository workflow and preserve the existing results.
The second neural-reference case remains unconfirmed, as documented in
[methods](METHODS.md).
