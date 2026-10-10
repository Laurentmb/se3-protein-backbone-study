"""Render measured numerical convergence and the two refolded CA overlays."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_folds import read_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    numerical = json.loads((root/'integrator_tests.json').read_text())

    # Plot error against actual field evaluations, rather than step count.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, color in [('euler', '#536b82'), ('midpoint', '#098578')]:
        rows = numerical['convergence'][method]
        for ax, key, title in zip(axes, ['translation_error', 'rotation_error_rad'],
                ['Translation error', 'Rotation error (rad)']):
            ax.loglog([r['field_evaluations'] for r in rows],
                [r[key] for r in rows], 'o-', color=color, label=method.title())
            ax.set(title=title, xlabel='Vector-field evaluations', ylabel='Error')
            ax.grid(True, which='both', alpha=.2)
            ax.legend()
    fig.suptitle('Analytic test: time- and state-dependent, noncommuting rotations')
    fig.savefig(str(root/'integrator_convergence.png'), dpi=180)
    plt.close(fig)

    # Both predicted folds were fitted with Kabsch using all 100 CA atoms.
    report = json.loads((root/'pipeline/evaluation/evaluation.json').read_text())
    _, backbone, _, _ = read_structure(root/'pipeline/backbone.pdb')
    center = backbone.mean(0)
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    for i, row in enumerate(report['candidates']):
        _, predicted, _, _ = read_structure(root/'pipeline/evaluation'/
            (row['id']+'_aligned.pdb'))
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        b, p = backbone-center, predicted-center
        ax.plot(*b.T, color='#536b82', linewidth=2.0, label='Generated backbone')
        ax.plot(*p.T, color='#e07a35', linewidth=1.7, label='ESMFold prediction')
        span = np.abs(np.concatenate([b, p])).max()*1.1
        ax.set(xlim=(-span, span), ylim=(-span, span), zlim=(-span, span))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=35)
        ax.set_axis_off()
        ax.set_title('Sequence {} | CA RMSD {:.2f} A\nTM-score {:.3f}'.format(
            i+1, row['ca_rmsd_A'], row['tm_score_normalized_backbone']))
        ax.legend(loc='lower center', fontsize=8)
    fig.suptitle('Pipeline verification: two sequences designed for one FrameFlow backbone')
    fig.savefig(str(root/'refolding_overlay.png'), dpi=180)
    plt.close(fig)
    print('Saved numerical convergence and refolding figures')
