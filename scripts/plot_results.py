"""Plot all paired quality results and the separate neural-ODE error study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    summary=(json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists()
        else {'paired':[]})
    budgets=sorted({p['budget'] for p in summary['paired']})
    colors=plt.get_cmap('tab10')
    # A pending external step must not be visualized as zero quality scores.
    fig,axes=plt.subplots(1,max(1,len(budgets)),figsize=(10,4.6),sharey=True,constrained_layout=True)
    for ax,budget in zip(np.atleast_1d(axes),budgets):
        for index,pair in enumerate(p for p in summary['paired'] if p['budget']==budget):
            if not pair['complete']:
                continue
            ax.plot([0,1],[pair['euler_scRMSD_A'],pair['midpoint_scRMSD_A']],
                'o-',color=colors(index),label='Seed {}'.format(pair['seed']),alpha=.85)
        ax.axhline(2,color='#737373',ls='--',lw=1,label='2 A threshold')
        ax.set(xticks=[0,1],xticklabels=['Euler','Lie midpoint'],
            xlim=(-.15,1.15),title='{} total network evaluations'.format(budget),
            ylabel='Best-of-two CA RMSD (A)',ylim=(0,None))
        ax.grid(axis='y',alpha=.2)
    if budgets:
        np.atleast_1d(axes)[-1].legend(fontsize=8,loc='upper right')
        fig.suptitle('Paired FrameFlow pilot | 100 residues | self-conditioning enabled')
        fig.savefig(str(root/'paired_quality.png'),dpi=180)
    plt.close(fig)

    numerical_path=root/'numerical_reference.json'
    if numerical_path.exists():
        numerical=json.loads(numerical_path.read_text())
        fig,axes=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
        for seed_index,item in enumerate(numerical['seeds']):
            for method,color in [('euler','#536b82'),('midpoint','#098578')]:
                values=[r for r in item['low_budget'] if r['method']==method]
                values.sort(key=lambda r:r['network_calls_without_final_projection'])
                for ax,key,title in zip(axes,['translation_rms_A','rotation_rms_rad'],
                        ['Translation deviation (A)','Rotation deviation (rad)']):
                    ax.loglog([v['network_calls_without_final_projection'] for v in values],
                        [v[key] for v in values],marker='o' if seed_index==0 else 's',
                        ls='-' if seed_index==0 else '--',color=color,
                        label='{} | seed {}{}'.format(method.title(),item['seed'],
                            '' if item['reference_confirmed_within_tolerances'] else ' (provisional)'))
                    ax.set(title=title,xlabel='Integration network evaluations',ylabel='Deviation')
                    ax.grid(which='both',alpha=.2)
                    ax.legend(fontsize=8)
        confirmed=(len(numerical['seeds'])==2 and all(
            item['reference_confirmed_within_tolerances'] for item in numerical['seeds']))
        fig.suptitle('Neural ODE: t=0.01 to 0.99, no self-conditioning\n'
            +('Refined references agree within predefined tolerances' if confirmed
                else 'Unconfirmed references are labelled provisional'))
        fig.savefig(str(root/'numerical_deviation.png'),dpi=180)
        plt.close(fig)

        # Show reference checks separately from low-budget solver deviations.
        # A small gap on one refinement does not establish convergence by itself.
        items=numerical['seeds']
        fig,axes=plt.subplots(2,len(items),figsize=(5*len(items),7),
            squeeze=False,constrained_layout=True)
        for column,item in enumerate(items):
            for row,(key,unit,tolerance) in enumerate([
                    ('translation_rms_A','A',.05),('rotation_rms_rad','rad',.005)]):
                ax=axes[row,column]
                for check,label,color,marker in [
                        ('difference_from_previous','Midpoint: N versus N/2','#098578','o'),
                        ('cross_solver_difference','Euler 2N versus midpoint N','#536b82','x')]:
                    points=[r for r in item['reference_refinements'] if check in r]
                    ax.loglog([r['midpoint_intervals'] for r in points],
                        [r[check][key] for r in points],marker=marker,color=color,
                        label=label)
                ax.axhline(tolerance,color='#098578',ls='--',lw=1,
                    label='Grid tolerance')
                ax.axhline(2*tolerance,color='#536b82',ls=':',lw=1,
                    label='Cross-solver tolerance')
                ax.set(xlabel='Midpoint reference intervals N',ylabel='Deviation ('+unit+')',
                    title='Seed {} | {}'.format(item['seed'],
                        'accepted' if item['reference_confirmed_within_tolerances'] else 'unconfirmed'))
                ax.grid(which='both',alpha=.2)
                ax.legend(fontsize=7)
        fig.suptitle('Reference refinement checks | no self-conditioning')
        fig.savefig(str(root/'reference_refinement.png'),dpi=180)
        plt.close(fig)
    print('Saved comparison figures')
