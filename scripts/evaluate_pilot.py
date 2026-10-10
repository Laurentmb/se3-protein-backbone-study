"""Score every pilot backbone, preserve failed folds, and summarize paired results.

Quality differences are paired by initial state and network budget. Positive
midpoint-minus-Euler RMSD means midpoint was worse for that pair. This small
pilot is descriptive; it does not support a general claim of solver superiority.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from Bio.PDB import PDBParser
import numpy as np


def save_csv(path, rows):
    """Use the union of fields so failure details are retained in flat exports."""
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    # An unsubmitted batch is pending work, not forty failed predictions.
    folding_path=root/'folding.json'
    if not folding_path.exists():
        raise SystemExit('No folding request records exist. Run the authorized folding batch first.')
    folding=json.loads(folding_path.read_text())
    candidates=json.loads((root/'candidates.json').read_text())
    if {r['id'] for r in folding}!={r['id'] for r in candidates}:
        raise SystemExit('The folding batch is incomplete. Resume it before evaluating the pilot.')
    runs=json.loads((root/'generation.json').read_text())
    rows=[]
    sequences=[]
    for run in runs:
        folder=root/'evaluation'/run['id']
        command=[sys.executable,'-B',str(Path(__file__).parent/'evaluate_folds.py'),
            '--backbone',str(root/run['pdb']),
            '--fasta',str(root/'design'/run['id']/'candidates.fasta'),
            '--folds',str(root/'folds'),'--output',str(folder)]
        process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (folder/'evaluation.log').write_text(process.stdout,encoding='utf-8')
        # Individual missing predictions remain visible instead of aborting all targets.
        report=json.loads((folder/'evaluation.json').read_text())
        row={**run,'complete_evaluation':report['pipeline_verified'],
            'successful_predictions':report['successful_predictions'],
            'scRMSD_A':report.get('scRMSD_A'), 'scTM':report.get('scTM'),
            'passes_scRMSD_under_2A':report.get('passes_scRMSD_under_2A',False)}
        structure=PDBParser(QUIET=True).get_structure(run['id'],str(root/run['pdb']))
        chains=list(structure.get_chains())
        residues=list(chains[0].get_residues())
        assert len(chains)==1 and len(residues)==run['length']
        assert all(all(a in r for a in ('N','CA','C','O')) for r in residues)
        xyz=np.array([[r[a].coord for a in ('N','CA','C','O')] for r in residues],dtype=float)
        assert np.isfinite(xyz).all()
        ca=xyz[:,1]
        distances=np.linalg.norm(ca[:,None]-ca[None,:],axis=-1)
        row.update(ca_breaks_over_4_5_A=int(np.count_nonzero(np.linalg.norm(np.diff(ca,axis=0),axis=-1)>4.5)),
            ca_pairs_under_1_5_A=int(np.count_nonzero(distances[np.triu_indices(len(ca),k=1)]<1.5)),
            peptide_c_n_pairs_over_2_A=int(np.count_nonzero(np.linalg.norm(xyz[:-1,2]-xyz[1:,0],axis=-1)>2)))
        rows.append(row)
        sequences.extend([{**r,'backbone_id':run['id'],'seed':run['seed'],
            'budget':run['budget'],'method':run['method']} for r in report['candidates']])

    # Match by seed AND budget; never compare unrelated random generations.
    lookup={(r['seed'],r['budget'],r['method']):r for r in rows}
    paired=[]
    for seed,budget in sorted({(r['seed'],r['budget']) for r in rows}):
        e,m=lookup[(seed,budget,'euler')],lookup[(seed,budget,'midpoint')]
        pair={'seed':seed,'budget':budget,'complete':e['complete_evaluation'] and m['complete_evaluation'],
            'euler_scRMSD_A':e['scRMSD_A'],'midpoint_scRMSD_A':m['scRMSD_A'],
            'euler_success':e['passes_scRMSD_under_2A'],
            'midpoint_success':m['passes_scRMSD_under_2A'],
            'euler_seconds':e['sampling_seconds'],'midpoint_seconds':m['sampling_seconds']}
        if pair['complete']:
            pair['midpoint_minus_euler_scRMSD_A']=m['scRMSD_A']-e['scRMSD_A']
        paired.append(pair)

    groups=[]
    for budget in sorted({r['budget'] for r in rows}):
        for method in ('euler','midpoint'):
            all_rows=[r for r in rows if r['budget']==budget and r['method']==method]
            complete=[r for r in all_rows if r['complete_evaluation']]
            groups.append({'budget':budget,'method':method,'attempted_backbones':len(all_rows),
                'complete_backbones':len(complete),
                'designable_count':sum(r['passes_scRMSD_under_2A'] for r in complete),
                'median_scRMSD_A':float(np.median([r['scRMSD_A'] for r in complete])) if complete else None,
                'mean_scRMSD_A':float(np.mean([r['scRMSD_A'] for r in complete])) if complete else None,
                'median_scTM':float(np.median([r['scTM'] for r in complete])) if complete else None,
                'median_sampling_seconds':float(np.median([r['sampling_seconds'] for r in all_rows]))})
    summary={'attempted_backbones':len(rows),'attempted_sequences':len(sequences),
        'complete_backbones':sum(r['complete_evaluation'] for r in rows),
        'successful_predictions':sum(r['success'] for r in sequences),
        'groups':groups,'paired':paired}
    for name,data in [('backbone_results',rows),('sequence_results',sequences),('paired_results',paired),('group_results',groups)]:
        (root/(name+'.json')).write_text(json.dumps(data,indent=2))
        save_csv(root/(name+'.csv'),data)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
