"""Estimate neural-ODE discretization error against an empirically refined reference.

Disable self-conditioning: otherwise changing the steps also changes the history
seen by the network. Stop at 0.99 to avoid the translation field's 1/(1-t)
singularity. These errors are for this truncated memoryless ODE, not for the
production sampler or physical protein folding. No exact neural solution exists.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from lie_integrators import integrate
from sample_frameflow import FrameFlowField, sha256


def difference(state, reference):
    """Measure translation RMS distance and rotation RMS angle without alignment."""
    x, r = [v.detach().cpu().double().numpy() for v in state]
    tx, tr = [v.detach().cpu().double().numpy() for v in reference]
    relative = np.swapaxes(r, -1, -2) @ tr
    # SciPy's rotation conversion tolerates small float32 orthogonality drift.
    angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return {'translation_rms_A': float(np.sqrt(np.mean(np.sum((x-tx)**2,axis=-1)))),
        'rotation_rms_rad': float(np.sqrt(np.mean(angles**2))),
        'rotation_max_rad': float(angles.max())}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo, output=args.repo.resolve(),args.output.resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    study=protocol['numerical_study']
    folder=output/'numerical'
    folder.mkdir(exist_ok=True)
    sys.path.insert(0,str(repo))
    os.chdir(str(repo))
    import generate_backbone as loader
    from data.interpolant import Interpolant
    torch.set_num_threads(4)
    loader.DEVICE=args.device
    model,cfg,_,_,_=loader.frameflow(repo,protocol['length'],100,
        repo/'weights/pdb/published.ckpt')
    interpolant=Interpolant(cfg.inference.interpolant)
    interpolant.set_device(args.device)
    interpolant._cfg.self_condition=False
    report={'self_conditioning':False,'start':study['start'],'stop':study['stop'],
        'precision':'float32','error_definition':'pre-projection, no rigid alignment',
        'reference_is_exact':False,'seeds':[]}

    def run(initial, method, intervals, label):
        """A fresh field carries no history; count every network evaluation."""
        field=FrameFlowField(model,interpolant,protocol['length'],args.device)
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        start=time.perf_counter()
        with torch.no_grad():
            x,r,calls=integrate(field,initial['trans_0'].clone(),
                initial['rotmats_0'].clone(),study['start'],study['stop'],intervals,method)
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        assert field.calls==calls
        assert torch.isfinite(x).all() and torch.isfinite(r).all()
        torch.save({'translation':x.cpu(),'rotation':r.cpu()},str(folder/(label+'.pt')))
        print('{}: {} calls, {:.1f} s'.format(label,calls,time.perf_counter()-start),flush=True)
        return (x,r),calls

    for seed in study['seeds']:
        initial_path=output/'initial_states'/('seed{}.pt'.format(seed))
        initial=torch.load(str(initial_path),map_location=args.device)
        item={'seed':seed,'initial_state_sha256':sha256(initial_path),
            'low_budget':[],'reference_refinements':[]}
        low=[]
        for budget in protocol['network_budgets_including_final_prediction']:
            for method in protocol['methods']:
                intervals=(budget-1)//(1 if method=='euler' else 2)
                state,calls=run(initial,method,intervals,
                    'seed{}_{}_budget{}'.format(seed,method,budget))
                low.append((method,budget,intervals,calls,state))

        # Refine midpoint until two grids agree, then check an Euler reference.
        previous=None
        reference=None
        confirmed=False
        n=study['initial_reference_intervals'][0]
        while n<=study['maximum_reference_intervals']:
            reference,ref_calls=run(initial,'midpoint',n,
                'seed{}_midpoint_reference{}'.format(seed,n))
            refinement={'midpoint_intervals':n,'network_calls':ref_calls}
            if previous is not None:
                delta=difference(reference,previous)
                refinement['difference_from_previous']=delta
                grid_stable=(delta['translation_rms_A']<study['translation_stability_threshold_A']
                    and delta['rotation_rms_rad']<study['rotation_stability_threshold_rad'])
                if grid_stable:
                    cross,cross_calls=run(initial,'euler',2*n,
                        'seed{}_euler_cross_reference{}'.format(seed,2*n))
                    cross_delta=difference(cross,reference)
                    refinement['cross_solver_difference']=cross_delta
                    refinement['cross_solver_network_calls']=cross_calls
                    confirmed=(cross_delta['translation_rms_A']<2*study['translation_stability_threshold_A']
                        and cross_delta['rotation_rms_rad']<2*study['rotation_stability_threshold_rad'])
            item['reference_refinements'].append(refinement)
            if confirmed:
                break
            previous=reference
            n*=2
        item.update(reference_intervals=n//2 if n>study['maximum_reference_intervals'] else n,
            reference_confirmed_within_tolerances=confirmed)
        for method,budget,intervals,calls,state in low:
            item['low_budget'].append({'method':method,'matched_production_budget':budget,
                'intervals':intervals,'network_calls_without_final_projection':calls,
                **difference(state,reference)})
        report['seeds'].append(item)
        (output/'numerical_reference.json').write_text(json.dumps(report,indent=2))
        print('Seed {} reference confirmed: {}'.format(seed,confirmed),flush=True)
