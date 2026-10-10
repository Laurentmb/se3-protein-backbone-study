"""Generate a paired pilot with one checkpoint and exact shared initial tensors.

Run GPU jobs sequentially so solver timings are not contaminated by each other.
Order alternates by seed/budget; both methods receive one unmeasured warmup.
Self-conditioning and terminal-time conventions are fixed in protocol.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from sample_frameflow import FrameFlowField, sample, sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    device = args.device
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    import generate_backbone as loader
    from data.interpolant import Interpolant, _centered_gaussian, _uniform_so3
    from data import all_atom
    torch.set_num_threads(4)
    loader.DEVICE = device
    checkpoint = repo/'weights/pdb/published.ckpt'
    model, cfg, _, _, exporter = loader.frameflow(repo, protocol['length'], 100, checkpoint)
    interpolant = Interpolant(cfg.inference.interpolant)
    interpolant.set_device(device)
    interpolant._cfg.self_condition = protocol['self_conditioning']
    checkpoint_hash = sha256(checkpoint)
    output.joinpath('initial_states').mkdir(exist_ok=True)
    output.joinpath('backbones').mkdir(exist_ok=True)
    runs = []
    expected_runs = len(protocol['seeds'])*len(protocol['network_budgets_including_final_prediction'])*len(protocol['methods'])

    # Seed 123 reuses the tensors from the verified sampler; new seeds are saved.
    states = {}
    for seed in protocol['seeds']:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        path = output/'initial_states'/('seed{}.pt'.format(seed))
        verified = Path(__file__).resolve().parents[1]/'results/history/verification/sampler/initial_state.pt'
        if path.exists():
            state = torch.load(str(path), map_location=device)
        elif seed == 123 and verified.exists():
            state = torch.load(str(verified), map_location=device)
            torch.save({k:v.cpu() for k,v in state.items()}, str(path))
        else:
            state = {'trans_0': _centered_gaussian(1, protocol['length'], device)*10,
                'rotmats_0': _uniform_so3(1, protocol['length'], device)}
            torch.save({k:v.cpu() for k,v in state.items()}, str(path))
        if state['trans_0'].shape != (1, protocol['length'], 3) or state['rotmats_0'].shape != (1, protocol['length'], 3, 3):
            raise ValueError('Initial-state length does not match protocol: '+str(path))
        states[seed] = state

    # Trigger model/CUDA initialization before measuring either solver.
    with torch.no_grad():
        for method in protocol['methods']:
            field = FrameFlowField(model, interpolant, protocol['length'], device)
            sample(field, states[protocol['seeds'][0]], method, 2,
                protocol['self_conditioning'], protocol['terminal_time_mode'])
    torch.cuda.synchronize() if device.startswith('cuda') else None

    # The terminal clean prediction consumes one call in both evaluation budgets.
    for seed_index, seed in enumerate(protocol['seeds']):
        for budget_index, budget in enumerate(protocol['network_budgets_including_final_prediction']):
            order = protocol['methods'][::(-1 if (seed_index+budget_index)%2 else 1)]
            for method in order:
                identifier = 'seed{}_nfe{}_{}'.format(seed, budget, method)
                factor = 1 if method == 'euler' else 2
                intervals = (budget-1)//factor
                assert intervals*factor+1 == budget
                field = FrameFlowField(model, interpolant, protocol['length'], device)
                counted = [0]
                def count(module, inputs, result):
                    counted[0] += 1
                hook = model.register_forward_hook(count)
                torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
                torch.cuda.synchronize() if device.startswith('cuda') else None
                start = time.perf_counter()
                with torch.no_grad():
                    prediction, orthogonality = sample(field, states[seed], method,
                        intervals, protocol['self_conditioning'],
                        protocol['terminal_time_mode'])
                torch.cuda.synchronize() if device.startswith('cuda') else None
                seconds = time.perf_counter()-start
                hook.remove()
                assert field.calls == counted[0] == budget
                assert orthogonality < 1e-3
                atoms = all_atom.transrot_to_atom37([(prediction['pred_trans'],
                    prediction['pred_rotmats'])], field.batch['res_mask'])[-1][0].numpy()
                assert np.isfinite(atoms).all()
                pdb = output/'backbones'/(identifier+'.pdb')
                exporter.write_prot_to_pdb(atoms, str(pdb), no_indexing=True)
                row = {'id': identifier, 'seed': seed, 'budget': budget, 'method': method,
                    'intervals': intervals, 'network_calls': counted[0],
                    'sampling_seconds': seconds, 'length': protocol['length'],
                    'self_conditioning': protocol['self_conditioning'],
                    'terminal_time_mode': protocol['terminal_time_mode'],
                    'max_orthogonality_error': orthogonality,
                    'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
                    'checkpoint_sha256': checkpoint_hash,
                    'initial_state_sha256': sha256(output/'initial_states'/('seed{}.pt'.format(seed))),
                    'pdb_sha256': sha256(pdb), 'pdb': pdb.relative_to(output).as_posix()}
                runs.append(row)
                (output/'generation.json').write_text(json.dumps(runs, indent=2))
                print('Generated {} ({}/{}), {:.2f} s'.format(identifier,len(runs),expected_runs,seconds),flush=True)
    OmegaConf.save(cfg, str(output/'resolved_config.yaml'))
    assert len(runs) == expected_runs
    (output/'generation_environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'numpy':np.__version__,
        'gpu':torch.cuda.get_device_name() if device.startswith('cuda') else None,
        'repo_commit': __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip(),
        'protocol_sha256':sha256(output/'protocol.json')},indent=2))
