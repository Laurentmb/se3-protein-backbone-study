"""Smoke-test a Lie midpoint sampler with the original FrameFlow checkpoint.

Scope: unconditional generation only, with every residue free. No motif guidance.
Both methods use exactly the same saved initial frames and final clean prediction.
The final network call is counted in the evaluation budget. Self-conditioning is
frozen during a midpoint step and updated from that step's midpoint prediction.
Disable self-conditioning to study a genuinely memoryless neural ODE.
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

from lie_integrators import euler_step, midpoint_step


def sha256(path):
    """Identify the exact checkpoint and initial state used by this run."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class FrameFlowField:
    """Convert clean-frame predictions to the authors' translation/rotation field."""
    def __init__(self, model, interpolant, length, device):
        from data import so3_utils
        self.log = so3_utils.calc_rot_vf
        self.model = model
        self.interpolant = interpolant
        self.calls = 0
        self.condition = None
        self.last_prediction = None
        self.batch = {'res_mask': torch.ones(1, length, device=device),
            'diffuse_mask': torch.ones(1, length, device=device),
            'res_idx': torch.arange(length, device=device, dtype=torch.float32)[None]}

    def predict(self, t, x, rotation):
        """One network evaluation, including the final clean projection at t=1."""
        batch = dict(self.batch)
        batch.update(trans_t=x, rotmats_t=rotation)
        time_tensor = x.new_full((x.shape[0], 1), float(t))
        batch.update(t=time_tensor, so3_t=time_tensor, r3_t=time_tensor)
        if self.condition is not None:
            batch['trans_sc'] = self.condition
        prediction = self.model(batch)
        self.calls += 1
        self.last_prediction = prediction
        return prediction

    def __call__(self, t, x, rotation):
        """Do not evaluate the singular translation field at t=1."""
        if t >= 1:
            raise ValueError('Translation vector field is singular at t=1')
        prediction = self.predict(t, x, rotation)
        velocity = (prediction['pred_trans']-x)/(1-float(t))
        schedule = self.interpolant._rots_cfg.sample_schedule
        scale = (self.interpolant._rots_cfg.exp_rate if schedule == 'exp'
            else 1/(1-float(t)))
        omega = scale*self.log(rotation, prediction['pred_rotmats'])
        return velocity, omega


def sample(field, state, method, intervals, self_condition, terminal_mode='endpoint'):
    """Apply fixed steps followed by the same clean projection as upstream."""
    x, rotation = state['trans_0'].clone(), state['rotmats_0'].clone()
    times = torch.linspace(field.interpolant._cfg.min_t, 1.0, intervals+1,
        device=x.device)
    max_orthogonality = 0.0
    step_fn = euler_step if method == 'euler' else midpoint_step
    for left, right in zip(times[:-1], times[1:]):
        # Leave the history fixed during both evaluations of a midpoint step.
        if method == 'euler':
            # Reuse upstream arithmetic, including its Rodrigues exponential.
            clean = field.predict(left, x, rotation)
            x, rotation = (
                field.interpolant._trans_euler_step(right-left, left,
                    clean['pred_trans'], x),
                field.interpolant._rots_euler_step(right-left, left,
                    clean['pred_rotmats'], rotation))
        else:
            x, rotation = step_fn(field, float(left), x, rotation, float(right-left))
        if self_condition:
            field.condition = field.last_prediction['pred_trans'].detach()
        identity = torch.eye(3, device=x.device)
        max_orthogonality = max(max_orthogonality, float(torch.max(torch.abs(
            rotation.transpose(-1, -2) @ rotation-identity))))
        assert torch.isfinite(x).all() and torch.isfinite(rotation).all()
    # The terminal output is a clean prediction, not the raw integrated state.
    # Upstream leaves so3_t/r3_t at the previous grid point in its final call.
    # Endpoint mode explicitly uses t=1 for both solvers, independent of steps.
    final = field.predict(float(times[-2]) if terminal_mode == 'upstream' else 1.0,
        x, rotation)
    return final, max_orthogonality


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--method', choices=['euler', 'midpoint'], required=True)
    parser.add_argument('--intervals', type=int, default=10)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-self-conditioning', action='store_true')
    parser.add_argument('--verify-upstream', action='store_true')
    parser.add_argument('--terminal-mode', choices=['endpoint', 'upstream'],
        default='endpoint')
    args = parser.parse_args()
    if args.intervals < 1:
        raise ValueError('intervals must be positive')
    root = Path(__file__).resolve().parents[3]
    repo, output, initial_path = [p.resolve() for p in
        (args.repo, args.output, args.initial_state)]
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'outputs/se3-pilot/scripts'))
    import generate_backbone as upstream_runtime
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    upstream_runtime.DEVICE = args.device
    checkpoint = repo / 'weights/pdb/published.ckpt'
    model, cfg, _, _, exporter = upstream_runtime.frameflow(
        repo, args.length, args.intervals+1, checkpoint)
    from data.interpolant import Interpolant, _centered_gaussian, _uniform_so3
    from data import all_atom
    interpolant = Interpolant(cfg.inference.interpolant)
    interpolant.set_device(args.device)
    self_condition = not args.no_self_conditioning
    interpolant._cfg.self_condition = self_condition

    # Persist the actual tensors: matching seeds alone is weaker evidence.
    if initial_path.exists():
        state = torch.load(str(initial_path), map_location=args.device)
    else:
        state = {'trans_0': _centered_gaussian(1, args.length, args.device)*10,
            'rotmats_0': _uniform_so3(1, args.length, args.device)}
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k:v.cpu() for k,v in state.items()}, str(initial_path))
    assert state['trans_0'].shape == (1, args.length, 3)
    assert state['rotmats_0'].shape == (1, args.length, 3, 3)
    field = FrameFlowField(model, interpolant, args.length, args.device)
    torch.cuda.synchronize() if args.device.startswith('cuda') else None
    start = time.perf_counter()
    with torch.no_grad():
        prediction, orthogonality = sample(field, state, args.method,
            args.intervals, self_condition, args.terminal_mode)
    torch.cuda.synchronize() if args.device.startswith('cuda') else None
    seconds = time.perf_counter()-start
    atoms = all_atom.transrot_to_atom37([(prediction['pred_trans'],
        prediction['pred_rotmats'])], field.batch['res_mask'])[-1][0].numpy()
    expected_calls = args.intervals*(1 if args.method == 'euler' else 2)+1
    assert field.calls == expected_calls
    assert np.isfinite(atoms).all() and orthogonality < 1e-3
    pdb_path = output / ('frameflow_{}_L{}_seed{}.pdb'.format(
        args.method, args.length, args.seed))
    exporter.write_prot_to_pdb(atoms, str(pdb_path), no_indexing=True)
    metadata = {'method': args.method, 'length': args.length, 'seed': args.seed,
        'intervals': args.intervals, 'network_evaluations': field.calls,
        'sampling_seconds': seconds, 'self_conditioning': self_condition,
        'midpoint_history_policy': 'frozen within step; update from midpoint prediction',
        'terminal_projection': 'one clean prediction, counted in budget',
        'terminal_time_mode': args.terminal_mode,
        'max_integrated_orthogonality_error': orthogonality,
        'checkpoint_sha256': sha256(checkpoint),
        'initial_state_sha256': sha256(initial_path), 'pdb_sha256': sha256(pdb_path)}
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2),
        encoding='utf-8')

    # Compare the Euler adapter with the original sampler on identical tensors.
    if args.verify_upstream:
        if args.method != 'euler':
            raise ValueError('Upstream equivalence check is for Euler only')
        if args.terminal_mode != 'upstream':
            raise ValueError('Select --terminal-mode upstream for strict equivalence')
        calls = [0]
        def count_calls(module, inputs, result):
            calls[0] += 1
        hook = model.register_forward_hook(count_calls)
        with torch.no_grad():
            trajectory, _, _ = interpolant.sample(1, args.length, model,
                num_timesteps=args.intervals+1, trans_0=state['trans_0'].clone(),
                rotmats_0=state['rotmats_0'].clone(),
                diffuse_mask=field.batch['diffuse_mask'])
        hook.remove()
        original = trajectory[-1][0].numpy()
        ca_error = np.linalg.norm(atoms[:,1]-original[:,1], axis=-1)
        metadata['upstream_equivalence'] = {'network_evaluations': calls[0],
            'unaligned_CA_RMSD_A': float(np.sqrt(np.mean(ca_error**2))),
            'max_CA_difference_A': float(ca_error.max())}
        exporter.write_prot_to_pdb(original, str(output/'upstream_euler.pdb'),
            no_indexing=True)
        (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2),
            encoding='utf-8')
        assert calls[0] == expected_calls
        assert metadata['upstream_equivalence']['unaligned_CA_RMSD_A'] < 0.02
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2),
        encoding='utf-8')
    OmegaConf.save(cfg, str(output/'resolved_config.yaml'))
    print(json.dumps(metadata, indent=2))
