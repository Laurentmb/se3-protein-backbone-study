"""Generate one backbone with official weights and upstream sampling code.

Run each model in a separate process because their top-level packages overlap.
FrameDiff's three inference methods are loaded verbatim from its training class
using Python's AST, avoiding imports of training/evaluation-only dependencies.
FrameFlow uses FlowModel and Interpolant.sample directly, without a Trainer.
Neither model architecture, state dictionary nor numerical sampler is changed.
Index constants are explicitly cast to int64 for Windows NumPy compatibility.
"""
import argparse
import ast
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import tree
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def framediff(repo, length, steps, checkpoint):
    from analysis import utils as au
    from data import all_atom, se3_diffuser, utils as du
    from model.score_network import ScoreNetwork
    from openfold.utils import rigid_utils as ru

    all_atom.GROUP_IDX = all_atom.GROUP_IDX.long()

    with initialize_config_dir(version_base=None, config_dir=str(repo / 'config')):
        cfg = compose(config_name='inference', overrides=['hydra/launcher=basic'])
    OmegaConf.set_struct(cfg, False)
    weights = torch.load(str(checkpoint), map_location='cpu')
    cfg.model = OmegaConf.merge(cfg.model, weights['conf'].model)
    diffuser = se3_diffuser.SE3Diffuser(cfg.diffuser)
    model = ScoreNetwork(cfg.model, diffuser)
    state = {k.replace('module.', ''): v for k, v in weights['model'].items()}
    model.load_state_dict(state, strict=True)
    del weights, state
    model = model.to(DEVICE).eval()

    source_path = repo / 'experiments' / 'train_se3_diffusion.py'
    source = source_path.read_text(encoding='utf-8')
    upstream_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Experiment')
    names = {'_self_conditioning', '_set_t_feats', 'inference_fn'}
    methods = [n for n in upstream_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    context = {'torch':torch, 'np':np, 'copy':copy, 'du':du, 'ru':ru, 'all_atom':all_atom}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), 'exec'), context)
    Runtime = type('FrameDiffSamplingRuntime', (), {name:context[name] for name in names})
    runtime = Runtime()
    runtime.model = model
    runtime.diffuser = diffuser
    runtime._model_conf = cfg.model
    runtime._data_conf = cfg.data
    ref = diffuser.sample_ref(n_samples=length, as_tensor_7=True)
    feats = {
        'res_mask': np.ones(length), 'seq_idx': torch.arange(1, length + 1),
        'fixed_mask': np.zeros(length), 'torsion_angles_sin_cos': np.zeros((length, 7, 2)),
        'sc_ca_t': np.zeros((length, 3)), **ref,
    }
    feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), feats)
    feats = tree.map_structure(lambda x: x[None].to(DEVICE), feats)
    metadata = {
        'num_steps':steps, 'min_t':float(cfg.inference.diffusion.min_t),
        'noise_scale':float(cfg.inference.diffusion.noise_scale),
        'upstream_sampling_methods_sha256':hashlib.sha256('\n'.join(ast.get_source_segment(source,n) for n in methods).encode()).hexdigest(),
    }
    return model, cfg, metadata, lambda: runtime.inference_fn(
        feats, num_t=steps, min_t=cfg.inference.diffusion.min_t,
        noise_scale=cfg.inference.diffusion.noise_scale, aux_traj=False)['prot_traj'][0,0], au


def frameflow(repo, length, steps, checkpoint):
    from analysis import utils as au
    from data import all_atom
    from data.interpolant import Interpolant
    from models.flow_model import FlowModel
    all_atom.GROUP_IDX = all_atom.GROUP_IDX.long()
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
        cfg = compose(config_name='inference_unconditional')
    ckpt_cfg = OmegaConf.load(str(checkpoint.parent / 'config.yaml'))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(ckpt_cfg, False)
    cfg = OmegaConf.merge(cfg, ckpt_cfg)
    weights = torch.load(str(checkpoint), map_location='cpu')
    state = {k[len('model.'):]:v for k,v in weights['state_dict'].items() if k.startswith('model.')}
    assert state, 'Model weights missing'
    model = FlowModel(cfg.model)
    model.load_state_dict(state, strict=True)
    del weights, state
    model = model.to(DEVICE).eval()
    cfg.inference.interpolant.sampling.num_timesteps = steps
    interpolant = Interpolant(cfg.inference.interpolant)
    interpolant.set_device(DEVICE)
    def sample():
        traj, _, _ = interpolant.sample(1, length, model,
            diffuse_mask=torch.ones(1, length, device=DEVICE))
        return traj[-1][0].detach().cpu().numpy()
    return model, cfg, {'num_steps':steps, 'interpolant':OmegaConf.to_container(cfg.inference.interpolant, resolve=True)}, sample, au


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['framediff','frameflow'], required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    runner_path = Path(__file__).resolve()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(repo))
    os.chdir(str(repo))
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    DEVICE = args.device
    if DEVICE.startswith('cuda'):
        assert torch.cuda.is_available(), 'CUDA runtime unavailable'
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    steps = args.steps or (500 if args.model == 'framediff' else 100)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else repo / ('weights/paper_weights.pth' if args.model == 'framediff' else 'weights/pdb/published.ckpt')
    start = time.perf_counter()
    model, cfg, details, sample, au = globals()[args.model](repo,args.length,steps,checkpoint)
    calls = [0]
    def progress(module, inputs, result):
        calls[0] += 1
        if calls[0] == 1 or calls[0] % 50 == 0:
            print('Model forward pass {} / ~{}'.format(calls[0],steps),flush=True)
    hook = model.register_forward_hook(progress)
    setup_seconds = time.perf_counter() - start
    if DEVICE.startswith('cuda'):
        torch.cuda.synchronize()
    sample_start = time.perf_counter()
    coordinates = sample()
    if DEVICE.startswith('cuda'):
        torch.cuda.synchronize()
    sampling_seconds = time.perf_counter() - sample_start
    hook.remove()
    assert coordinates.shape == (args.length,37,3)
    assert np.isfinite(coordinates).all()
    pdb_path = output / ('{}_L{}_seed{}.pdb'.format(args.model,args.length,args.seed))
    au.write_prot_to_pdb(coordinates,str(pdb_path),no_indexing=True)
    OmegaConf.save(cfg,str(output / 'resolved_config.yaml'))
    metadata = {
        'model':args.model, 'length':args.length, 'seed':args.seed,
        'repo_url':'https://github.com/' + ('jasonkyuyim/se3_diffusion' if args.model == 'framediff' else 'microsoft/protein-frame-flow'),
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip(),
        'checkpoint_filename':str(checkpoint.relative_to(repo)), 'checkpoint_sha256':sha256(checkpoint),
        'device':DEVICE, 'gpu':torch.cuda.get_device_name() if DEVICE.startswith('cuda') else None,
        'torch':torch.__version__, 'python':sys.version, 'numpy':np.__version__,
        'setup_seconds':setup_seconds, 'sampling_seconds':sampling_seconds,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if DEVICE.startswith('cuda') else None,
        'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if DEVICE.startswith('cuda') else None,
        'num_forward_passes':calls[0], 'pdb_sha256':sha256(pdb_path),
        'compatibility_adaptations':['Windows atom group indices cast to int64','generation-only wrapper; no training or folding models loaded'],
        'runner_sha256':sha256(runner_path), **details,
    }
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (output/'runner_used.py').write_text(runner_path.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)
    print('Saved',pdb_path,flush=True)
