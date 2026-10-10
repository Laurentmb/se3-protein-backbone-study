"""Check order, noncommuting rotations, SO(3) constraints and equivariance.

The reference trajectory is analytic, not an output from the implementation
under test. Tests use float64 to separate discretization error from roundoff.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from lie_integrators import exp_so3, integrate

DTYPE = torch.float64


def log_so3(matrix):
    """Use SciPy's independent logarithm for the analytic test field."""
    shape = matrix.shape[:-2]
    vector = Rotation.from_matrix(matrix.numpy().reshape(-1, 3, 3)).as_rotvec()
    return torch.tensor(vector.reshape(shape + (3,)), dtype=DTYPE)


def reference(t):
    """An exact path with noncommuting rotation axes and curved translation."""
    a = torch.tensor([0.7, -0.2, 0.1], dtype=DTYPE)
    b = torch.tensor([-0.1, 0.4, 0.3], dtype=DTYPE)
    x = torch.tensor([np.exp(-t), np.sin(t), 0.5 + t*t], dtype=DTYPE)
    # SciPy supplies the analytic rotations independently of our exponential.
    r_a = torch.tensor(Rotation.from_rotvec((t*a).numpy()).as_matrix(), dtype=DTYPE)
    r_b = torch.tensor(Rotation.from_rotvec((t*t*b).numpy()).as_matrix(), dtype=DTYPE)
    r = r_a @ r_b
    # Differentiating R=Exp(tA)Exp(t^2 B) yields this body angular velocity.
    omega = r_b.T @ a + 2*t*b
    velocity = torch.tensor([-np.exp(-t), np.cos(t), 2*t], dtype=DTYPE)
    return x, r, velocity, omega


def field(t, x, rotation):
    """Feedback makes the field depend on state as well as time."""
    target_x, target_r, velocity, omega = reference(t)
    return velocity + 0.4*(target_x-x), omega + 0.6*log_so3(
        rotation.transpose(-1, -2) @ target_r)


def constraints(rotation):
    identity = torch.eye(3, dtype=rotation.dtype)
    return {'orthogonality_error': float(torch.max(torch.abs(
        rotation.transpose(-1, -2) @ rotation - identity))),
        'determinant_error': float(torch.max(torch.abs(torch.det(rotation)-1)))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    start_x, start_r, _, _ = reference(0.0)
    target_x, target_r, _, _ = reference(1.0)
    result = {'precision': 'float64', 'convergence': {}, 'checks': {}}

    # Error should halve for Euler and quarter for midpoint when h is halved.
    for method in ('euler', 'midpoint'):
        rows = []
        for steps in (8, 16, 32, 64):
            x, r, calls = integrate(field, start_x, start_r, 0, 1, steps, method)
            rows.append({'steps': steps, 'field_evaluations': calls,
                'translation_error': float(torch.linalg.norm(x-target_x)),
                'rotation_error_rad': float(torch.linalg.norm(log_so3(
                    r.T @ target_r))), **constraints(r)})
        for key in ('translation_error', 'rotation_error_rad'):
            order = np.log2(rows[-2][key] / rows[-1][key])
            assert (0.85 < order < 1.15) if method == 'euler' else (1.8 < order < 2.2)
            result['checks'][method + '_' + key + '_order'] = float(order)
        assert max(row['orthogonality_error'] for row in rows) < 1e-11
        assert max(row['determinant_error'] for row in rows) < 1e-11
        result['convergence'][method] = rows

    # Constant angular velocity has an exact exponential solution.
    angular = torch.tensor([0.1, -0.4, 0.2], dtype=DTYPE)
    velocity = torch.tensor([0.2, 0.3, -0.1], dtype=DTYPE)
    constant = lambda t, x, r: (velocity.expand_as(x), angular.expand(x.shape))
    for method in ('euler', 'midpoint'):
        x, r, _ = integrate(constant, start_x, start_r, 0, 1, 12, method)
        assert torch.allclose(x, start_x+velocity, atol=1e-12, rtol=0)
        assert torch.allclose(r, start_r @ exp_so3(angular), atol=1e-12, rtol=0)
    result['checks']['constant_field_exact'] = True

    # A global change of reference frame must commute with integration.
    q = exp_so3(torch.tensor([0.3, 0.1, -0.2], dtype=DTYPE))
    offset = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
    def transformed(t, x, r):
        v, w = field(t, q.T @ (x-offset), q.T @ r)
        return q @ v, w
    errors = {}
    for method in ('euler', 'midpoint'):
        x, r, _ = integrate(field, start_x, start_r, 0, 1, 32, method)
        tx, tr, _ = integrate(transformed, q @ start_x+offset,
            q @ start_r, 0, 1, 32, method)
        error = max(float(torch.max(torch.abs(tx-(q @ x+offset)))),
            float(torch.max(torch.abs(tr-q @ r))))
        assert error < 1e-11
        errors[method] = error
    result['checks']['global_SE3_equivariance_errors'] = errors

    # Batched residues and zero angular velocity exercise shape/identity cases.
    bx = start_x.repeat(2, 4, 1)
    br = start_r.repeat(2, 4, 1, 1)
    zero = lambda t, x, r: (torch.zeros_like(x), torch.zeros_like(x))
    x, r, calls = integrate(zero, bx, br, 0, 1, 4, 'midpoint')
    assert torch.equal(x, bx) and torch.equal(r, br) and calls == 8
    result['checks']['batched_zero_field'] = True
    result['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
