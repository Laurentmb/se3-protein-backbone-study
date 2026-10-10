"""Explicit Euler and midpoint on SO(3)^L x R^(3L), FrameFlow's product metric.

Convention: R_dot = R hat(omega), so omega is expressed in body coordinates.
These routines integrate a smooth, memoryless vector field. Neural-network
self-conditioning is handled separately by the FrameFlow adapter; its history
means the complete sampler need not inherit the formal second-order accuracy.
"""
import torch


def hat(vector):
    """Turn angular velocity vectors into skew-symmetric matrices."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3))


def exp_so3(vector):
    """Exponentiate in the Lie algebra; never average rotation matrices."""
    return torch.matrix_exp(hat(vector))


def euler_step(field, t, x, rotation, h):
    """Use one field evaluation and a rotation exponential for a full step."""
    velocity, omega = field(t, x, rotation)
    return x + h * velocity, rotation @ exp_so3(h * omega)


def midpoint_step(field, t, x, rotation, h):
    """Use two evaluations: predict a half-step, then update from the base.

For R_dot=R hat(omega), this explicit Lie midpoint has local error O(h^3)
for smooth fields. At order two, the omitted dexp correction contributes
only O(h^3), since the two angular velocities differ by O(h).
"""
    velocity_1, omega_1 = field(t, x, rotation)
    x_half = x + 0.5 * h * velocity_1
    rotation_half = rotation @ exp_so3(0.5 * h * omega_1)
    velocity_half, omega_half = field(t + 0.5 * h, x_half, rotation_half)
    return x + h * velocity_half, rotation @ exp_so3(h * omega_half)


def integrate(field, x, rotation, start, stop, steps, method):
    """Integrate fixed steps and expose the exact field-evaluation budget."""
    if steps < 1 or stop <= start:
        raise ValueError('Require positive steps and stop > start')
    if method not in ('euler', 'midpoint'):
        raise ValueError('Unknown integrator: ' + method)
    step = euler_step if method == 'euler' else midpoint_step
    h = (stop - start) / steps
    for i in range(steps):
        x, rotation = step(field, start + i * h, x, rotation, h)
    return x, rotation, steps * (1 if method == 'euler' else 2)
