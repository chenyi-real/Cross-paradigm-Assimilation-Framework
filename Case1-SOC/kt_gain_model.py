"""Causal GRU Kalman gain and differentiable scalar SOC correction.

The observation is the frozen SOC predictor. The physical prior integrates
current. Reference SOC is never an input to either the gain or the filter.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class TargetKalmanGain(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=50):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)  # initial K=0.5, no imposed final gain

    def forward(self, x, hidden=None):
        x = torch.relu(self.layer_norm(self.fc_in(x)))
        z, hidden = self.gru(x, hidden)
        return torch.sigmoid(self.fc_out(z).squeeze(-1)), hidden

    def initialize_historical(self, checkpoint, mean, std):
        """Preserve the original three-input NASA network before fine-tuning.

        Absorb train-only scaling into the first layer and copy only the SOC
        gain row: RC voltage is unobservable in the SOC-only measurement model.
        """
        old = torch.load(checkpoint, map_location='cpu', weights_only=True)
        with torch.no_grad():
            self.fc_in.weight.zero_()
            self.fc_in.weight[:, :3].copy_(old['fc_in.weight'] * torch.as_tensor(std[:3]))
            self.fc_in.bias.copy_(old['fc_in.bias'] + old['fc_in.weight'] @ torch.as_tensor(mean[:3]))
            for name in ['weight','bias']:
                getattr(self.layer_norm,name).copy_(old[f'layer_norm.{name}'])
            self.gru.load_state_dict({k[4:]:v for k,v in old.items() if k.startswith('gru.')})
            self.fc_out.weight.copy_(old['fc_out.weight'][:1])
            self.fc_out.bias.copy_(old['fc_out.bias'][:1])


def scalar_filter_numpy(obs, delta, gain, initial=None, valid=None):
    """Exact clipped recursive correction, batch x time, float32 or float64."""
    obs, delta, gain = np.asarray(obs), np.asarray(delta), np.asarray(gain)
    if obs.ndim == 1:
        obs, delta, gain = obs[None], delta[None], gain[None]
    if valid is None:
        valid = np.ones_like(obs, dtype=bool)
        valid[:, 0] = False
    state = obs[:, 0].copy() if initial is None else np.asarray(initial).copy()
    out = np.empty_like(obs)
    local_k = np.zeros_like(obs)
    local_state = np.zeros_like(obs)
    for i in range(obs.shape[1]):
        prior_raw = state + delta[:, i]
        prior = np.clip(prior_raw, 0., 1.)
        raw = prior + gain[:, i] * (obs[:, i]-prior)
        updated = np.clip(raw, 0., 1.)
        interior = (raw > 0.) & (raw < 1.)
        local_k[:, i] = np.where(valid[:, i], interior*(obs[:, i]-prior), 0.)
        local_state[:, i] = np.where(valid[:, i], interior*(1-gain[:, i]) *
                                     ((prior_raw > 0.) & (prior_raw < 1.)), 1.)
        state = np.where(valid[:, i], updated, state)
        out[:, i] = state
    return out, local_k, local_state


class _SOCFilter(torch.autograd.Function):
    """Analytic BPTT, with a small CPU scan to avoid thousands of GPU launches.

    Only gains need gradients: observer, physical inputs and chunk initial
    states are frozen/detached. Derivatives include both clipping operations.
    """
    @staticmethod
    def forward(ctx, gain, obs, delta, initial, valid):
        arrays = [x.detach().cpu().numpy() for x in [obs,delta,gain,initial,valid]]
        out, dk, dx = scalar_filter_numpy(arrays[0],arrays[1],arrays[2],arrays[3],arrays[4])
        ctx.dk, ctx.dx, ctx.device, ctx.dtype = dk, dx, gain.device, gain.dtype
        return torch.from_numpy(out).to(device=gain.device, dtype=gain.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.detach().cpu().numpy()
        adj = np.zeros(g.shape[0], dtype=g.dtype)
        result = np.empty_like(g)
        for i in range(g.shape[1]-1, -1, -1):
            adj = adj + g[:, i]
            result[:, i] = adj * ctx.dk[:, i]
            adj = adj * ctx.dx[:, i]
        return torch.from_numpy(result).to(device=ctx.device, dtype=ctx.dtype), None, None, None, None


def differentiable_filter(gain, obs, delta, initial, valid):
    return _SOCFilter.apply(gain, obs, delta, initial, valid)


def verify_filter():
    """Compare forward and gradients to ordinary PyTorch on clipped paths."""
    generator = torch.Generator().manual_seed(112)
    obs = torch.rand(3, 19, generator=generator, dtype=torch.float64)
    delta = torch.randn(3, 19, generator=generator, dtype=torch.float64)*.35
    gain = torch.rand(3, 19, generator=generator, dtype=torch.float64).requires_grad_()
    valid = torch.ones_like(obs, dtype=torch.bool)
    valid[:, 0] = False
    valid[1, 15:] = False
    initial = obs[:, 0].clone()
    actual = differentiable_filter(gain,obs,delta,initial,valid)
    state = initial
    expected = []
    for i in range(obs.shape[1]):
        prior = (state+delta[:, i]).clamp(0, 1)
        updated = (prior+gain[:, i]*(obs[:, i]-prior)).clamp(0, 1)
        state = torch.where(valid[:, i], updated, state)
        expected.append(state)
    expected = torch.stack(expected,1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    a = torch.autograd.grad(actual.square().sum(),gain,retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(),gain)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    # K=1 exactly reproduces the frozen observation after initialization.
    unit = scalar_filter_numpy(obs.numpy(),delta.numpy(),np.ones_like(obs.numpy()))[0]
    np.testing.assert_allclose(unit,obs.numpy(),atol=1e-12)
    return {'forward_and_gradient_check':'passed', 'K_1_observer_identity':'passed'}
