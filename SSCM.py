import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_




DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from timm.models.layers import DropPath



def selective_scan_ref_modified(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                                return_last_state=False, initial_state=None):

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    if initial_state is not None:
        assert initial_state.shape[-1] == dstate, "d_state of initial_state must match A.shape[1]"
        x = initial_state.float()
    else:
        x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:

            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")

    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops




class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):  # i: index;   feat: value
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class HierarchicalSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self._init_path_params("semantic", factory_kwargs, d_conv, conv_bias, bias, dt_scale, dt_init, dt_min, dt_max,
                               dt_init_floor, **kwargs)
        self._init_path_params("structural", factory_kwargs, d_conv, conv_bias, bias, dt_scale, dt_init, dt_min, dt_max,
                               dt_init_floor, **kwargs)

        self.act = nn.SiLU()

        self.out_norm = nn.LayerNorm(self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model * 2, bias=bias, **factory_kwargs)

    def _init_path_params(self, path_name, factory_kwargs, d_conv, conv_bias, bias, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **kwargs):

        in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs
        )
        x_proj_tuple = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj_tuple], dim=0))

        dt_projs_tuple = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs_tuple], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs_tuple], dim=0))

        A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        Ds = self.D_init(self.d_inner, copies=4, merge=True)

        setattr(self, f"{path_name}_in_proj", in_proj)
        setattr(self, f"{path_name}_conv2d", conv2d)
        setattr(self, f"{path_name}_x_proj_weight", x_proj_weight)
        setattr(self, f"{path_name}_dt_projs_weight", dt_projs_weight)
        setattr(self, f"{path_name}_dt_projs_bias", dt_projs_bias)
        setattr(self, f"{path_name}_A_logs", A_logs)
        setattr(self, f"{path_name}_Ds", Ds)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):

        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_hierarchical(self, x_sem, x_str):
        B, C, H, W = x_sem.shape
        L = H * W
        K = 4


        def scan_path(x, path_name, initial_state=None):

            xs = torch.stack([x.view(B, C, L), torch.transpose(x, 2, 3).contiguous().view(B, C, L)], 1).view(B, 2, C, L)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], 1)


            x_proj_weight = getattr(self, f"{path_name}_x_proj_weight")
            dt_projs_weight = getattr(self, f"{path_name}_dt_projs_weight")
            dt_projs_bias = getattr(self, f"{path_name}_dt_projs_bias")
            A_logs = getattr(self, f"{path_name}_A_logs")
            Ds = getattr(self, f"{path_name}_Ds")

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)

            As = -torch.exp(A_logs.float()).view(-1, self.d_state)


            out_y, last_state = selective_scan_ref_modified(
                xs.float().view(B, -1, L), dts.contiguous().float().view(B, -1, L), As,
                Bs.float().view(B, K, -1, L), Cs.float().view(B, K, -1, L), Ds.float().view(-1),
                delta_bias=dt_projs_bias.float().view(-1), delta_softplus=True,
                return_last_state=True,
                initial_state=initial_state
            )
            out_y = out_y.view(B, K, C, L)


            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

            return (out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y), last_state


        y_sem, last_state_sem = scan_path(x_sem, "semantic")

        y_str, _ = scan_path(x_str, "structural", initial_state=last_state_sem)

        return y_sem, y_str

    def forward(self, semantic_x, structural_x):
        B, H, W, C_sem = semantic_x.shape


        sem_xz = self.semantic_in_proj(semantic_x)
        sem_x, sem_z = sem_xz.chunk(2, dim=-1)
        sem_x = self.act(self.semantic_conv2d(sem_x.permute(0, 3, 1, 2).contiguous()))


        str_xz = self.structural_in_proj(structural_x)
        str_x, str_z = str_xz.chunk(2, dim=-1)
        str_x = self.act(self.structural_conv2d(str_x.permute(0, 3, 1, 2).contiguous()))


        y_sem, y_str = self.forward_core_hierarchical(sem_x, str_x)


        y = torch.cat([y_sem, y_str], dim=1)
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)

        z = torch.cat([sem_z, str_z], dim=-1)


        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y

class HierarchicalVSSBlock(nn.Module):
    def __init__(self, hidden_dim, d_state=16, drop_path=0., **kwargs):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be divisible by 2 for semantic/structural split"
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = HierarchicalSS2D(d_model=hidden_dim // 2, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        shortcut = input
        x = self.ln_1(input)
        semantic_x, structural_x = x.chunk(2, dim=-1)
        y = self.self_attention(semantic_x, structural_x)
        return shortcut + self.drop_path(y)







