import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile





DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
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
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
class C_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size = 2,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.h_w = d_model//patch_size
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model // 2)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.x_proj = (
            nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner//2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner//2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner//2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner//2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner//2, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner//2, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner//2)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.fc_pos1 = nn.Linear(1, 2*d_state)
        self.fc_pos2 = nn.Linear(2*d_state, d_state)

    def c_selective_scan_ref(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                             return_last_state=False):
        """
        u: r(B D L)
        delta: r(B D L)
        A: c(D N) or r(D N)
        B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

        out: r(B D L)
        last_state (optional): r(B D dstate) or c(B D dstate)
        """

        dtype_in = u.dtype
        u = u.float()
        L = u.shape[-1]
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
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        epsilon = 1e-6
        for i in range(u.shape[2]):
            u_i = u[:, :, i]
            if not is_variable_B:
                deltaB_u_i = torch.einsum('bd,dn,bd->bdn', delta[:, :, i], B, u_i)
            else:
                if B.dim() == 3:
                    deltaB_u_i = torch.einsum('bd,bn,bd->bdn', delta[:, :, i], B[:, :, i], u_i)
                else:
                    B_i = repeat(B[:, :, :, i], "B G N -> B (G H) N", H=dim // B.shape[1])
                    deltaB_u_i = torch.einsum('bd,bdn,bd->bdn', delta[:, :, i], B_i, u_i)

            Bxt = deltaB_u_i
            Bxt_norm = torch.norm(Bxt, dim=-1, keepdim=True) + epsilon
            h_norm = torch.norm(x, dim=-1, keepdim=True) + epsilon
            similarity = torch.einsum('bdn,bdn->bd', Bxt, x) / (Bxt_norm * h_norm).squeeze(-1)
            similarity  = similarity.unsqueeze(-1)
            h_pos = F.relu(self.fc_pos1(similarity))
            out_pos = self.fc_pos2(h_pos)
            modulation = F.sigmoid(out_pos)
            modulated_Bxt = (self.alpha * modulation + 1) * Bxt
            x = deltaA[:, :, i] * x + modulated_Bxt

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
        y = torch.stack(ys, dim=2)  # (batch dim L)
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = self.c_selective_scan_ref

        B, C, H, W = x.shape # x (b,d,h,w)
        L = H * W
        K = 4
        # 构造水平+垂直扫描特征块
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # 添加正向/反向扫描方向 (K=4: H正向,H反向,W正向,W反向)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y



    def forward(self, Fin: torch.Tensor,B,H,W,C, **kwargs):
        x_h,x_w = Fin.chunk(2,dim=1)

        y_H1, y_H2, y_H3, y_H4 = self.forward_core(x_h)
        y_W1, y_W2, y_W3, y_W4 = self.forward_core(x_w)
        assert y_H1.dtype == torch.float32
        assert y_W1.dtype == torch.float32
        y_H = y_H1 + y_H2 + y_H3 + y_H4
        y_W = y_W1 + y_W2 + y_W3 + y_W4


        y_H = torch.transpose(y_H, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_W = torch.transpose(y_W, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_H = self.out_norm(y_H)
        y_W = self.out_norm(y_W)
        return y_H,y_W



class C_VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.6,
        d_state: int = 8,
        d_conv=3,
        expand=2,
        conv_bias=True,
        bias=False,
        dt_rank="auto",
        **factory_kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.C_SS2D= C_SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **factory_kwargs)
        self.drop_path = DropPath(drop_path)

        self.d_model = hidden_dim
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model // 2)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_h = nn.Linear(self.d_model // 2, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_w = nn.Linear(self.d_model // 2, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.out_proj_H = nn.Linear(self.d_inner // 2, self.d_model // 2, bias=bias, **factory_kwargs)
        self.out_proj_W = nn.Linear(self.d_inner // 2, self.d_model // 2, bias=bias, **factory_kwargs)

    def forward(self, input: torch.Tensor):

        x = self.ln_1(input)
        #  b max(h,w) c h+w
        B,H,W,C = x.shape #b h+w w c
        x_h, x_w = x.chunk(2, dim=-1)
        xz_h = self.in_proj_h(x_h)
        xz_w = self.in_proj_w(x_w)

        x_h, z_h = xz_h.chunk(2, dim=-1)  # (b, h, w, d)
        x_w, z_w = xz_w.chunk(2, dim=-1)

        x = torch.cat([x_h, x_w], dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()# (b, d, h, w)
        x = self.act(self.conv2d(x))
        x = x.permute(0, 2, 3, 1).contiguous()#(b,h,w,d)
        x_h, x_w = x.chunk(2, dim=-1)
        x_h = x_h.permute(0, 3, 1, 2)# (b, d, h, w)
        x_w = x_w.permute(0, 3, 1, 2)
        Fin = torch.cat([x_h,x_w],dim=1)
        y_H,y_W = self.C_SS2D(Fin,B,H,W,C)

        y_H = y_H * F.silu(z_h)
        y_W = y_W * F.silu(z_w)
        # y = y * F.silu(z)
        out_H = self.out_proj_H(y_H)
        out_W = self.out_proj_W(y_W)

        out = torch.cat([out_H, out_W], dim=-1)


        x = input + self.drop_path(out)

        return x

