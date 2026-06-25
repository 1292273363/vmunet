import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.region_mamba.soft_slic import SoftSLIC
from models.region_mamba import SuperpixelRegionGraphMambaBlock, SuperpixelSkipRefine
from models.region_mamba.sp_scan_ordering import (
    batched_gather_tokens,
    build_superpixel_graph_token_permutation,
)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


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

    flops = 0 # below code flops = 0
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


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
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
        use_sp_scan=False,
        sp_scan_cfg=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
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

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.use_sp_scan = use_sp_scan
        self.sp_scan_cfg = dict(sp_scan_cfg or {})
        self.sp_scan_mode = self.sp_scan_cfg.get('mode', 'replacement')
        self.sp_scan_replace_mode = self.sp_scan_cfg.get('replace_mode', 'two_paths')
        self.extra_path_types = ()
        self.last_sp_scan_stats = None
        self.sp_scan_soft_slic = None
        self.gamma_graph = None
        self.gamma_reverse_graph = None
        if self.use_sp_scan:
            if self.sp_scan_cfg.get('graph_order', 'greedy') != 'greedy':
                raise ValueError("SS2D sp_scan currently supports graph_order='greedy' only.")
            if self.sp_scan_mode == 'replacement':
                if self.sp_scan_replace_mode not in {'none', 'one_path', 'two_paths'}:
                    raise ValueError(
                        "SS2D sp_scan replace_mode must be one of "
                        "['none', 'one_path', 'two_paths']."
                    )
            elif self.sp_scan_mode == 'extra_path':
                self.extra_path_types = tuple(self.sp_scan_cfg.get('extra_path_types', ()))
                if not self.extra_path_types:
                    raise ValueError("extra_path mode requires at least one extra_path_types entry.")
                unknown_types = set(self.extra_path_types) - {'graph', 'reverse_graph'}
                if unknown_types:
                    raise ValueError(
                        "extra_path_types supports only ['graph', 'reverse_graph']; "
                        f"got {sorted(unknown_types)}."
                    )
                gamma_sp_init = float(self.sp_scan_cfg.get('gamma_sp_init', 1e-3))
                if 'graph' in self.extra_path_types:
                    self.gamma_graph = nn.Parameter(torch.tensor(gamma_sp_init, **factory_kwargs))
                if 'reverse_graph' in self.extra_path_types:
                    self.gamma_reverse_graph = nn.Parameter(torch.tensor(gamma_sp_init, **factory_kwargs))
            else:
                raise ValueError("SS2D sp_scan mode must be 'replacement' or 'extra_path'.")
            soft_slic_cfg = {
                'num_regions': self.sp_scan_cfg.get('num_regions', (2, 2)),
                'num_iters': self.sp_scan_cfg.get('num_iters', 5),
                'tau': self.sp_scan_cfg.get('tau', 0.2),
                'xy_weight': self.sp_scan_cfg.get('xy_weight', 2.0),
                'feat_weight': self.sp_scan_cfg.get('feat_weight', 0.1),
                'normalize_assign': self.sp_scan_cfg.get('normalize_assign', True),
                'assign_norm': self.sp_scan_cfg.get('assign_norm', 'layer'),
                'return_distance_stats': self.sp_scan_cfg.get('debug_stats', True),
            }
            self.sp_scan_soft_slic = SoftSLIC(**soft_slic_cfg)

        # self.selective_scan = selective_scan_fn
        self.forward_core = (
            self.forward_core_sp_scan
            if self.use_sp_scan and self.sp_scan_mode == 'replacement' and self.sp_scan_replace_mode != 'none'
            else self.forward_corev0
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

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

    def _restore_graph_scan_output(self, y, inv_perm):
        # y: [B, D, L] in graph order, inv_perm: [B, L]
        y_tokens = y.transpose(1, 2).contiguous()
        y_tokens = batched_gather_tokens(y_tokens, inv_perm)
        return y_tokens.transpose(1, 2).contiguous()

    def _build_sp_scan_permutation(self, x):
        """Return graph ordering for dense tokens without changing their representation."""
        return build_superpixel_graph_token_permutation(
            x,
            self.sp_scan_soft_slic,
            k_spatial=self.sp_scan_cfg.get('k_spatial', 3),
            k_feature=self.sp_scan_cfg.get('k_feature', 3),
            alpha=self.sp_scan_cfg.get('alpha', 0.5),
            beta=self.sp_scan_cfg.get('beta', 0.5),
            token_inner_order=self.sp_scan_cfg.get('token_inner_order', 'raster'),
            detach_order=self.sp_scan_cfg.get('detach_order', True),
        )

    def _run_extra_selective_scan(self, paths, path_indices):
        """Scan extra paths while reusing the original K=4 SS2D parameters.

        Args:
            paths: Tensor[B, P, D, L].
            path_indices: tuple/list of original SS2D path indices in [0, 3].
        """
        self.selective_scan = selective_scan_fn
        B, num_paths, _, L = paths.shape
        indices = torch.as_tensor(path_indices, device=paths.device, dtype=torch.long)

        x_proj_weight = self.x_proj_weight.index_select(0, indices)
        dt_projs_weight = self.dt_projs_weight.index_select(0, indices)
        dt_projs_bias = self.dt_projs_bias.index_select(0, indices)
        Ds = self.Ds.float().view(4, self.d_inner).index_select(0, indices).reshape(-1)
        As = -torch.exp(self.A_logs.float()).view(4, self.d_inner, self.d_state)
        As = As.index_select(0, indices).reshape(-1, self.d_state)

        x_dbl = torch.einsum("b p d l, p c d -> b p c l", paths, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b p r l, p d r -> b p d l", dts, dt_projs_weight)

        xs = paths.float().reshape(B, -1, L)
        dts = dts.contiguous().float().reshape(B, -1, L)
        Bs = Bs.float()
        Cs = Cs.float()

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias.float().reshape(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, num_paths, -1, L)
        assert out_y.dtype == torch.float
        return out_y

    def forward_core_extra_path(self, x: torch.Tensor):
        """Keep original SS2D paths and add graph-ordered paths as gated residuals."""
        B, _, H, W = x.shape
        L = H * W

        # Preserve the original four-path VMamba computation exactly.
        y0, y1, y2, y3 = self.forward_corev0(x)
        y_original = y0 + y1 + y2 + y3

        perm, inv_perm, sp_stats = self._build_sp_scan_permutation(x)
        raster_tokens = x.view(B, -1, L).transpose(1, 2).contiguous()
        graph_tokens = batched_gather_tokens(raster_tokens, perm)
        graph_path = graph_tokens.transpose(1, 2).contiguous()
        reverse_graph_path = torch.flip(graph_path, dims=[-1])

        # E1 reuses the path3 parameters used by V1's graph replacement.
        # E2 reuses path2 for graph and path3 for reverse_graph, matching V2.
        extra_paths = []
        extra_indices = []
        if 'graph' in self.extra_path_types:
            graph_param_index = 2 if 'reverse_graph' in self.extra_path_types else 3
            extra_paths.append(graph_path)
            extra_indices.append(graph_param_index)
        if 'reverse_graph' in self.extra_path_types:
            extra_paths.append(reverse_graph_path)
            extra_indices.append(3)

        extra_outputs = self._run_extra_selective_scan(
            torch.stack(extra_paths, dim=1),
            extra_indices,
        )
        output_idx = 0
        y_graph = None
        y_reverse_graph = None
        if 'graph' in self.extra_path_types:
            y_graph = self._restore_graph_scan_output(extra_outputs[:, output_idx], inv_perm)
            output_idx += 1
        if 'reverse_graph' in self.extra_path_types:
            y_reverse_graph = self._restore_graph_scan_output(
                torch.flip(extra_outputs[:, output_idx], dims=[-1]),
                inv_perm,
            )

        self.last_sp_scan_stats = dict(sp_stats)
        self.last_sp_scan_stats.update({
            'sp_scan_mode': 'extra_path',
            'extra_path_types': self.extra_path_types,
            'extra_path_param_indices': tuple(extra_indices),
            'token_inner_order': self.sp_scan_cfg.get('token_inner_order', 'raster'),
            'graph_order': self.sp_scan_cfg.get('graph_order', 'greedy'),
            'path_types': ('raster', 'transpose', 'reverse_raster', 'reverse_transpose'),
            'uses_mamba': 'selective_scan_fn' in globals(),
        })
        return y_original, y_graph, y_reverse_graph

    def forward_core_sp_scan(self, x: torch.Tensor):
        """Selective scan with one or two dense-token paths ordered by a superpixel graph."""
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        raster = x.view(B, -1, L)
        transpose_raster = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        reverse_raster = torch.flip(raster, dims=[-1])

        perm, inv_perm, sp_stats = self._build_sp_scan_permutation(x)
        x_tokens = raster.transpose(1, 2).contiguous()  # [B, L, D]
        graph_tokens = batched_gather_tokens(x_tokens, perm)
        graph_path = graph_tokens.transpose(1, 2).contiguous()  # [B, D, L]
        reverse_graph_path = torch.flip(graph_path, dims=[-1])

        replace_mode = self.sp_scan_replace_mode
        if replace_mode == 'one_path':
            xs = torch.stack([raster, transpose_raster, reverse_raster, graph_path], dim=1)
            path_types = ('raster', 'transpose', 'reverse_raster', 'graph')
        elif replace_mode == 'two_paths':
            xs = torch.stack([raster, transpose_raster, graph_path, reverse_graph_path], dim=1)
            path_types = ('raster', 'transpose', 'graph', 'reverse_graph')
        else:
            raise ValueError(f"Unsupported sp_scan replace_mode: {replace_mode}")

        self.last_sp_scan_stats = dict(sp_stats)
        self.last_sp_scan_stats.update({
            'sp_scan_mode': 'replacement',
            'replace_mode': replace_mode,
            'path_types': path_types,
            'token_inner_order': self.sp_scan_cfg.get('token_inner_order', 'raster'),
            'graph_order': self.sp_scan_cfg.get('graph_order', 'greedy'),
            'uses_mamba': 'selective_scan_fn' in globals(),
        })

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y0 = out_y[:, 0]
        y1 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        if replace_mode == 'one_path':
            y2 = torch.flip(out_y[:, 2], dims=[-1])
            y3 = self._restore_graph_scan_output(out_y[:, 3], inv_perm)
        else:
            y2 = self._restore_graph_scan_output(out_y[:, 2], inv_perm)
            y3 = self._restore_graph_scan_output(torch.flip(out_y[:, 3], dims=[-1]), inv_perm)

        return y0, y2, y1, y3

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        if self.use_sp_scan and self.sp_scan_mode == 'extra_path':
            y_original, y_graph, y_reverse_graph = self.forward_core_extra_path(x)
            assert y_original.dtype == torch.float32
            y = y_original
            gamma_values = []

            if y_graph is not None:
                y = y + self.gamma_graph * y_graph
                gamma_values.append(self.gamma_graph)
            if y_reverse_graph is not None:
                y = y + self.gamma_reverse_graph * y_reverse_graph
                gamma_values.append(self.gamma_reverse_graph)

            def _rms_norm(tensor):
                return tensor.float().square().mean().sqrt().detach()

            stats = self.last_sp_scan_stats
            stats['gamma_graph'] = self.gamma_graph.detach() if self.gamma_graph is not None else None
            stats['gamma_reverse_graph'] = (
                self.gamma_reverse_graph.detach() if self.gamma_reverse_graph is not None else None
            )
            if gamma_values:
                gamma_stack = torch.stack(gamma_values)
                stats['gamma_sp_mean'] = gamma_stack.mean().detach()
                stats['gamma_sp_abs_mean'] = gamma_stack.abs().mean().detach()
            stats['y_original_norm'] = _rms_norm(y_original)
            if y_graph is not None:
                stats['y_graph_norm'] = _rms_norm(y_graph)
                stats['graph_orig_norm_ratio'] = (
                    stats['y_graph_norm'] / stats['y_original_norm'].clamp_min(1e-6)
                )
            if y_reverse_graph is not None:
                stats['y_reverse_graph_norm'] = _rms_norm(y_reverse_graph)
                stats['reverse_graph_orig_norm_ratio'] = (
                    stats['y_reverse_graph_norm'] / stats['y_original_norm'].clamp_min(1e-6)
                )
        else:
            y1, y2, y3, y4 = self.forward_core(x)
            assert y1.dtype == torch.float32
            y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_sp_scan: bool = False,
        sp_scan_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            use_sp_scan=use_sp_scan,
            sp_scan_cfg=sp_scan_cfg,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        sp_scan_blocks=None,
        sp_scan_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        def _use_sp_scan_for_block(block_idx):
            if sp_scan_blocks is None:
                return False
            if sp_scan_blocks == 'all':
                return True
            if sp_scan_blocks == 'last':
                return block_idx == depth - 1
            if sp_scan_blocks == 'last2':
                return block_idx >= max(depth - 2, 0)
            if isinstance(sp_scan_blocks, (list, tuple, set)):
                return block_idx in sp_scan_blocks
            raise ValueError(
                "sp_scan_blocks must be None, 'last', 'last2', 'all', or a list/tuple/set of indices."
            )

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_sp_scan=_use_sp_scan_for_block(i),
                sp_scan_cfg=sp_scan_cfg,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, use_sp_rgm=False, sp_rgm_cfg=None,
                 use_sp_scan=False, sp_scan_cfg=None, sp_scan_stage=None, sp_scan_blocks=None,
                 use_ssr=False, ssr_cfg=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.use_sp_rgm = use_sp_rgm
        self.use_sp_scan = use_sp_scan
        self.use_ssr = use_ssr
        self.sp_scan_cfg = dict(sp_scan_cfg or {})
        self.sp_scan_stage = sp_scan_stage
        self.sp_scan_blocks = sp_scan_blocks
        self.ssr_cfg = dict(ssr_cfg or {})
        self.bottleneck_depth = depths[-1]
        self.stage2_depth = depths[2] if len(depths) > 2 else None
        self.enabled_sp_scan_stage_index = None
        self.enabled_sp_scan_block_indices = []
        if sum(bool(flag) for flag in (self.use_sp_rgm, self.use_sp_scan, self.use_ssr)) > 1:
            raise ValueError("use_sp_rgm, use_sp_scan, and use_ssr are mutually exclusive.")
        if self.use_sp_scan:
            if self.sp_scan_stage == 'bottleneck_last':
                self.sp_scan_blocks = 'last'
            elif self.sp_scan_stage == 'bottleneck_all':
                self.sp_scan_stage = 'bottleneck'
                self.sp_scan_blocks = 'all'
            elif self.sp_scan_stage == 'bottleneck':
                self.sp_scan_blocks = self.sp_scan_blocks or self.sp_scan_cfg.get('sp_scan_blocks', 'last')
            elif self.sp_scan_stage == 'stage2':
                if self.num_layers <= 2:
                    raise ValueError("sp_scan_stage='stage2' requires an encoder with at least three stages.")
                self.sp_scan_blocks = self.sp_scan_blocks or self.sp_scan_cfg.get('sp_scan_blocks', 'last')
            else:
                raise ValueError(
                    "SPScan supports sp_scan_stage in "
                    "['bottleneck_last', 'bottleneck', 'bottleneck_all', 'stage2']."
                )
            if self.sp_scan_blocks not in {'last', 'last2', 'all'} and not isinstance(self.sp_scan_blocks, (list, tuple, set)):
                raise ValueError(
                    "sp_scan_blocks must be 'last', 'last2', 'all', or a list/tuple/set of block indices."
                )
            self.enabled_sp_scan_stage_index = (
                2 if self.sp_scan_stage == 'stage2' else self.num_layers - 1
            )

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_sp_scan_blocks = None
            if self.use_sp_scan and i_layer == self.enabled_sp_scan_stage_index:
                layer_sp_scan_blocks = self.sp_scan_blocks
                if layer_sp_scan_blocks == 'last':
                    self.enabled_sp_scan_block_indices = [depths[i_layer] - 1]
                elif layer_sp_scan_blocks == 'last2':
                    self.enabled_sp_scan_block_indices = list(range(max(depths[i_layer] - 2, 0), depths[i_layer]))
                elif layer_sp_scan_blocks == 'all':
                    self.enabled_sp_scan_block_indices = list(range(depths[i_layer]))
                else:
                    self.enabled_sp_scan_block_indices = sorted(int(idx) for idx in layer_sp_scan_blocks)
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                sp_scan_blocks=layer_sp_scan_blocks,
                sp_scan_cfg=self.sp_scan_cfg if layer_sp_scan_blocks is not None else None,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        if self.use_sp_rgm:
            sp_rgm_cfg = sp_rgm_cfg or {}
            self.sp_rgm = SuperpixelRegionGraphMambaBlock(
                dim=self.num_features,
                **sp_rgm_cfg,
            )
        else:
            self.sp_rgm = None

        self.ssr_modules = nn.ModuleDict()
        self.ssr_stages = tuple(self.ssr_cfg.get('ssr_stages', ())) if self.use_ssr else tuple()
        self.last_ssr_stats = {}
        if self.use_ssr:
            if not self.ssr_cfg.get('enabled', True):
                raise ValueError("use_ssr=True requires ssr_cfg['enabled']=True.")
            valid_stages = {f'stage{i}' for i in range(self.num_layers)}
            unknown_stages = set(self.ssr_stages) - valid_stages
            if unknown_stages:
                raise ValueError(f"Unknown SSR stages: {sorted(unknown_stages)}. Valid stages: {sorted(valid_stages)}.")
            if not self.ssr_stages:
                raise ValueError("use_ssr=True requires ssr_cfg['ssr_stages'] to be non-empty.")

            num_regions_cfg = self.ssr_cfg.get('num_regions', {})
            shared_num_regions = None if isinstance(num_regions_cfg, dict) else num_regions_cfg
            for stage_name in self.ssr_stages:
                stage_index = int(stage_name.replace('stage', ''))
                stage_num_regions = (
                    tuple(num_regions_cfg.get(stage_name, (4, 4)))
                    if isinstance(num_regions_cfg, dict)
                    else tuple(shared_num_regions or (4, 4))
                )
                self.ssr_modules[stage_name] = SuperpixelSkipRefine(
                    dim=dims[stage_index],
                    num_regions=stage_num_regions,
                    num_iters=self.ssr_cfg.get('num_iters', 5),
                    tau=self.ssr_cfg.get('tau', 0.2),
                    xy_weight=self.ssr_cfg.get('xy_weight', 2.0),
                    feat_weight=self.ssr_cfg.get('feat_weight', 0.1),
                    normalize_assign=self.ssr_cfg.get('normalize_assign', True),
                    assign_norm=self.ssr_cfg.get('assign_norm', 'layer'),
                    use_pos_embed=self.ssr_cfg.get('use_pos_embed', True),
                    use_avg_pool=self.ssr_cfg.get('use_avg_pool', True),
                    use_max_pool=self.ssr_cfg.get('use_max_pool', True),
                    use_graph=self.ssr_cfg.get('use_graph', False),
                    region_update=self.ssr_cfg.get('region_update', 'mlp'),
                    gamma_init=self.ssr_cfg.get('gamma_init', 1e-3),
                    gate_type=self.ssr_cfg.get('gate_type', 'bounded_tanh'),
                    gate_scale=self.ssr_cfg.get('gate_scale', 0.1),
                    debug_stats=self.ssr_cfg.get('debug_stats', True),
                    stage_name=stage_name,
                )

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    @torch.jit.ignore
    def get_sp_scan_stats(self):
        """Return aggregated SPScan diagnostics from enabled SS2D blocks."""
        if self.enabled_sp_scan_stage_index is None:
            return None
        stats_by_block = []
        for block_idx, block in enumerate(self.layers[self.enabled_sp_scan_stage_index].blocks):
            stats = getattr(block.self_attention, 'last_sp_scan_stats', None)
            if stats is not None:
                stats_by_block.append((block_idx, dict(stats)))
        if not stats_by_block:
            return None
        stats_list = [stats for _, stats in stats_by_block]

        aggregated = {}
        numeric_keys = set()
        for stats in stats_list:
            for key, value in stats.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    numeric_keys.add(key)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_keys.add(key)

        for key in numeric_keys:
            values = []
            for stats in stats_list:
                value = stats.get(key)
                if torch.is_tensor(value) and value.numel() == 1:
                    values.append(value.detach().float())
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(torch.tensor(float(value)))
            if values:
                aggregated[key] = torch.stack([v.to(values[0].device) for v in values]).mean()

        latest_stats = stats_list[-1]
        for key, value in latest_stats.items():
            aggregated.setdefault(key, value)

        aggregated['sp_scan_stage'] = self.sp_scan_stage
        aggregated['sp_scan_blocks'] = self.sp_scan_blocks
        aggregated['enabled_sp_scan_stage_index'] = self.enabled_sp_scan_stage_index
        aggregated['enabled_sp_scan_block_indices'] = tuple(self.enabled_sp_scan_block_indices)
        aggregated['stage2_depth'] = self.stage2_depth
        aggregated['bottleneck_depth'] = self.bottleneck_depth
        aggregated['num_enabled_sp_scan_blocks'] = len(self.enabled_sp_scan_block_indices)
        aggregated['sp_scan_mode'] = latest_stats.get('sp_scan_mode', 'replacement')
        if self.sp_scan_stage == 'stage2':
            aggregated['stage2_feat_h'] = aggregated.get('feat_h')
            aggregated['stage2_feat_w'] = aggregated.get('feat_w')
        if 'extra_path_types' in latest_stats:
            aggregated['extra_path_types'] = latest_stats['extra_path_types']
            aggregated['num_extra_paths'] = len(latest_stats['extra_path_types'])
        aggregated['perm_valid'] = all(bool(stats.get('perm_valid', False)) for stats in stats_list)
        aggregated['uses_mamba'] = all(bool(stats.get('uses_mamba', False)) for stats in stats_list)

        gamma_values = []
        gamma_names = []
        for block_idx, stats in stats_by_block:
            for gamma_key in ('gamma_graph', 'gamma_reverse_graph'):
                gamma_value = stats.get(gamma_key)
                if gamma_value is not None:
                    aggregated[f'{gamma_key}_block{block_idx}'] = gamma_value
                    gamma_values.append(gamma_value.detach().float())
                    gamma_names.append(
                        f'layers.{self.enabled_sp_scan_stage_index}.blocks.{block_idx}.self_attention.{gamma_key}'
                    )
        if gamma_values:
            gamma_stack = torch.stack(gamma_values)
            aggregated['gamma_sp_mean'] = gamma_stack.mean()
            aggregated['gamma_sp_abs_mean'] = gamma_stack.abs().mean()
            aggregated['extra_gate_parameter_names'] = tuple(gamma_names)
            aggregated['extra_gate_parameter_count'] = len(gamma_names)
        return aggregated

    @torch.jit.ignore
    def get_ssr_stats(self):
        """Return aggregated SSR diagnostics from the refined skip feature(s)."""
        if not self.use_ssr or not self.last_ssr_stats:
            return None

        stats_items = list(self.last_ssr_stats.items())
        stats_list = [stats for _, stats in stats_items]
        aggregated = {}
        numeric_keys = set()
        for stats in stats_list:
            for key, value in stats.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    numeric_keys.add(key)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_keys.add(key)

        for key in numeric_keys:
            values = []
            for stats in stats_list:
                value = stats.get(key)
                if torch.is_tensor(value) and value.numel() == 1:
                    values.append(value.detach().float())
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(torch.tensor(float(value)))
            if values:
                aggregated[key] = torch.stack([v.to(values[0].device) for v in values]).mean()

        latest_stats = stats_list[-1]
        for key, value in latest_stats.items():
            aggregated.setdefault(key, value)
        aggregated['use_ssr'] = True
        aggregated['ssr_stages'] = tuple(self.ssr_stages)
        aggregated['ssr_enabled_stages'] = tuple(stage for stage, _ in stats_items)
        return aggregated

    def apply_ssr_to_skip(self, skip, stage_name):
        """Apply SSR to a BHWC skip tensor and return the same BHWC shape."""
        if not self.use_ssr or stage_name not in self.ssr_modules:
            return skip
        skip_nchw = skip.permute(0, 3, 1, 2).contiguous()
        refined_nchw, stats = self.ssr_modules[stage_name](skip_nchw, return_stats=True)
        self.last_ssr_stats[stage_name] = stats
        return refined_nchw.permute(0, 2, 3, 1).contiguous()

    def forward_features(self, x, return_aux=False):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)

        aux = {}
        if self.sp_rgm is not None:
            # x_nchw: [B, C, H, W] -> x: [B, H, W, C]
            x_nchw = x.permute(0, 3, 1, 2).contiguous()
            x_nchw, sp_rgm_aux = self.sp_rgm(x_nchw)
            x = x_nchw.permute(0, 2, 3, 1).contiguous()
            if return_aux:
                aux['sp_rgm'] = sp_rgm_aux

        if return_aux and self.use_sp_scan:
            sp_scan_stats = self.get_sp_scan_stats()
            if sp_scan_stats is not None:
                aux['sp_scan'] = sp_scan_stats

        return x, skip_list, aux
    
    def forward_features_up(self, x, skip_list):
        self.last_ssr_stats = {}
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip = skip_list[-inx]
                stage_index = self.num_layers - inx
                stage_name = f'stage{stage_index}'
                skip = self.apply_ssr_to_skip(skip, stage_name)
                x = layer_up(x + skip)

        return x
    
    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x, return_aux=False):
        x, skip_list, aux = self.forward_features(x, return_aux=return_aux)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)

        if return_aux:
            if self.use_ssr:
                ssr_stats = self.get_ssr_stats()
                if ssr_stats is not None:
                    aux['ssr'] = ssr_stats
            return x, aux
        return x




    
