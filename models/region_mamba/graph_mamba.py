import torch
from torch import nn

from .graph_ordering import (
    argsort_xy,
    batched_gather_tokens,
    greedy_graph_path,
    invert_permutation,
)
from .region_graph import build_region_graph

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class RegionGraphMamba(nn.Module):
    """Run multi-path sequence modeling over superpixel region tokens."""

    VALID_PATH_MODES = {'yx', 'xy', 'graph', 'reverse_graph'}
    DEFAULT_PATH_MODES = ('yx', 'xy', 'graph', 'reverse_graph')

    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        k_spatial=6,
        k_feature=6,
        alpha=0.5,
        beta=0.5,
        init_gamma=1e-3,
        path_modes=None,
        allow_gru_fallback=False,
    ):
        super().__init__()
        self.k_spatial = k_spatial
        self.k_feature = k_feature
        self.alpha = alpha
        self.beta = beta
        if path_modes is None:
            path_modes = self.DEFAULT_PATH_MODES
        elif isinstance(path_modes, str):
            path_modes = (path_modes,)
        self.path_modes = tuple(path_modes)
        unknown_modes = sorted(set(self.path_modes) - self.VALID_PATH_MODES)
        if unknown_modes:
            raise ValueError(
                f"Unknown RegionGraphMamba path mode(s): {unknown_modes}. "
                f"Supported modes are {sorted(self.VALID_PATH_MODES)}."
            )
        if len(self.path_modes) == 0:
            raise ValueError('RegionGraphMamba requires at least one path mode.')

        use_mamba = Mamba is not None and not (allow_gru_fallback and not torch.cuda.is_available())
        if use_mamba:
            self.sequence_model = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.uses_mamba = True
        else:
            if not allow_gru_fallback:
                raise ImportError(
                    'mamba_ssm is required for RegionGraphMamba formal experiments. '
                    'Set allow_gru_fallback=True only for shape smoke tests.'
                )
            # This fallback only keeps shape smoke tests runnable on machines
            # without mamba_ssm. Final experiments should use mamba_ssm.Mamba.
            self.sequence_model = nn.GRU(dim, dim, batch_first=True)
            self.uses_mamba = False

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((1,), init_gamma))

    def _run_sequence_model(self, x):
        # x: [B, K, C]
        if self.uses_mamba:
            return self.sequence_model(x)
        out, _ = self.sequence_model(x)
        return out

    def forward(self, region_tokens, region_xy):
        """Update region tokens through graph-aware multi-path scanning.

        Args:
            region_tokens: Tensor[B, K, C]
            region_xy: Tensor[B, K, 2]

        Returns:
            updated_tokens: Tensor[B, K, C]
        """
        A = build_region_graph(
            region_tokens,
            region_xy,
            k_spatial=self.k_spatial,
            k_feature=self.k_feature,
            alpha=self.alpha,
            beta=self.beta,
        )

        graph_perm = None
        perms = []
        for mode in self.path_modes:
            if mode == 'yx':
                perms.append(argsort_xy(region_xy, mode='yx'))
            elif mode == 'xy':
                perms.append(argsort_xy(region_xy, mode='xy'))
            elif mode == 'graph':
                if graph_perm is None:
                    graph_perm = greedy_graph_path(A, region_xy)
                perms.append(graph_perm)
            elif mode == 'reverse_graph':
                if graph_perm is None:
                    graph_perm = greedy_graph_path(A, region_xy)
                perms.append(torch.flip(graph_perm, dims=[1]))
            else:
                raise ValueError(
                    f"Unknown RegionGraphMamba path mode: {mode}. "
                    f"Supported modes are {sorted(self.VALID_PATH_MODES)}."
                )

        restored_paths = []
        for perm in perms:
            # ordered_tokens: [B, K, C]
            ordered_tokens = batched_gather_tokens(region_tokens, perm)
            ordered_out = self._run_sequence_model(ordered_tokens)
            inverse_perm = invert_permutation(perm)
            restored_paths.append(batched_gather_tokens(ordered_out, inverse_perm))

        # mixed: [B, K, C], updated_tokens: [B, K, C]
        mixed = torch.stack(restored_paths, dim=0).mean(dim=0)
        residual = self.proj(self.norm(mixed))
        updated_tokens = region_tokens + self.gamma * residual
        return updated_tokens
