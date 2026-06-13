# Superpixel-Guided Scan Order in VSSBlock

## Related Methods

VMamba introduces the VSS block with SS2D as its core visual state-space operator. SS2D bridges the gap between 2D image grids and 1D selective scan by unfolding a feature map into four traversal paths, applying selective scan to the paths, then merging the restored outputs back to the 2D feature layout. In this VM-UNet codebase, `VSSBlock.forward` applies `SS2D` after `LayerNorm`, and `SS2D.forward_corev0` builds the four paths as raster, transposed raster, reverse raster, and reverse transposed raster.

Recent visual Mamba work repeatedly shows that scan order is not a cosmetic detail. Because selective scan is sequence-order dependent, a 2D feature map must be linearized carefully; many variants explore alternative, windowed, multi-directional, topology-aware, or graph-guided scan orders. This makes scan order a natural place to inject structure without changing the surrounding VM-UNet architecture.

Superpixel Sampling Networks (SSN) make SLIC-style superpixels differentiable through soft pixel-to-superpixel assignment and iterative center updates. Our existing `SoftSLIC`, `region_graph`, and `graph_ordering` modules already implement the useful pieces in pure PyTorch: soft assignment `Q`, region token pooling, region coordinates, region graph construction, and greedy graph traversal.

References used for this design:

- VMamba paper: https://arxiv.org/html/2401.10166v4
- VMamba implementation: https://github.com/MzeroMiko/VMamba
- Superpixel Sampling Networks paper: https://arxiv.org/abs/1807.10174
- SSN project page: https://varunjampani.github.io/ssn/
- Visual Mamba survey: https://www.mdpi.com/2076-3417/14/13/5683

## Why Move From External SP-RGM to VSSBlock Scan Order

The bottleneck SP-RGM block was useful as a first probe: it showed that SoftSLIC assignment and graph-guided region paths can affect lesion recall and segmentation behavior. However, as an external bottleneck enhancement, it does not change the VMamba mechanism that processes the main feature hierarchy. The network still mostly relies on fixed raster-style SS2D scan paths.

The next minimal step is therefore not another external branch, but a structure-aware scan order inside SS2D. The new SPScan path keeps all dense tokens, does not pool the Mamba input down to region tokens, and only reorders dense tokens according to a superpixel graph path. This keeps the claim precise: Mamba still scans `H*W` dense tokens, but one or two scan paths become superpixel-guided.

## Why Keep K=4 and Parameter Shapes

The current SS2D parameters are organized around four scan paths:

- `x_proj_weight`: `[K=4, C_proj, D_inner]`
- `dt_projs_weight`: `[K=4, D_inner, dt_rank]`
- `A_logs` and `Ds`: flattened copies for four paths

Keeping `K=4` preserves parameter shapes and keeps pretrained VMamba weights compatible. The implementation changes only the order of tokens entering selected paths and the inverse permutation used to restore their outputs. It does not add learnable routing parameters and does not change the ModuleList indices of VSS layers.

## Why Replace Only One or Two Paths

Replacing all four paths would erase too much of the pretrained inductive bias at once. The stable first version keeps at least raster and transposed raster paths unchanged:

- `one_path`: raster, transpose, reverse raster, graph
- `two_paths`: raster, transpose, graph, reverse graph

This gives a conservative ablation ladder. If one path is stable but two paths drops, the graph ordering is probably useful but too strong. If two paths improves, the superpixel graph order is helping beyond a small perturbation.

## Risks and Engineering Protections

- Ordering is discrete. The default `detach_order=True` computes SoftSLIC labels, graph order, and argsort under `torch.no_grad()`. The token gather still preserves gradients from selective scan output back to the original dense features.
- Spatial alignment can break silently. Every graph-path output is inverse-permuted back to raster order before path merging.
- Pretrained loading can be fragile. The load path should accept only keys with matching shapes and use `strict=False`, printing skipped and missing keys.
- Experiment mixing is easy. `use_sp_rgm` and `use_sp_scan` must not be enabled together.
- Fallback behavior should be explicit. Formal SPScan experiments use the VMamba selective scan implementation; no GRU fallback is introduced here.
- Scope stays narrow. The first implementation supports only `sp_scan_stage="bottleneck_last"` so the deepest encoder VSSBlock can be tested without broad architectural changes.
