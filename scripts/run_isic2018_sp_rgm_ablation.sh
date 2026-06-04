#!/usr/bin/env bash
set -euo pipefail

# `train.py` now accepts `--config` and resolves filenames under `configs/`.
# The diagnostic tools accept the same filename style for consistency.

# 1. One-batch overfit sanity check
python tools/overfit_sp_rgm_one_batch.py \
  --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py \
  --iters 300

# 2. Visualization
python tools/visualize_sp_rgm.py \
  --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py \
  --save-path-order

# 3. First-round formal experiments
python train.py --config config_setting_isic2018_A0_vmunet_baseline.py
python train.py --config config_setting_isic2018_A1_sp_rgm_module_only_4x4.py
python train.py --config config_setting_isic2018_A2_sp_rgm_full_4x4.py

# A3 is tracked as a requested experiment config, but the current core
# RegionGraphMamba implementation is fixed at 4 paths and does not consume
# `num_paths`; run this only after true single-path support is implemented.
# python train.py --config config_setting_isic2018_A3_sp_rgm_single_path_4x4.py

python train.py --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py

# 4. Region-count ablations
python train.py --config config_setting_isic2018_B1_sp_rgm_regions_4x4.py
python train.py --config config_setting_isic2018_B2_sp_rgm_regions_6x6.py
python train.py --config config_setting_isic2018_B3_sp_rgm_regions_8x8.py

# 5. Optional stability ablations
python train.py --config config_setting_isic2018_C1_sp_rgm_low_region_loss_4x4.py
python train.py --config config_setting_isic2018_C2_sp_rgm_gamma_0p01_4x4.py
python train.py --config config_setting_isic2018_C3_sp_rgm_tau_0p2_4x4.py
python train.py --config config_setting_isic2018_C4_sp_rgm_tau_0p05_4x4.py
