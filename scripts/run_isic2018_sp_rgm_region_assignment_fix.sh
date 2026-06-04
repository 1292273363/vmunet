#!/usr/bin/env bash
set -euo pipefail

# Region assignment repair experiments for ISIC2018 SP-RGM.
# R1: test whether reducing bottleneck regions to 2x2 lowers empty_region_ratio.
# R2: test whether region mass balance loss reduces 4x4 region collapse.
# R3: test whether 2x2 regions plus mass balance further stabilizes assignment.
# R4: optional, test stronger spatial constraint with xy_weight=4.0.

python train.py --config config_setting_isic2018_R1_sp_rgm_regions_2x2_tau0p2_lite_loss.py
python train.py --config config_setting_isic2018_R2_sp_rgm_4x4_tau0p2_balance.py
python train.py --config config_setting_isic2018_R3_sp_rgm_2x2_tau0p2_balance.py

# Optional: run only if R2 still shows high empty_region_ratio / region_mass_cv.
# python train.py --config config_setting_isic2018_R4_sp_rgm_4x4_tau0p2_xy4_balance.py
