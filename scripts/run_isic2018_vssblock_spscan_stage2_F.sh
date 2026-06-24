#!/usr/bin/env bash
set -euo pipefail

# Stage2 last-block replacement SPScan experiments F1/F2.
# F1 uses one_path; F2 uses two_paths.
# No external SP-RGM, extra-path residual branch, region auxiliary loss,
# bottleneck SPScan, decoder SPScan, or multi-stage SPScan is enabled.

python train.py --config config_setting_isic2018_F1_vssblock_spscan_stage2_last_one_path.py
python train.py --config config_setting_isic2018_F2_vssblock_spscan_stage2_last_two_paths.py
