#!/usr/bin/env bash
set -euo pipefail

# Bottleneck-level S1b follow-up experiments only.
# This script does not include multi-scale SP-RGM.
# This script does not include stage3/stage2/decoder SP-RGM.
# This script does not retrain the VM-UNet baseline.

python train.py --config config_setting_isic2018_S1b_no_region_loss.py
python train.py --config config_setting_isic2018_S2b_assign_norm_balance.py
python train.py --config config_setting_isic2018_S1b_graph_path_only.py
python train.py --config config_setting_isic2018_S1b_yx_graph_paths.py
