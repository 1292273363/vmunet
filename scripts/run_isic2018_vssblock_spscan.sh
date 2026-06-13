#!/usr/bin/env bash
set -euo pipefail

# VSSBlock-internal SPScan experiments only.
# These runs do not use the external SP-RGM block, do not add multiscale modules,
# and do not change the VM-UNet backbone structure.
# V1 replaces one SS2D path; V2 replaces two paths.

python train.py --config config_setting_isic2018_V1_vssblock_spscan_one_path.py
python train.py --config config_setting_isic2018_V2_vssblock_spscan_two_paths.py

# Optional spatial-only variant:
# python train.py --config config_setting_isic2018_V3_vssblock_spscan_two_paths_spatial.py
