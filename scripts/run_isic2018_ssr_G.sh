#!/usr/bin/env bash
set -euo pipefail

# G1/G2: Superpixel Skip Refinement experiments.
# These runs do not enable SPScan, do not enable the external SP-RGM block,
# do not change VSSBlock scan order, and do not use region auxiliary losses.
# They only test SuiT-style avg+max superpixel aggregation on decoder skips.

python train.py --config config_setting_isic2018_G1_ssr_stage2_skip_avgmax.py
python train.py --config config_setting_isic2018_G2_ssr_stage1_skip_avgmax.py
