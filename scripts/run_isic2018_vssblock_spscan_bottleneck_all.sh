#!/usr/bin/env bash
set -euo pipefail

# Bottleneck all-block SPScan extension experiments only.
# These runs do not use the external SP-RGM block.
# These runs do not use region auxiliary or mass-balance losses.
# These runs do not extend SPScan to stage3/stage2/decoder.
# Goal: test whether one-path/two-path replacement remains stable when enabled
# on all VSSBlocks in the bottleneck stage.

python train.py --config config_setting_isic2018_V4_vssblock_spscan_bottleneck_all_one_path.py
python train.py --config config_setting_isic2018_V5_vssblock_spscan_bottleneck_all_two_paths.py
