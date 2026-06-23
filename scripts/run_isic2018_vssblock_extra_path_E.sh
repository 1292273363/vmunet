#!/usr/bin/env bash
set -euo pipefail

# Extra superpixel-path experiments E1/E2.
# Original SS2D raster, transpose, reverse-raster, and reverse-transpose paths
# are preserved; graph paths are gated residual additions rather than replacements.
# No external SP-RGM, region auxiliary loss, or multistage SPScan is enabled.
# With the current bottleneck depth of 2, last2 enables both bottleneck blocks.

python train.py --config config_setting_isic2018_E1_vssblock_extra_graph_bottleneck_last2.py
python train.py --config config_setting_isic2018_E2_vssblock_extra_graph_reverse_bottleneck_last2.py
