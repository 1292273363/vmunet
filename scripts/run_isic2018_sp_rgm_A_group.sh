#!/usr/bin/env bash
set -e

# ISIC2018 SP-RGM A-group only.
# A0/A1 have already been completed in the current first-stage run; skip them
# when you only need the remaining A2/A3/A4 experiments.
# Current stage intentionally does not run B/C ablations.
#
# A2 is the key next experiment: it tests whether auxiliary region losses reduce
# the false-positive increase observed in A1.
# A3/A4 compare single graph path vs. multi-path graph-guided Mamba.

python train.py --config config_setting_isic2018_A0_vmunet_baseline.py
python train.py --config config_setting_isic2018_A1_sp_rgm_module_only_4x4.py
python train.py --config config_setting_isic2018_A2_sp_rgm_full_loss_4x4.py
python train.py --config config_setting_isic2018_A3_sp_rgm_single_path_4x4.py
python train.py --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py

# Optional checkpoint-specific final tests:
# python train.py --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py --test-checkpoint-type best_dice
# python train.py --config config_setting_isic2018_A4_sp_rgm_multi_path_4x4.py --test-checkpoint-type best_iou
