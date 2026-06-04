#!/usr/bin/env bash
set -e

# ISIC2018 SP-RGM A2 revised experiments only.
#
# A2R1: tests whether reducing region reconstruction loss avoids the original
# A2 under-segmentation / sensitivity drop.
# A2R2: tests whether tau=0.2 softens Q and reduces hard assignment / empty regions.
# A2R3: tests lightweight compactness only after A2R2 improves Q diagnostics.
# If A2R2 is still ineffective, pause A2R3 and inspect threshold sweep first.

python train.py --config config_setting_isic2018_A2R1_sp_rgm_lite_region_loss_4x4.py
python train.py --config config_setting_isic2018_A2R2_sp_rgm_lite_region_loss_tau0p2_4x4.py
python train.py --config config_setting_isic2018_A2R3_sp_rgm_lite_full_loss_tau0p2_4x4.py
