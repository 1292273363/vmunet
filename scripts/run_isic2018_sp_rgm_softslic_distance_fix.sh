#!/usr/bin/env bash
set -euo pipefail

# SoftSLIC assignment distance fix experiments for ISIC2018 SP-RGM.
# S1 is the key experiment in this round: assignment normalization + feat_weight=0.2.
# If S1 still has q_max_mean ~= 1 and Q_entropy ~= 0, pause S2 and try S1b.
# S2 only makes sense after S1 confirms that Q becomes softer.
# This round does not run A3/A4 path ablations.

python train.py --config config_setting_isic2018_S1_sp_rgm_2x2_assign_norm.py
python train.py --config config_setting_isic2018_S2_sp_rgm_2x2_assign_norm_balance.py

# Optional: lower assignment feature weight if S1 remains too hard.
# python train.py --config config_setting_isic2018_S1b_sp_rgm_2x2_assign_norm_feat0p1.py
