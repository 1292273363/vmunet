from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 use_sp_rgm=False,
                 sp_rgm_cfg=None,
                 use_sp_scan=False,
                 sp_scan_cfg=None,
                 sp_scan_stage=None,
                 sp_scan_blocks=None,
                 use_ssr=False,
                 ssr_cfg=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.input_channels = input_channels
        if sum(bool(flag) for flag in (use_sp_rgm, use_sp_scan, use_ssr)) > 1:
            raise ValueError("VMUNet does not allow use_sp_rgm, use_sp_scan, and use_ssr at the same time.")

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           use_sp_rgm=use_sp_rgm,
                           sp_rgm_cfg=sp_rgm_cfg,
                           use_sp_scan=use_sp_scan,
                           sp_scan_cfg=sp_scan_cfg,
                           sp_scan_stage=sp_scan_stage,
                           sp_scan_blocks=sp_scan_blocks,
                           use_ssr=use_ssr,
                           ssr_cfg=ssr_cfg,
                        )
    
    def forward(self, x, return_aux=False):
        actual_channels = x.size(1)
        if self.input_channels == 3 and actual_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif actual_channels != self.input_channels:
            raise ValueError(
                f"VMUNet expected input_channels={self.input_channels}, "
                f"but got input with {actual_channels} channels."
            )
        model_out = self.vmunet(x, return_aux=return_aux)

        if return_aux:
            logits, aux = model_out
        else:
            logits = model_out

        if self.num_classes == 1:
            # Disable this sigmoid if training with BCEWithLogitsLoss.
            logits = torch.sigmoid(logits)

        if return_aux:
            return logits, aux
        return logits

    def get_sp_scan_stats(self):
        return self.vmunet.get_sp_scan_stats()

    def get_ssr_stats(self):
        return self.vmunet.get_ssr_stats()

    @staticmethod
    def _extract_checkpoint_state(checkpoint):
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            if 'model' in checkpoint:
                return checkpoint['model']
        return checkpoint

    def _safe_load_state_dict(self, pretrained_dict, label):
        model_dict = self.vmunet.state_dict()
        loadable_dict = {}
        unexpected_keys = []
        shape_mismatch = []

        for key, value in pretrained_dict.items():
            if key not in model_dict:
                unexpected_keys.append(key)
                continue
            if model_dict[key].shape != value.shape:
                shape_mismatch.append((key, tuple(value.shape), tuple(model_dict[key].shape)))
                continue
            loadable_dict[key] = value

        load_result = self.vmunet.load_state_dict(loadable_dict, strict=False)
        print(
            f"{label}: loaded={len(loadable_dict)}, missing={len(load_result.missing_keys)}, "
            f"unexpected={len(unexpected_keys)}, shape_mismatch={len(shape_mismatch)}"
        )
        if load_result.missing_keys:
            print(f"{label} missing key examples: {load_result.missing_keys[:20]}")
        if unexpected_keys:
            print(f"{label} unexpected key examples: {unexpected_keys[:20]}")
        if shape_mismatch:
            print(f"{label} shape mismatch examples: {shape_mismatch[:10]}")

    def load_from(self):
        if self.load_ckpt_path is not None:
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            pretrained_odict = self._extract_checkpoint_state(modelCheckpoint)
            self._safe_load_state_dict(pretrained_odict, 'encoder pretrained')
            print("encoder loaded finished!")

            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            self._safe_load_state_dict(pretrained_dict, 'decoder pretrained')
            print("decoder loaded finished!")
