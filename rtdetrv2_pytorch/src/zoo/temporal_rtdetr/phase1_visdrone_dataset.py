import os
import copy
import random
from collections import defaultdict
import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision import datapoints

from src.core import register
from src.data.transforms import Compose

@register()
class VisDroneTemporalDataset(CocoDetection):
    """
    Temporal Dataset for VisDrone-VID.
    Uses vanilla PyTorch CocoDetection to prevent hidden double-transforms, 
    while manually wrapping targets in tv_tensors for V2 augmentation compatibility.
    """
    __inject__ = ['transforms']

    def __init__(self, root_dir, ann_file, transforms=None, return_masks=False, 
                 remap_mscoco_category=False,
                 pair_sampling_strategy='random', frame_stride=1, max_frame_gap=10):
        
        # Inherit from vanilla PyTorch CocoDetection to stop hidden prepare() calls
        super().__init__(root_dir, ann_file)
        
        self._transforms = transforms
        self.prepare = Compose([]) if transforms is None else transforms
        self.return_masks = return_masks
        
        # Temporal config
        self.pair_sampling_strategy = pair_sampling_strategy
        self.frame_stride = frame_stride
        self.max_frame_gap = max_frame_gap

        # 1. Group images into video sequences
        self.sequences = defaultdict(list)
        print("Grouping VisDrone frames into video sequences...")
        
        for img_id in self.coco.getImgIds():
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            
            # VisDrone-VID layout: 'sequence_name/0000001.jpg'
            seq_name = os.path.dirname(file_name)
            if not seq_name:
                seq_name = file_name.split('_')[0] 
                
            self.sequences[seq_name].append(img_info)

        # 2. Sort frames chronologically within each sequence
        for seq_name in self.sequences:
            self.sequences[seq_name].sort(key=lambda x: x['file_name'])
            
        print(f"Loaded {len(self.sequences)} video sequences.")

        # 3. Build Key / Non-Key Pairs
        self.pairs = []
        self._build_pairs()
        print(f"Generated {len(self.pairs)} temporal pairs using '{self.pair_sampling_strategy}'.")
        
        # Note: self.coco from parent (CocoDetection) uses original 1-indexed categories (1-11)
        # The model outputs 1-indexed labels, so we keep COCO as-is for evaluation

    def _build_pairs(self):
        """Generates (key_id, non_key_id) tuples based on the strategy"""
        for seq_name, frames in self.sequences.items():
            num_frames = len(frames)
            
            if self.pair_sampling_strategy == 'random':
                for i in range(0, num_frames, self.frame_stride):
                    k_info = frames[i]
                    max_idx = min(i + self.max_frame_gap, num_frames - 1)
                    if max_idx > i:
                        nk_idx = random.randint(i + 1, max_idx)
                        nk_info = frames[nk_idx]
                        self.pairs.append((k_info['id'], nk_info['id']))
                        
            elif self.pair_sampling_strategy == 'fixed_gap':
                for i in range(0, num_frames, self.frame_stride):
                    k_info = frames[i]
                    nk_idx = i + self.max_frame_gap
                    if nk_idx < num_frames:
                        nk_info = frames[nk_idx]
                        self.pairs.append((k_info['id'], nk_info['id']))
            
            elif self.pair_sampling_strategy == 'all':
                for i in range(0, num_frames, self.frame_stride):
                    k_info = frames[i]
                    max_offset = min(self.max_frame_gap + 1, num_frames - i)
                    for s in range(1, max_offset):
                        nk_info = frames[i + s]
                        self.pairs.append((k_info['id'], nk_info['id']))

    def load_item(self, img_id):
        """Helper to load a raw image and manually format tv_tensors"""
        img, target = super().__getitem__(self.ids.index(img_id))
        
        w, h = img.size 
        
        image_id = target[0]['image_id'] if len(target) > 0 else img_id
        res_target = {'image_id': torch.tensor([image_id])}
        
        if len(target) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor([obj['bbox'] for obj in target], dtype=torch.float32)
            # torchvision CocoDetection returns category_ids as-is from JSON (1-10 for VisDrone)
            # Keep them unchanged - matches pretrained model output range
            labels = torch.tensor([obj['category_id'] for obj in target], dtype=torch.int64)
            area = torch.tensor([obj['area'] for obj in target], dtype=torch.float32)
            iscrowd = torch.tensor([obj.get('iscrowd', 0) for obj in target], dtype=torch.int64)
            
        # Manually wrap boxes to satisfy strict V2 Transforms
        res_target['boxes'] = datapoints.BoundingBox(
            boxes, 
            format="XYWH", 
            spatial_size=(h, w)
        )
        res_target['labels'] = labels
        res_target['area'] = area
        res_target['iscrowd'] = iscrowd
        # Note: orig_size should be [width, height] to match postprocessor expectations
        res_target['orig_size'] = torch.tensor([w, h])
        res_target['size'] = torch.tensor([h, w])
            
        return img, res_target

    def _apply_shared_pair_transforms(self, img_k, target_k, img_nk, target_nk):
        # Reuse identical RNG states so key/non-key sample identical transform params.
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        np_state = np.random.get_state()
        global_samples = getattr(self.prepare, 'global_samples', None)

        img_k, target_k = self.prepare(img_k, target_k)

        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        if global_samples is not None and hasattr(self.prepare, 'global_samples'):
            self.prepare.global_samples = global_samples

        img_nk, target_nk = self.prepare(img_nk, target_nk)
        if global_samples is not None and hasattr(self.prepare, 'global_samples'):
            self.prepare.global_samples = global_samples + 1

        return img_k, target_k, img_nk, target_nk

    def __getitem__(self, idx):
        """Returns the Key and Non-Key tuple ready for the temporal network"""
        key_id, non_key_id = self.pairs[idx]
        
        # 1. Load raw images and V2-wrapped targets
        img_k, target_k = self.load_item(key_id)
        img_nk, target_nk = self.load_item(non_key_id)
        
        # 2. Apply transforms EXACTLY ONCE
        if self._transforms is not None:
            img_k, target_k, img_nk, target_nk = self._apply_shared_pair_transforms(
                img_k,
                target_k,
                img_nk,
                target_nk,
            )
            
        return img_k, target_k, img_nk, target_nk

    def __len__(self):
        return len(self.pairs)
