"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from typing import List

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


class StridedDataset:
    """Dataset view that keeps every Nth sample and preserves set_epoch/load_item hooks."""

    def __init__(self, dataset, stride: int, split_name: str):
        if stride < 1:
            raise ValueError(f"{split_name} stride must be >= 1, got {stride}")
        self.dataset = dataset
        self.stride = int(stride)
        self.split_name = split_name
        self.indices: List[int] = list(range(0, len(dataset), self.stride))
        self.sampled_img_ids = self._extract_sampled_img_ids()

    def _extract_sampled_img_ids(self):
        if hasattr(self.dataset, 'ids'):
            return [int(self.dataset.ids[i]) for i in self.indices]
        if hasattr(self.dataset, 'image_list'):
            return [int(self.dataset.image_list[i]) for i in self.indices]
        return None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def load_item(self, idx):
        if not hasattr(self.dataset, 'load_item'):
            raise AttributeError(f"{self.dataset.__class__.__name__} does not implement load_item")
        return self.dataset.load_item(self.indices[idx])

    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    @property
    def epoch(self):
        return getattr(self.dataset, 'epoch', -1)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _rebuild_loader_with_stride(loader, stride: int, split_name: str):
    if stride == 1:
        return loader

    base_dataset = loader.dataset
    strided_dataset = StridedDataset(base_dataset, stride=stride, split_name=split_name)
    shuffle = getattr(loader, 'shuffle', False)
    print(
        f"Applying {split_name} stride={stride}: "
        f"{len(base_dataset)} -> {len(strided_dataset)} samples"
    )

    loader_kwargs = {
        'dataset': strided_dataset,
        'batch_size': loader.batch_size,
        'shuffle': shuffle,
        'num_workers': loader.num_workers,
        'collate_fn': loader.collate_fn,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
    }

    if hasattr(loader, 'persistent_workers'):
        loader_kwargs['persistent_workers'] = loader.persistent_workers
    if hasattr(loader, 'prefetch_factor') and loader.num_workers > 0 and loader.prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = loader.prefetch_factor
    if hasattr(loader, 'timeout'):
        loader_kwargs['timeout'] = loader.timeout
    if hasattr(loader, 'worker_init_fn') and loader.worker_init_fn is not None:
        loader_kwargs['worker_init_fn'] = loader.worker_init_fn

    new_loader = loader.__class__(**loader_kwargs)
    new_loader.shuffle = shuffle
    return new_loader


def _set_cfg_loader(cfg, split_name: str, loader):
    """Set dataloader on YAMLConfig even when property setter is unavailable."""
    prop_name = f'{split_name}_dataloader'
    private_name = f'_{split_name}_dataloader'
    try:
        setattr(cfg, prop_name, loader)
    except AttributeError:
        setattr(cfg, private_name, loader)


def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Invalid positive int value: {value}")
    return ivalue


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', 'train_stride', 'val_stride'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    if not args.test_only and args.train_stride > 1:
        train_loader = _rebuild_loader_with_stride(
            cfg.train_dataloader, stride=args.train_stride, split_name='train')
        _set_cfg_loader(cfg, 'train', train_loader)

    if args.val_stride > 1:
        val_loader = _rebuild_loader_with_stride(
            cfg.val_dataloader, stride=args.val_stride, split_name='val')
        _set_cfg_loader(cfg, 'val', val_loader)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    parser.add_argument('--train-stride', type=positive_int, default=1,
                        help='use every Nth sample for train dataloader (default: 1, no striding)')
    parser.add_argument('--val-stride', type=positive_int, default=1,
                        help='use every Nth sample for val dataloader (default: 1, no striding)')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
