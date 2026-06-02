import torch
from torch.utils.data.distributed import DistributedSampler

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .dl3dv import DL3DV_Multi
from .vkitti2 import VirtualKITTI2_Multi
from .hypersim import HyperSim_Multi
from .tartanair import TartanAir_Multi
from .unreal4k import UnReal4K_Multi
from .pointodyssey import PointOdyssey_Multi
from .project_aria_seq import Aria_Seq
from .layereddepth import LayeredDepth_Multi
from .hsod_glass import GlassSegmentationDataset
from .ade20k_glass_dataset import ADE20KGlassDataset
from .trans10k_glass import Trans10KGlassDataset
from .dynamic_replica import DynamicReplica
from .omniworld_game import OmniWorldGame_Multi
from .spring import Spring


from src.dust3r.datasets_cut3r.utils.misc import get_world_size, get_rank
from src.dust3r.datasets_cut3r.base.ratio_loader import RatioDataLoader
from src.dust3r.datasets_cut3r.base.custom_collate import collate_like_default


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    multiprocessing_context=None,
    max_num_of_images_per_gpu=16,
    use_dynamic_sampler=True,
    max_sequence_length=16,
    min_view_size=4,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()
    
    # try:
    if True:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            max_num_of_images_per_gpu=max_num_of_images_per_gpu,
            use_dynamic_sampler=use_dynamic_sampler,
            max_sequence_length=max_sequence_length,
            min_view_size=min_view_size,
        )
        batch_sampler = sampler
        sampler = None
    else:
    # except Exception:
        # not avail for this dataset
        batch_sampler = None
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    if batch_sampler is not None:
        # breakpoint()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            # collate_fn=collate_like_default,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            # collate_fn=collate_like_default,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

    return data_loader




def get_data_loader_2dataloader(
    dataset1,
    dataset2,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    multiprocessing_context=None,
    max_num_of_images_per_gpu=16,
    use_dynamic_sampler=True,
    max_sequence_length=16,
    min_view_size=4,
):
    import torch

    # pytorch dataset
    if isinstance(dataset1, str):
        dataset1 = eval(dataset1)
        
    if isinstance(dataset2, str):
        dataset2 = eval(dataset2)

    world_size = get_world_size()
    rank = get_rank()
    # print('world_size: ', world_size, 'rank: ', rank)
    # assert False
    
    if True:
        sampler1 = dataset1.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            max_num_of_images_per_gpu=max_num_of_images_per_gpu,
            use_dynamic_sampler=use_dynamic_sampler,
            max_sequence_length=max_sequence_length,
            min_view_size=min_view_size,
        )
        
        sampler2 = dataset2.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            use_dynamic_sampler=False,
            min_view_size=min_view_size,
        )   
        batch_sampler1 = sampler1
        batch_sampler2 = sampler2
        sampler1 = None
        sampler2 = None

    data_loader = RatioDataLoader(
        dataset1,
        dataset2,
        batch_sampler1, batch_sampler2, 
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent_workers,
        multiprocessing_context=multiprocessing_context,
    )

    return data_loader

