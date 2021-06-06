import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
from datasets.radiology_dataset import IUXRAY
import samplers.distributed
import numpy as np
import argparse
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument('--image_dir', type=str, default='/content/iu_xray_resized/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/content/iu_xray_resized/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--dataset_name', type=str, default='IUXRAY', choices=['IUXRAY', 'MIMICCXR'],
                        help='the dataset to be used.')
    parser.add_argument('--submodel', type=str, default='RGMG', choices=['RGMG', 'VSEGCN'],
                        help='the knowledge graph to be used.')
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



args = parse_args()

if args.submodel =='RGMG' and args.dataset_name =='IUXRAY':
  mask_dim = 70
elif args.submodel == 'VSEGCN' and args.dataset_name =='IUXRAY':
  mask_dim = 74
elif args.submodel == 'VSEGCN' and args.dataset_name =='MIMICCXR':
  mask_dim = 86


def sample_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)

    max_seq_length = max([len(x) for x in input_seq])
    input_seqs = np.zeros((len(input_seq), max_seq_length), dtype=int)
    target_seqs = np.zeros((len(target_seq), max_seq_length), dtype=int)
    # print(max_seq_length)

    for i, input in enumerate(input_seq):
        input_seqs[i, :len(input)] = input

    for i, target in enumerate(target_seq):
        target_seqs[i, :len(target)] = target

    indices = np.stack(indices, axis=0).reshape(-1)

    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    # IT DOESNT MATTER WHAT THE SHAPE OF MASK IS, WE WILL GET THE MASK LATER
    mask_arr = torch.ones([len(att_feats), mask_dim]).float()

    att_mask = torch.cat([mask_arr], 0)
    att_feats = torch.stack(att_feats, 0) # TODO for mimic
    """
    indices (40, )
    input_seq (40, 60)
    target_seq (40, 60)
    gv_feat (40, 1)
    att_feats (40, 2, 3, 224, 224) 
    att_mask (40, 1, 3) => (40, 49)
    """
    return indices, torch.LongTensor(input_seqs), torch.LongTensor(target_seqs), gv_feat, att_feats, att_mask

def sample_collate_val(batch):

    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)

    max_seq_length = max([len(x) for x in input_seq])
    input_seqs = np.zeros((len(input_seq), max_seq_length), dtype=int)
    target_seqs = np.zeros((len(target_seq), max_seq_length), dtype=int)
    # print(max_seq_length)

    for i, input in enumerate(input_seq):
        input_seqs[i, :len(input)] = input

    for i, target in enumerate(target_seq):
        target_seqs[i, :len(target)] = target

    indices = np.stack(indices, axis=0).reshape(-1)

    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    # IT DOESNT MATTER WHAT THE SHAPE OF MASK IS, WE WILL GET THE MASK LATER
    mask_arr = torch.ones([len(att_feats), mask_dim]).float()

    att_mask = torch.cat([mask_arr], 0)
    att_feats = torch.stack(att_feats, 0)
    """
    indices (40, )
    input_seq (40, 60)
    target_seq (40, 60)
    gv_feat (40, 1)
    att_feats (40, 2, 3, 224, 224) 
    att_mask (40, 1, 3) => (40, 49)
    """
    return indices, torch.LongTensor(target_seqs), gv_feat, att_feats, att_mask


def load_train(distributed, epoch, dataset):
    sampler = samplers.distributed.DistributedSampler(dataset, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle,
        num_workers = cfg.DATA_LOADER.NUM_WORKERS,
        drop_last = cfg.DATA_LOADER.DROP_LAST,
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler,
        collate_fn = sample_collate
    )
    return loader

def load_val(dataset):
    # coco_set = CocoDataset(
    #     image_ids_path = image_ids_path,
    #     input_seq = None,
    #     target_seq = None,
    #     gv_feat_path = gv_feat_path,
    #     att_feats_folder = att_feats_folder,
    #     seq_per_img = 1,
    #     max_feat_num = cfg.DATA_LOADER.MAX_FEAT
    # )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False,
        num_workers = cfg.DATA_LOADER.NUM_WORKERS,
        drop_last = False,
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn = sample_collate_val
    )
    return loader
