import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
from datasets.radiology_dataset import IUXRAY
from datasets.tokenizers import Tokenizer
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_network()
        self.val_evaler = Evaler(
            IUXRAY(
                image_dir=args.image_dir,
                ann_path=args.ann_path,
                tokenizer=self.tokenizer,
                split='val'
            ),
            tokenizer = self.tokenizer
        ) # TODO
        self.test_evaler = Evaler(
            IUXRAY(
                image_dir=args.image_dir,
                ann_path=args.ann_path,
                tokenizer=self.tokenizer,
                split='test'
            ),
            tokenizer = self.tokenizer
        ) # TODO
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            # self.model = torch.nn.DataParallel(model).cuda() # strange
            self.model = model.cuda()  # strange

        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

        self.optim = Optimizer(self.model)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()
        
    def setup_dataset(self, dataset = 'IUXRAY'):
        self.tokenizer = Tokenizer(ann_path = args.ann_path,dataset_name = args.dataset_name)
        if dataset == 'IUXRAY':
            self.dataset = IUXRAY(
                image_dir = args.image_dir,
                ann_path = args.ann_path,
                tokenizer = self.tokenizer,
                split = 'train'
            )
        # self.coco_set = datasets.coco_dataset.CocoDataset(
        #     image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
        #     input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
        #     target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
        #     gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT,
        #     att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
        #     seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
        #     max_feat_num = cfg.DATA_LOADER.MAX_FEAT
        # )


    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.dataset)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
            
        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        test_res = self.test_evaler(self.model,'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def forward(self, kwargs):
        if self.rl_stage == False:
            logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
        else:
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            att_feats = kwargs[cfg.PARAM.ATT_FEATS]
            att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
            target_seq = kwargs[cfg.PARAM.TARGET_SENT]

            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(target_seq, seq_max.data.cpu().numpy().tolist()) # Modified
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids) # to check?
            gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            rewards_sample, rewards_info_sample = self.scorer(target_seq, seq_sample.data.cpu().numpy().tolist()) # Modified

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info

    def train(self):
        self.model.train()
        self.optim.zero_grad()

        iteration = 0
        for epoch in  range(cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(self.training_loader):
                data_time.update(time.time() - start)

                input_seq = input_seq.cuda()
                target_seq = target_seq.cuda()
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()

                kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
                loss, loss_info = self.forward(kwargs)
                loss.backward()
                utils.clip_gradient(self.optim.optimizer, self.model,
                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                self.optim.step()
                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')
                
                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                self.display(iteration, data_time, batch_time, losses, loss_info)
                iteration += 1

                if self.distributed:
                    dist.barrier()
        
            self.save_model(epoch)
            val = self.eval(epoch)
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument('--image_dir', type=str, default='/content/iu_xray_resized/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/content/iu_xray_resized/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--dataset_name', type=str, default='iuxray', choices=['iuxray', 'mimiccxr'], help='the dataset to be used.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()
