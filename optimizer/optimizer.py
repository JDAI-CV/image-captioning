import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW

class Optimizer(nn.Module):
    def __init__(self, model):
        super(Optimizer, self).__init__()
        self.setup_optimizer(model)

    def setup_optimizer(self, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR 
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                momentum = cfg.SOLVER.SGD.MOMENTUM
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            self.optimizer = torch.optim.Adam(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        else:
            raise NotImplementedError

        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = cfg.SOLVER.LR_POLICY.STEP_SIZE, 
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,  
                factor = cfg.SOLVER.LR_POLICY.PLATEAU_FACTOR, 
                patience = cfg.SOLVER.LR_POLICY.PLATEAU_PATIENCE
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = cfg.SOLVER.LR_POLICY.STEPS,
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if cfg.SOLVER.LR_POLICY.TYPE != 'Plateau':
            val = None

        if lrs_type == cfg.SOLVER.LR_POLICY.SETP_TYPE:
            self.scheduler.step(val)

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr
