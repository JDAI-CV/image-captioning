import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

class LabelSmoothing(nn.Module):
    def __init__(self):
        super(LabelSmoothing, self).__init__()
        self.true_dist = None
        self.smoothing = cfg.LOSSES.LABELSMOOTHING
        self.confidence = 1.0 - self.smoothing
        #self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, logit, target_seq):
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1) 
        target_seq = target_seq.view(-1)
        mask = target_seq >= 0

        assign_seq = target_seq
        assign_seq[assign_seq < 0] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        loss = torch.masked_select(loss, mask).mean()
        return loss, {'LabelSmoothing Loss': loss.item()}

