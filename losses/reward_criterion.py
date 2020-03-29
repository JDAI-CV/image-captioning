import torch
import torch.nn as nn

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logP, rewards):
        mask = seq > 0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)
        return loss