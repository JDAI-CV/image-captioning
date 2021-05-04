#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import torch
from torch import relu
from torch.nn import Dropout, Linear, Sequential
from torch.nn.functional import adaptive_avg_pool2d, cross_entropy
from torchvision import models


# # Image processes
# if image_model is None:
#     image_model = 'densenet'
# self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
#                                                                  image_pretrained, device)

class ImageClassification(torch.nn.Module):
    def __init__(self, model, num_labels, num_classes, multi_image=1, dropout=0.0, pretrained=True):
        super(ImageClassification, self).__init__()
        self.image_feats, self.image_dim = self.image_features(model, False, pretrained)
        for i in range(num_labels):
            setattr(self, 'linear{0}'.format(i), Linear(self.image_dim, num_classes))
        self.num_labels = num_labels
        self.multi_image = multi_image
        self.dropout = Dropout(p=dropout)

    @classmethod
    def fix_layers(cls, model):
        for param in model.parameters():
            param.requires_grad = False

    @classmethod
    def image_features(cls, name, fixed_weight=False, pretrained=True, pretrained_model=None, device='gpu'):
        if pretrained_model is None:
            if name == 'densenet121' or name == 'densenet':
                m = models.densenet121(pretrained=pretrained)
                if fixed_weight:
                    cls.fix_layers(m)
                return Sequential(*list(m.features.children())), 1024
            elif name == 'resnet50':
                m = models.resnet50(pretrained=pretrained)
                if fixed_weight:
                    cls.fix_layers(m)
                return Sequential(*list(m.children())[:-2]), 2048
            elif name == 'resnet152' or name == 'resnet':
                m = models.resnet152(pretrained=pretrained)
                if fixed_weight:
                    cls.fix_layers(m)
                return Sequential(*list(m.children())[:-2]), 2048
            elif name == 'vgg19' or name == 'vgg':
                m = models.vgg19(pretrained=pretrained)
                if fixed_weight:
                    cls.fix_layers(m)
                return Sequential(*list(m.features.children())[:-1]), 512
            else:
                raise ValueError('Unknown model {0}'.format(name))
        else:
            d = torch.device('cpu')if device == 'cpu' else torch.device('cuda:0')
            with gzip.open(pretrained_model) as f:
                state = torch.load(f, map_location=d)
            m = ImageClassification(name, 14, 3, pretrained=False)
            m.load_state_dict(state['model'])
            if fixed_weight:
                cls.fix_layers(m)
            return m.image_feats, m.image_dim

    def deflatten_image(self, x):
        if self.multi_image > 1:
            x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, x.shape[1])
            x, _ = torch.max(x, dim=1)
        return x

    def flatten_image(self, x):
        if self.multi_image > 1:
            return x.flatten(start_dim=0, end_dim=1)
        else:
            return x

    def forward(self, x):
        x = self.flatten_image(x)
        x = self.image_feats(x)
        x = relu(x)
        x = adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.deflatten_image(x)
        xs = []
        for i in range(self.num_labels):
            xi = self.dropout(x)
            xi = getattr(self, 'linear{0}'.format(i))(xi).unsqueeze(dim=2)
            xs.append(xi)
        x = torch.cat(xs, dim=2)
        return x