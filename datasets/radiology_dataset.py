import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image
from .tokenizers import Tokenizer


class BaseDataset(Dataset):
    def __init__(self, image_dir, ann_path, tokenizer, split):
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_seq_length = 60
        self.split = split
        self.tokenizer = tokenizer
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.texts = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.texts[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = self.tokenizer(self.examples[i]['report'])[:self.max_seq_length]

    def __len__(self):
        return len(self.examples)


class IUXRAY(BaseDataset):
    def __getitem__(self, idx):
        # indices = np.array([idx]).astype('int') # Modified
        image_id = self.examples[idx]['id']
        indices = np.array([image_id])
        example = self.examples[idx]
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)


        report_ids = np.array(example['ids'])

        input_sequence = np.zeros(self.max_seq_length, dtype='int')
        target_sequence = np.zeros(self.max_seq_length, dtype='int')

        input_sequence[:len(report_ids)] = report_ids
        target_sequence[:len(report_ids)-1] = report_ids[1:]

        gv_feat = np.zeros((1, 1)) # Never been used
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        return indices, input_sequence, target_sequence, gv_feat, image


class MIMICCXR(BaseDataset): # MimiccxrSingleImageDataset
    def __getitem__(self, idx):
        # indices = np.array([idx]).astype('int') # Modified
        image_id = self.examples[idx]['id']
        indices = np.array([image_id])
        example = self.examples[idx]
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = np.array(example['ids'])

        input_sequence = np.ones(self.max_seq_length, dtype='int')
        target_sequence = np.ones(self.max_seq_length, dtype='int')

        input_sequence[:len(report_ids)] = report_ids
        target_sequence[:len(report_ids) - 1] = report_ids[1:]

        gv_feat = np.zeros((1, 1))  # Never been used
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        return indices, input_sequence, target_sequence, gv_feat, image
