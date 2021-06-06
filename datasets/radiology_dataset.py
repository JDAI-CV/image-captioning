import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image
from .tokenizers import Tokenizer
import random
import time
import copy

def random_position(image_1, image_2, thr):
    if random.random() < thr:
        img = torch.stack((image_1, image_2), 0)
    else:
        img = torch.stack((image_2, image_1), 0)
    return img



class BaseDataset(Dataset):
    def __init__(self, image_dir, ann_path, tokenizer, split, args):
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_seq_length = 60 # hardcode
        self.split = split
        self.args = args
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
        if args.dataset_name == 'MIMICCXR_MultiImages':
            self.examples = self.convert_to_multi_images(self.examples)
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = self.tokenizer(self.examples[i]['report'])[:self.max_seq_length]

    def convert_to_multi_images(self, dataset, print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('{} set: Converting to multiple image reports ... '.format(self.split), end='', flush=True)
        mergedDataset = []
        total = len(dataset)

        buffer = None
        for i in range(total):
            document = dataset[i]
            id = document['id']
            image_path = document['image_path'][0]
            # report = document['report']
            # split = document['split']
            study_id = document['study_id']
            # subject_id = document['subject_id']

            if study_id == buffer:
                mergedDataset[-1]['image_path'].append(image_path)
                mergedDataset[-1]['id'].append(id)
            else:
                newDocument = copy.deepcopy(document)
                newDocument['id'] = [newDocument['id']]
                mergedDataset.append(newDocument)
                n += 1
            buffer = study_id
        if print_num:
            print('done %d->%d (%.2fs)' % (total, n, time.time() - t), flush=True)
        return mergedDataset

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
        if self.split  == 'train':
            image = random_position(image_1, image_2, 0.5)
        else:
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

class MimiccxrMultiImage(BaseDataset): # MimiccxrMultiImageDataset
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

class MimiccxrMultiImage(BaseDataset):
    def __getitem__(self, idx):
        image_id = str(self.examples[idx]['subject_id']) + '_' + str(self.examples[idx]['study_id'])
        indices = np.array([image_id])
        example = self.examples[idx]
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        try:
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        # if this record only have one image, duplicate image1
        except:
            image_2 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        def random_position(image_1, image_2, thr):
          if random.random() < thr:
            img = torch.stack((image_1, image_2), 0)
          else:
            img = torch.stack((image_2, image_1), 0)
          return img

        if self.split == 'train':
            image = random_position(image_1, image_2, 0.5)
        else:
            image = torch.stack((image_1, image_2), 0)

        report_ids = np.array(example['ids'])

        input_sequence = np.zeros(self.max_seq_length, dtype='int')
        target_sequence = np.zeros(self.max_seq_length, dtype='int')

        input_sequence[:len(report_ids)] = report_ids
        target_sequence[:len(report_ids)-1] = report_ids[1:]

        gv_feat = np.zeros((1, 1)) # Never been used
        return indices, input_sequence, target_sequence, gv_feat, image