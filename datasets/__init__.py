from datasets.coco_dataset import CocoDataset
from datasets.radiology_dataset import  IUXRAY
from datasets.radiology_dataset import MIMICCXR

__factory = {
    'IUXRAY': IUXRAY,
    'MIMICCXR': MIMICCXR,
    'COCO': CocoDataset,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown Dataset:", name)
    return __factory[name](*args, **kwargs)