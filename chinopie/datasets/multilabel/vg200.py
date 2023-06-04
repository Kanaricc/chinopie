import os
import subprocess
import json
from typing import Any, Dict, List, Set
from torch.functional import Tensor
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from ... import logging
_logger=logging.get_logger(__name__)

URLS = {
    "train_img": "http://images.cocodataset.org/zips/train2014.zip",
    "val_img": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def prepare_vg(root: str, phase: str, include_segmentations:bool=False):
    raise NotImplemented


def get_vg_labels(root: str) -> List[str]:
    with open(os.path.join(root, "category.json"), "r") as f:
        LABEL2ID = json.load(f)
    keys = ["" for i in range(len(LABEL2ID))]
    for k in LABEL2ID:
        keys[LABEL2ID[k]] = k
    return keys


class COCO2014Dataset(Dataset):
    img_list: List[Any]
    one_hot: bool

    def __init__(
        self,
        root: str,
        preprocess: Any,
        phase: str = "train",
        negatives_as_neg1=False
    ):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.preprocess = preprocess
        self.negatives_as_neg1=negatives_as_neg1

        prepare_coco2014(root, phase)
        self.load_annotation()
        _logger.info(
            f"[dataset] COCO2014 classification {phase} phase, {self.num_classes} classes, {len(self.img_list)} images"
        )
    
    def reg_extra_preprocess(self,preprocess:Any):
        self.extra_preprocess=preprocess
    
    def retain_range(self,l:int,r:int):
        self.img_list=self.img_list[l:r]

    def load_annotation(self):
        with open(os.path.join(self.root, f"{self.phase}_annotation.json"), "r") as f:
            self.img_list = json.load(f)
        with open(os.path.join(self.root, f"category.json"), "r") as f:
            self.cat2idx = json.load(f)
        self.num_classes = len(self.cat2idx)

        # add negative label
        full_set=set(range(self.num_classes))
        for img in self.img_list:
            img['negative_labels']=list(full_set-set(img['labels']))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index: int):
        item = self.img_list[index]
        filename = item["file_name"]
        labels = sorted(item["labels"])
        negative_labels=sorted(item['negative_labels'])
        rgb_image=Image.open(os.path.join(self.root, f"{self.phase}2014", filename)).convert(
                "RGB"
            )
        image = self.preprocess(rgb_image)
        

        target = torch.zeros(self.num_classes, dtype=torch.int)
        target[labels] = 1
        if self.negatives_as_neg1:
            target[negative_labels]=-1

        res={
            "index": index,
            "name": filename,
            "image": image,
            "target": target,
        }
        if hasattr(self,'extra_preprocess'):
            extra_image=self.extra_preprocess(rgb_image)
            res['extra_image']=extra_image
        return res
    
    def get_all_labels(self):
        tmp=torch.zeros((len(self.img_list),self.num_classes),dtype=torch.long)
        for i in range(len(self.img_list)):
            tmp[i][self.img_list[i]['labels']]=1
            if self.negatives_as_neg1:
                tmp[i][self.img_list[i]['negative_labels']]=-1
                
        return tmp
    
    def apply_new_labels(self,labels:Tensor):
        assert labels.size(0)==len(self.img_list) and labels.size(1)==self.num_classes
        assert labels.dtype==torch.int or labels.dtype==torch.long
        for i in range(len(self.img_list)):
            label=labels[i]
            self.img_list[i]['labels']=(label==1).nonzero(as_tuple=True)[0].tolist()
            self.img_list[i]['negative_labels']=(label==-1).nonzero(as_tuple=True)[0].tolist()