import os
import subprocess
import json
from typing import Any, Dict, List, Set,Optional
from torch.functional import Tensor
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import urllib.request
import zipfile
from tqdm import tqdm

from .. import download_with_progress,extract_zip
from . import MultiLabelLocalDataset

from ... import logging
_logger=logging.get_logger(__name__)

URLS = {
    "train_img": "http://images.cocodataset.org/zips/train2014.zip",
    "val_img": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def prepare_coco2014(root: str, phase: str, include_segmentations:bool=False):
    work_dir = os.getcwd()
    root = os.path.abspath(root)
    tmpdir = os.path.join(root, "tmp")

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    if phase == "train":
        filename = "train2014.zip"
    elif phase == "val":
        filename = "val2014.zip"
    else:
        _logger.error(f"unknown phase: {phase}")
        exit(0)

    # image file
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        _logger.info(f"downloading {URLS[phase+'_img']} to {cached_file}")

        download_with_progress(URLS[phase + "_img"],cached_file)

    # extract image
    img_data = os.path.join(root, filename.split(".")[0])
    if not os.path.exists(img_data):
        _logger.info(
            "[dataset] Extracting tar file {file} to {path}".format(
                file=cached_file, path=root
            )
        )
        extract_zip(cached_file,root)
    _logger.info("[dataset] Done!")

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, "annotations_trainval2014.zip")
    if not os.path.exists(cached_file):
        _logger.info(
            'Downloading: "{}" to {}\n'.format(URLS["annotations"], cached_file)
        )
        download_with_progress(URLS["annotations"],cached_file)
    annotations_data = os.path.join(root, "annotations")
    if not os.path.exists(annotations_data):
        _logger.info(
            "[dataset] Extracting tar file {file} to {path}".format(
                file=cached_file, path=root
            )
        )
        extract_zip(cached_file,root)
    _logger.info("[annotation] Done!")

    annotations_data = os.path.join(root, "annotations")
    anno = os.path.join(root, "{}_annotation.json".format(phase))
    img_id: Dict[str, Dict[str, Any]] = {}
    annotations_id: Dict[str, Set[int]] = {}
    seg_id: Dict[str, Dict[int,List[Any]]] = {}
    if not os.path.exists(anno):
        annotations_file = json.load(
            open(os.path.join(annotations_data, "instances_{}2014.json".format(phase)))
        )
        annotations = annotations_file["annotations"]
        category = annotations_file["categories"]
        category_id: Dict[int, str] = {}
        for cat in category:
            category_id[cat["id"]] = cat["name"]
        cat2idx = category_to_idx(sorted(category_id.values()))
        images = annotations_file["images"]
        for annotation in annotations:
            if annotation["image_id"] not in annotations_id:
                annotations_id[annotation["image_id"]] = set()
                seg_id[annotation['image_id']]={}
            label_id=cat2idx[category_id[annotation["category_id"]]]
            annotations_id[annotation["image_id"]].add(
                label_id
            )
            if label_id not in seg_id[annotation['image_id']]:
                seg_id[annotation['image_id']][label_id]=[]
            seg_id[annotation['image_id']][label_id].append(annotation["segmentation"])

        for img in images:
            if img["id"] not in annotations_id:
                continue
            if img["id"] not in img_id:
                img_id[img["id"]] = {}
            img_id[img["id"]]["file_name"] = img["file_name"]
            img_id[img["id"]]["labels"] = list(annotations_id[img["id"]])
            if include_segmentations:
                img_id[img["id"]]['segmentations']=seg_id[img["id"]]
        anno_list: List[Any] = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, "w"))
        if not os.path.exists(os.path.join(root, "category.json")):
            json.dump(cat2idx, open(os.path.join(root, "category.json"), "w"))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    _logger.info("[json] Done!")
    os.chdir(work_dir)


def category_to_idx(category: List[str]) -> Dict[str, int]:
    cat2idx: Dict[str, int] = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


def get_coco_labels(root: str) -> List[str]:
    with open(os.path.join(root, "category.json"), "r") as f:
        LABEL2ID = json.load(f)
    keys = ["" for i in range(len(LABEL2ID))]
    for k in LABEL2ID:
        keys[LABEL2ID[k]] = k
    return keys

def _get_preprocess(phase:str):
    if phase=='train':
        return transforms.Compose([
            transforms.Resize((448,448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
        ])

class COCO2014Dataset(MultiLabelLocalDataset):
    def __init__(self,root:str,phase:str,preprocess:Optional[Any],extra_preprocess:Optional[Any]=None,negatives_as_neg1=False,prepreprocess=None) -> None:
        assert phase in ['train', 'val']
        self.root = os.path.abspath(root)
        self.phase = phase

        prepare_coco2014(root, phase)
        self.load_annotation()
        _logger.info(
            f"[dataset] COCO2014 classification {phase} phase, {self.num_classes} classes, {len(self.img_list)} images"
        )

        annotation_labels=get_coco_labels(root)
        num_labels=len(annotation_labels)
        img_paths=list(map(lambda x:os.path.join(self.root, f"{self.phase}2014", x['file_name']),self.img_list))
        annotations=list(map(lambda x:x['labels'],self.img_list))

        if preprocess==None:
            preprocess=_get_preprocess(phase)

        super().__init__(img_paths, num_labels, annotations, annotation_labels, preprocess, extra_preprocess, negatives_as_neg1,prepreprocess)
    

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