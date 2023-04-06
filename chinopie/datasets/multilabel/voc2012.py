import os
import random
import subprocess
from typing import Any, List, Optional, Tuple
import json
from PIL import Image
import torch
from torch.functional import Tensor
import shutil

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from loguru import logger

from . import MultiLabelDataset,MultiLabelSample


DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
TEST_DATA_URL = "http://pjreddie.com/media/files/VOC2012test.tar"

LABEL2ID = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}


def get_voc_labels() -> List[str]:
    keys = ["" for i in range(len(LABEL2ID))]
    for k in LABEL2ID:
        keys[LABEL2ID[k]] = k
    return keys


def prepare_voc12(root: str):
    work_dir = os.getcwd()

    if not os.path.exists(root):
        os.mkdir(root)

    tmp_dir = os.path.join(root, "tmp")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    cached_file = os.path.join(tmp_dir, "VOCtrainval_11-May-2012.tar")
    if not os.path.exists(cached_file):
        logger.warning(f"downloading voc12 dataset")
        subprocess.call(f"wget -O {cached_file} {DATA_URL}", shell=True)
        logger.warning("done")

    extracted_dir = os.path.join(tmp_dir, "extracted")
    if not os.path.exists(extracted_dir):
        logger.warning(f"extracting dataset")
        os.mkdir(extracted_dir)
        subprocess.call(f"tar -xf {cached_file} -C {extracted_dir}", shell=True)
        logger.warning("done")

    vocdevkit = os.path.join(extracted_dir, "VOCdevkit", "VOC2012")
    assert os.path.exists(vocdevkit)

    img_dir = os.path.join(root, "img")
    if not os.path.exists(img_dir):
        logger.warning(f"refactor images dir structure")
        assert os.path.exists(os.path.join(vocdevkit, "JPEGImages"))
        subprocess.call(
            f"mv '{os.path.join(vocdevkit,'JPEGImages')}' '{img_dir}'", shell=True
        )
        logger.warning("done")

    cat_json = os.path.join(root, "categories.json")
    if not os.path.exists(cat_json):
        logger.warning(f"dumping categories mapping relations")
        json.dump(LABEL2ID, open(cat_json, "w"))
        logger.warning("done")

    anno_json = os.path.join(root, "annotations_train.json")
    if not os.path.exists(anno_json):
        logger.warning(f"generating annotations json")
        labels_dir = os.path.join(vocdevkit, "ImageSets", "Main")

        for phase in ["train", "val"]:
            img_annotations = {}
            img_ids = []
            for label in LABEL2ID:
                label_file = os.path.join(labels_dir, f"{label}_{phase}.txt")
                assert os.path.exists(label_file)
                with open(label_file, "r") as f:
                    for line in f:
                        spline = list(map(lambda x: x.strip(), line.strip().split(" ")))
                        image_id, positive = spline[0], spline[-1]
                        if positive == "1":
                            if image_id not in img_annotations:
                                img_annotations[image_id] = []
                                img_ids.append(image_id)
                            img_annotations[image_id].append(LABEL2ID[label])

            anno: List = []
            for i, image_id in enumerate(img_ids):
                t = {
                    "id": i,
                    "name": f"{image_id}.jpg",
                    "labels": img_annotations[image_id],
                }
                anno.append(t)

            json_file = os.path.join(root, f"annotations_{phase}.json")
            json.dump(anno, open(json_file, "w"))

        logger.warning("done")


class VOC2012Dataset(MultiLabelDataset):
    def __init__(
        self,
        root: str,
        preprocess: Any,
        phase: str = "train",
        negatives_as_neg1: bool = False,
    ):
        assert phase == "train" or phase == "val"
        prepare_voc12(root)

        self.root = root
        self.preprocess = preprocess
        self.phase = phase
        self.negatives_as_neg1 = negatives_as_neg1

        with open(os.path.join(self.root, f"annotations_{self.phase}.json"), "r") as f:
            self.img_list = json.load(f)
        with open(os.path.join(self.root, f"categories.json"), "r") as f:
            self.cat2id = json.load(f)

        self.num_classes = len(self.cat2id)

        full_set = set(range(self.num_classes))
        for img in self.img_list:
            img["negative_labels"] = list(full_set - set(img["labels"]))

        logger.warning(
            f"[VOC2012] load num of classes {len(self.cat2id)}, num images {len(self.img_list)}"
        )

    def reg_extra_preprocess(self, preprocess: Any):
        self.extra_preprocess = preprocess

    def __getitem__(self, index)->MultiLabelSample:
        item = self.img_list[index]

        _, filename, target, nagatives = (
            item["id"],
            item["name"],
            item["labels"],
            item["negative_labels"],
        )
        rgb_image = Image.open(os.path.join(self.root, "img", filename)).convert("RGB")
        image = self.preprocess(rgb_image)

        target2 = torch.zeros(self.num_classes, dtype=torch.int)
        target2[target] = 1
        if self.negatives_as_neg1:
            target[nagatives] = -1

        res:MultiLabelSample = {
            "index": index,
            "name": filename,
            "image": image,
            "extra_image":None,
            "target": target2,
        }
        if hasattr(self, "extra_preprocess"):
            extra_image = self.extra_preprocess(rgb_image)
            res["extra_image"] = extra_image
        return res

    def __len__(self):
        return len(self.img_list)

    def retain(self, l: int, r: int):
        self.img_list = self.img_list[l:r]

    def get_all_labels(self):
        tmp = torch.zeros((len(self.img_list), self.num_classes), dtype=torch.long)
        for i in range(len(self.img_list)):
            tmp[i][self.img_list[i]["labels"]] = 1
            if self.negatives_as_neg1:
                tmp[i][self.img_list[i]["negative_labels"]] = -1

        return tmp

    def apply_new_labels(self, labels: Tensor):
        assert (
            labels.size(0) == len(self.img_list) and labels.size(1) == self.num_classes
        )
        assert labels.dtype == torch.int or labels.dtype == torch.long
        for i in range(len(self.img_list)):
            label = labels[i]
            self.img_list[i]["labels"] = (label == 1).nonzero(as_tuple=True)[0].tolist()
            self.img_list[i]["negative_labels"] = (
                (label == -1).nonzero(as_tuple=True)[0].tolist()
            )
