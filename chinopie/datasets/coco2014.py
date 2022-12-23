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
from loguru import logger

URLS = {
    "train_img": "http://images.cocodataset.org/zips/train2014.zip",
    "val_img": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def prepare_coco2014(root: str, phase: str):
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
        logger.error(f"unknown phase: {phase}")
        exit(0)

    # image file
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        logger.warning(f"downloading {URLS[phase+'_img']} to {cached_file}")
        os.chdir(tmpdir)
        subprocess.call("wget " + URLS[phase + "_img"], shell=True)
        os.chdir(root)

    # extract image
    img_data = os.path.join(root, filename.split(".")[0])
    if not os.path.exists(img_data):
        logger.warning(
            "[dataset] Extracting tar file {file} to {path}".format(
                file=cached_file, path=root
            )
        )
        command = "unzip -q {} -d {}".format(cached_file, root)
        os.system(command)
    logger.warning("[dataset] Done!")

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, "annotations_trainval2014.zip")
    if not os.path.exists(cached_file):
        logger.warning(
            'Downloading: "{}" to {}\n'.format(URLS["annotations"], cached_file)
        )
        os.chdir(tmpdir)
        subprocess.call("wget " + URLS["annotations"], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(root, "annotations")
    if not os.path.exists(annotations_data):
        logger.warning(
            "[dataset] Extracting tar file {file} to {path}".format(
                file=cached_file, path=root
            )
        )
        command = "unzip -q {} -d {}".format(cached_file, root)
        os.system(command)
    logger.warning("[annotation] Done!")

    annotations_data = os.path.join(root, "annotations")
    anno = os.path.join(root, "{}_annotation.json".format(phase))
    img_id: Dict[str, Dict[str, Any]] = {}
    annotations_id: Dict[str, Set[int]] = {}
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
            annotations_id[annotation["image_id"]].add(
                cat2idx[category_id[annotation["category_id"]]]
            )
        for img in images:
            if img["id"] not in annotations_id:
                continue
            if img["id"] not in img_id:
                img_id[img["id"]] = {}
            img_id[img["id"]]["file_name"] = img["file_name"]
            img_id[img["id"]]["labels"] = list(annotations_id[img["id"]])
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
    logger.warning("[json] Done!")
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
        logger.warning(
            f"[dataset] COCO2014 classification {phase} phase, {self.num_classes} classes, {len(self.img_list)} images"
        )
    
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
        image = self.preprocess(
            Image.open(os.path.join(self.root, f"{self.phase}2014", filename)).convert(
                "RGB"
            )
        )

        target = torch.zeros(self.num_classes, dtype=torch.int)
        target[labels] = 1
        if self.negatives_as_neg1:
            target[negative_labels]=-1


        return {
            "index": index,
            "name": filename,
            "image": image,
            "target": target,
        }
    
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