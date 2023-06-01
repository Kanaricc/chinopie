import os
import random
import subprocess
from typing import Any, List, Optional, Tuple
import json
from PIL import Image
from xml.dom import minidom
import torch
from torch.functional import Tensor
import shutil

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from loguru import logger

from .. import download_with_progress,extract_zip
from . import MultiLabelLocalDataset


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


def prepare_voc12(root: str,ignore_difficult_label:bool=True):
    work_dir = os.getcwd()

    if not os.path.exists(root):
        os.mkdir(root)

    tmp_dir = os.path.join(root, "tmp")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    cached_tv_file = os.path.join(tmp_dir, "VOCtrainval_11-May-2012.tar")
    cached_test_file = os.path.join(tmp_dir, "VOC2012test.tar")
    if not os.path.exists(cached_tv_file):
        logger.warning(f"downloading voc12 trainval dataset")
        download_with_progress(DATA_URL,cached_tv_file)
        logger.warning("done")
    if not os.path.exists(cached_test_file):
        logger.warning(f"downloading voc12 test dataset")
        download_with_progress(TEST_DATA_URL,cached_test_file)
        logger.warning("done")

    extracted_dir = os.path.join(tmp_dir, "extracted")
    if not os.path.exists(extracted_dir):
        logger.warning(f"extracting dataset")
        os.mkdir(extracted_dir)
        extract_zip(cached_tv_file,extracted_dir)
        extract_zip(cached_test_file,extracted_dir)
        logger.warning("done")

    vocdevkit = os.path.join(extracted_dir, "VOCdevkit", "VOC2012")
    anno_path=os.path.join(vocdevkit, "Annotations")
    assert os.path.exists(vocdevkit), f"no path {vocdevkit}"

    img_dir = os.path.join(root, "img")
    if not os.path.exists(img_dir):
        logger.warning(f"refactor images dir structure")
        assert os.path.exists(os.path.join(vocdevkit, "JPEGImages"))
        shutil.move(os.path.join(vocdevkit,'JPEGImages'),img_dir)
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

        warning_ignorance=False
        for phase in ["train", "val", "trainval", "test"]:
            img_annotations = {}
            label_difficulty = {}
            label_easy = {}
            img_ids = []
            for label in LABEL2ID:
                label_file = os.path.join(labels_dir, f"{label}_{phase}.txt")
                assert os.path.exists(label_file)
                with open(label_file, "r") as f:
                    for line in f:
                        spline = list(map(lambda x: x.strip(), line.strip().split(" ")))
                        image_id, positive = spline[0], spline[-1]
                        if positive == "1":
                            # first to see an image
                            if image_id not in img_annotations:
                                img_annotations[image_id] = []
                                img_ids.append(image_id)
                                # load difficulty
                                label_difficulty[image_id] = set()
                                label_easy[image_id]=set()
                                instance_anno=os.path.join(anno_path,f"{image_id}.xml")
                                dom_tree=minidom.parse(instance_anno).documentElement
                                objects=dom_tree.getElementsByTagName('object')
                                for obj in objects:
                                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                                        tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                                        label_difficulty[image_id].add(tag)
                                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '0':
                                        tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                                        label_easy[image_id].add(tag)
                                    
                            if ignore_difficult_label:
                                if label in label_easy[image_id]:
                                    img_annotations[image_id].append(LABEL2ID[label])
                                elif label not in label_difficulty[image_id]:
                                    img_annotations[image_id].append(LABEL2ID[label])
                                else:
                                    if not warning_ignorance:
                                        logger.debug(f"label like `{label}` will be ignored in `{image_id}` since it's tagged difficult.")
                                        warning_ignorance=True
                            else:
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


class VOC2012Dataset(MultiLabelLocalDataset):
    def __init__(self, root:str,phase:str,preprocess:Any,extra_preprocess=None,negatives_as_neg1=False) -> None:
        assert phase in ["train","val","trainval","test"]
        prepare_voc12(root)

        with open(os.path.join(root, f"annotations_{phase}.json"), "r") as f:
            img_list = json.load(f)
        with open(os.path.join(root, f"categories.json"), "r") as f:
            cat2id = json.load(f)


        logger.warning(
            f"[VOC2012] load num of classes {len(cat2id)}, num images {len(img_list)}"
        )

        num_labels = len(cat2id)
        img_paths=list(map(lambda x:os.path.join(root, "img", x['name']),img_list))
        annotations=list(map(lambda x:x['labels'],img_list))
        annotation_labels=get_voc_labels()


        super().__init__(img_paths, num_labels, annotations, annotation_labels, preprocess, extra_preprocess, negatives_as_neg1)