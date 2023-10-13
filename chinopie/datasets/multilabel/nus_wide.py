import os
import subprocess
import json
from typing import Any, Dict, List, Set,Optional
from torch import Tensor
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

def prepare_nus_wide(root:str):
    raise NotImplementedError()

import csv
import os
import os.path
import tarfile
import torch.utils.data as data
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
import pickle
import glob
from collections import defaultdict

fn_map = {}

def read_info(root, set):
    imagelist = {}
    hash2ids = {}
    if set == "trainval": 
        path = os.path.join(root, "ImageList", "TrainImagelist.txt")
    elif set == "test":
        path = os.path.join(root, "ImageList", "TestImagelist.txt")
    else:
        raise NotImplementedError(f"unknown phase {set}")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split('\\')[-1]
            start = line.index('_')
            end = line.index('.')
            imagelist[i] = line[start+1:end]
            hash2ids[line[start+1:end]] = i

    return imagelist


def read_object_labels_csv(file, imagelist, fn_map, header=True):
    images,targets = [],[]
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = int(row[0])
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                name2 = fn_map[imagelist[name]]
                images.append(name2)
                targets.append(labels)
            rownum += 1
    return images,targets

def get_defined_labels_from_csv(file:str):
    header=None
    with open(file,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            header=row
            break
    assert header is not None
    return header[1:]


NUS_WIDE_RAW_LABELS="elk,bridge,plane,waterfall,rainbow,lake,book,bear,sports,computer,cat,airport,reflection,flowers,zebra,tattoo,train,wedding,military,running,sunset,water,moon,fish,statue,whales,ocean,tree,sign,boats,beach,town,protest,food,fire,sand,police,toy,glacier,window,sun,clouds,plants,rocks,coral,cars,dog,cow,horses,leaf,tower,fox,flags,grass,snow,temple,swimmers,buildings,birds,valley,cityscape,road,map,house,frost,sky,harbor,dancing,tiger,garden,earthquake,vehicle,surf,street,person,mountain,animal,soccer,nighttime,railroad,castle".split(',')

def get_nus_wide_labels():
    return NUS_WIDE_RAW_LABELS

class NusWideDataset(MultiLabelLocalDataset):
    def __init__(self, root:str, phase:str, preprocess: Any, extra_preprocess: Optional[Any], negatives_as_neg1=False) -> None:
        images_dir=os.path.join(root,'images')

        global fn_map
        if len(fn_map)==0:
            for fn in glob.glob(os.path.join(root,'images',"*.jpg")):
                tmp = fn.split('_')[1]
                fn_map[tmp] = fn
            _logger.debug("[dataset] mapped NUS-WIDE image path")

        # define path of csv file
        path_csv = os.path.join(root, 'classification_labels')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + phase + '.csv')
        imagelist = read_info(root, phase)

        img_paths,targets=read_object_labels_csv(file_csv, imagelist, fn_map)
        annotations=torch.stack(targets)
        labels=get_defined_labels_from_csv(file_csv)

        assert labels==NUS_WIDE_RAW_LABELS
        assert len(img_paths)==annotations.size(0)
        assert len(labels)==annotations.size(1)

        _logger.info(f"[dataset] NUS-WIDE {phase} set: #image {len(img_paths)}, #class {annotations.size(1)}")

        super().__init__(img_paths, len(labels), annotations, labels, preprocess, extra_preprocess, negatives_as_neg1)