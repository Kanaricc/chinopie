import os
import pdb
import pickle
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import MultiClassInMemoryDataset
from .. import download_with_progress,extract_zip
from ... import logging
_logger=logging.get_logger(__name__)

class CIFAR10(MultiClassInMemoryDataset):
    num_class=10
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool,
        preprocess: Optional[Callable],
        extra_preprocess: Optional[Callable] = None,
    ) -> None:
        self.root=root
        self.train = train  # training set or test set

        self.download()

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        raw_data: List[Any] = []
        targets:List[int] = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                raw_data.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        data = np.vstack(raw_data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        super().__init__(data,self.num_class,annotations=targets,annotation_labels=self.classes,preprocess=preprocess,extra_preprocess=extra_preprocess)


    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes:List[str] = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def download(self) -> None:
        path=os.path.join(self.root,self.filename)
        if not os.path.exists(path):
            download_with_progress(self.url,path)
            _logger.info(f"downloaded cifar dataset zip to `{path}`")
        if not os.path.exists(os.path.join(self.root,self.base_folder)):
            extract_zip(path,self.root)
            _logger.info(f"extracted cifar dataset to `{self.root}`")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"



class CIFAR100(CIFAR10):
    num_class=100
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }