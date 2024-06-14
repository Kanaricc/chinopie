import os
from typing import Any, List
from . import MultiClassLocalDataset

# TODO: add downloading script for CUB-200-2011

class Cub200Dataset(MultiClassLocalDataset):
    def __init__(self,root:str, preprocess: Any, extra_preprocess: Any | None, negatives_as_neg1=False) -> None:
        
        with open(os.path.join(root,'classes.txt'),'r') as f:
            annotation_labels=[x.strip().split(' ')[1].split('.')[1] for x in f.readlines()]
        
        img_paths=[]
        annotations=[]
        for k,v in enumerate(annotation_labels):
            sub_dir=f"{k+1:03d}.{v}"
            for dirpath,dirname,filenames in os.walk(os.path.join(root,"CUB_200_2011",sub_dir)):
                img_paths.extend([os.path.join(dirpath,x) for x in filenames])
                annotations.extend([k for _ in range(len(filenames))])
        
        super().__init__(img_paths, len(annotation_labels), annotations, annotation_labels, preprocess, extra_preprocess, negatives_as_neg1)
