import os
import shutil
from typing import Any, List
from . import MultiClassLocalDataset
from .. import download_with_progress,extract_zip

from ... import logging
_logger=logging.get_logger(__name__)

URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"



# TODO: add downloading script for CUB-200-2011
def _download_cub_200_2011(root: str):
    root=os.path.abspath(root)
    
    if os.path.exists(os.path.join(root,'README')):
        # dataset exists
        return
    
    zip_filepath=os.path.join(root,'CUB_200_2011.tgz')
    _logger.info("downloading CUB-200-2011 dataset")
    download_with_progress(URL,zip_filepath)
    _logger.info("extracting files")
    extract_zip(zip_filepath,root)
    
    _logger.info("moving files")
    shutil.move(os.path.join(root,'CUB_200_2011'),root)

    _logger.info("finished preparing CUB-200-2011 dataset")
    
    
    

class Cub200Dataset(MultiClassLocalDataset):
    def __init__(self,root:str, preprocess: Any, extra_preprocess: Any | None, negatives_as_neg1=False) -> None:
        _download_cub_200_2011(root)
        
        with open(os.path.join(root,'classes.txt'),'r') as f:
            annotation_labels=[x.strip().split(' ')[1].split('.')[1] for x in f.readlines()]
        
        img_paths=[]
        annotations=[]
        for k,v in enumerate(annotation_labels):
            sub_dir=f"{k+1:03d}.{v}"
            for dirpath,dirname,filenames in os.walk(os.path.join(root,"images",sub_dir)):
                img_paths.extend([os.path.join(dirpath,x) for x in filenames])
                annotations.extend([k for _ in range(len(filenames))])
        
        super().__init__(img_paths, len(annotation_labels), annotations, annotation_labels, preprocess, extra_preprocess, negatives_as_neg1)
