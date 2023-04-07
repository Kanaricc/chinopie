from tqdm import tqdm
import urllib.request
import zipfile,tarfile

class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download_with_progress(url:str,tgt:str):
    with TqdmUpTo() as progress:
            urllib.request.urlretrieve(url,tgt,reporthook=progress.update_to)

def extract_zip(file:str,tgt:str):
    if file.endswith('.zip'):
        with zipfile.ZipFile(file,'r') as f:
            f.extractall(tgt)
    elif file.endswith('.tar'):
        with tarfile.TarFile(file,'r') as f:
            f.extractall()



from .multilabel.coco2014 import COCO2014Dataset,get_coco_labels
from .multilabel.voc2012 import VOC2012Dataset,get_voc_labels
from .multilabel.voc2007 import VOC2007Dataset,get_voc_labels
from .fakeset import FakeConstSet,FakeEmptySet,FakeOutput,FakeRandomSet,FakeNormalSet

