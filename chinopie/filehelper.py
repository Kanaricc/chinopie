import os,shutil
from typing import Optional,List
from datetime import datetime

from .ddpsession import DdpSession
from loguru import logger
import pathlib


DIR_SHARE_STATE = "state"
DIR_CHECKPOINTS = "checkpoints"
DIR_DATASET = "data"
DIR_TENSORBOARD = "boards"

class FileHelper:
    def __init__(
            self, disk_root: str, comment: str, ddp_session: Optional[DdpSession] = None
    ):
        self.disk_root = disk_root
        self.ddp_session = ddp_session
        self.comment = comment

        if self._is_main_process():
            if not os.path.exists(os.path.join(self.disk_root, DIR_CHECKPOINTS)):
                os.mkdir(os.path.join(self.disk_root, DIR_CHECKPOINTS))
            if not os.path.exists(os.path.join(self.disk_root, DIR_TENSORBOARD)):
                os.mkdir(os.path.join(self.disk_root, DIR_TENSORBOARD))
            if not os.path.exists(os.path.join(self.disk_root, DIR_DATASET)):
                os.mkdir(os.path.join(self.disk_root, DIR_DATASET))
            if not os.path.exists(os.path.join(self.disk_root, DIR_SHARE_STATE)):
                os.mkdir(os.path.join(self.disk_root, DIR_SHARE_STATE))

        if self.ddp_session:
            logger.debug("found initialized ddp session")
            DdpSession.barrier()
            logger.debug("waited for filehelper distributed initialization")

        self.ckpt_dir = os.path.join(self.disk_root, DIR_CHECKPOINTS, comment)
        # self.board_dir = os.path.join(
        #     self.disk_root,
        #     DIR_TENSORBOARD,
        #     f"{self.comment}-{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}",
        # )

    def prepare_checkpoint_dir(self):
        if not os.path.exists(self.ckpt_dir):
            if self._is_main_process():
                os.mkdir(self.ckpt_dir)
        if self.ddp_session:
            DdpSession.barrier()

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        find the latest checkpoint file at checkpoint dir.
        """

        if not os.path.exists(self.ckpt_dir):
            return None

        checkpoint_files: List[str] = []
        for (dirpath, dirnames, filenames) in os.walk(self.ckpt_dir):
            checkpoint_files.extend(filenames)

        latest_checkpoint_path = None
        latest_checkpoint_epoch = -1

        for file in checkpoint_files:
            if file.find("best") != -1:
                continue
            if file.find("init") != -1:
                continue
            if file.find("checkpoint") == -1:
                continue
            epoch = int(file.split(".")[0].split("-")[1])
            if epoch > latest_checkpoint_epoch:
                latest_checkpoint_epoch = epoch
                latest_checkpoint_path = file

        if latest_checkpoint_path:
            return os.path.join(self.ckpt_dir, latest_checkpoint_path)

    def get_initparams_slot(self) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        logger.info("[INIT] you have ask for initialization slot")
        filename = f"init.pth"
        return os.path.join(self.ckpt_dir, filename)

    def get_checkpoint_slot(self, cur_epoch: int) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        filename = f"checkpoint-{cur_epoch}.pth"
        return os.path.join(self.ckpt_dir, filename)

    def get_dataset_slot(self, dataset_id: str) -> str:
        return os.path.join(self.disk_root, DIR_DATASET, dataset_id)
    
    def get_state_slot(self,*name:str)->str:
        path=os.path.join(self.disk_root,DIR_SHARE_STATE,*name)
        parent=pathlib.Path(path).parent
        if not parent.exists():
            os.makedirs(parent)
        return path

    def get_best_checkpoint_slot(self) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        return os.path.join(self.ckpt_dir, "best.pth")
    
    @property
    def default_board_dir(self) -> str:
        return os.path.join(
            self.disk_root,DIR_TENSORBOARD,self.comment,
        )

    def _is_main_process(self):
        return self.ddp_session is None or DdpSession.is_main_process()
