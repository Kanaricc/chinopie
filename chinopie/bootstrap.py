from datetime import datetime
import logging
import os, sys, shutil,pdb,gc
import argparse
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


import torch
import torch.backends.mps
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import optuna
import numpy as np

from .modelhelper import HyperparameterManager,ModelStaff
from . import iddp as dist
from .filehelper import GlobalFileHelper,InstanceFileHelper
from .phasehelper import (
    PhaseEnv,
)
from .utils import show_params_in_3cols,create_snapshot,check_gitignore,set_fixed_seed
from .logging import get_logger,set_logger_file,set_verbosity

logger=get_logger(__name__)



# all block sync should be done in bootstrap. all data sync should be done in helper.

from .recipe import ModuleRecipe

class TrainBootstrap:
    def __init__(
        self,
        disk_root: str,
        num_epoch: int,
        load_checkpoint: bool,
        save_checkpoint: bool,
        comment: Optional[str],
        checkpoint_save_period: int = 1,
        enable_snapshot=False,
        enable_prune=False,
        world_size:int=1,
        seed:Optional[Any]=None,
        dev="",
        diagnose=False,
        verbose=False,
    ) -> None:
        argparser=argparse.ArgumentParser(
            prog='ChinoPie'
        )
        argparser.add_argument('-r','--disk_root',type=str,default=disk_root)
        argparser.add_argument('-n','--num_epoch',type=int,default=num_epoch)
        argparser.add_argument('-l','--load_checkpoint',action='store_true',default=load_checkpoint)
        argparser.add_argument('-s','--save_checkpoint',action='store_true',default=save_checkpoint)
        argparser.add_argument('-c','--comment',type=str,default=comment)
        argparser.add_argument('--dev',type=str,default=dev)
        argparser.add_argument('-d','--diagnose',action='store_true',default=diagnose)
        argparser.add_argument('-v','--verbose',action='store_true',default=verbose)
        argparser.add_argument('--clear',action='store_true',default=False)
        args,self._extra_arg_str=argparser.parse_known_args()

        self._disk_root = args.disk_root
        self._num_epoch = args.num_epoch
        if args.comment is not None:
            self._comment = args.comment
        else:
            self._comment = datetime.now().strftime("%Y%m%d%H%M%S")
        self._load_checkpoint:bool = args.load_checkpoint
        self._save_checkpoint:bool = args.save_checkpoint
        self._checkpoint_save_period = checkpoint_save_period
        self._dev = args.dev
        self._diagnose_mode = args.diagnose
        self._verbose:bool=args.verbose
        self._enable_prune=enable_prune
        self._world_size=world_size

        _init_logger(self._comment,self._verbose)

        self.file=GlobalFileHelper(disk_root)

        # set ddp
        if self._world_size>1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            dist.prefer_ddp()
            logger.info("[BOOTSTRAP] enable DDP")

        # set prune
        if self._enable_prune:
            logger.info('[BOOTSTRAP] early stop is enabled')

        # set clear
        if args.clear:
            # do clear
            input("[BOOTSTRAP] are you sure to clear all state files and logs? (press ctrl+c to quit)")
            self.clear()
            exit(0)
        
        # check git ignore
        check_gitignore([self._disk_root])

        # set diagnose mode
        if diagnose:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)
            logger.info("[BOOTSTRAP] diagnose mode enabled")
        
        # set snapshot
        if enable_snapshot:
            if not diagnose:
                create_snapshot(self._comment)
                logger.info("[BOOTSTRAP] created snapshot")
            else:
                logger.info("[BOOTSTRAP] snapshot is disabled in diagnose mode")
        
        # set fixed seed
        if seed is not None:
            self.set_fixed_seed(seed)
        
        # prepare hyperparameter manager
        self._hp_manager=HyperparameterManager()
        self._inherit_states:Dict[str,Any]={}
    
    @property
    def hp(self):
        return self._hp_manager
    
    def release(self):
        if self._diagnose_mode:
            self.file.clear_all_instance()
            logger.info("[BOOTSTRAP] deleted trial files in diagnose mode")
    
    def clear(self):
        if os.path.exists('logs'):shutil.rmtree('logs')
        if os.path.exists('opts'):shutil.rmtree('opts')
    
    def set_fixed_seed(self, seed: Any, ddp_seed=True):
        if not dist.is_preferred() or not ddp_seed:
            set_fixed_seed(seed)
        else:
            set_fixed_seed(seed+dist.get_rank())
            logger.info("[BOOTSTRAP] ddp detected, use different seed")
    
    def _flush_params(self):
        self._hp_manager.parse_args(self._extra_arg_str)
        

    def _init_ddp(self):
        logger.info("[BOOTSTRAP] initialized ddp")

    
    # def _report_info(self,helper:ModelStaff,board_dir:str):
    #     dataset_str = f"train({len(helper._data_train)}) val({len(helper._data_val)}) test({len(helper._data_test) if hasattr(helper, '_data_test') else 'not set'})"
    #     table = show_params_in_3cols(
    #         params={
    #             "proper device": self._dev,
    #             "diagnose": self._diagnose_mode,
    #             "epoch num": self._num_epoch,
    #             "early stop": self._enable_prune,
    #             "dataset": dataset_str,
    #             "board dir": board_dir,
    #             "checkpoint load/save": f"{self._load_checkpoint}/{self._save_checkpoint}",
    #             "custom probes": helper._custom_probes,
    #         }
    #     )
    #     logger.warning(f"[BOOTSTRAP] [INFO]\n{table}")
    #     logger.warning(f"[BOOTSTRAP] [HYPERPARAMETERS]\n{show_params_in_3cols(self._hp_manager.params)}")

    def _report_info(self):
        table = show_params_in_3cols(
            params={
                "prefer device": self._dev,
                "diagnose": self._diagnose_mode,
                "epoch num": self._num_epoch,
                "early stop": self._enable_prune,
                "checkpoint load/save": f"{self._load_checkpoint}/{self._save_checkpoint}",
            }
        )
        logger.warning(f"[BOOTSTRAP] [INFO]\n{table}")
    
    def _get_comment(self,stage:Optional[int]=None):
        if stage is None:
            stage_comment=self._comment
        else:
            stage_comment=f"{self._comment}({stage})"
        return stage_comment
    
    def _get_study_path(self,comment:str):
        return os.path.join("opts", f"{comment}.db")

    def optimize(
        self, recipe:ModuleRecipe,direction:str,inf_score:float, n_trials: int,num_epoch:Optional[int]=None, stage:Optional[int]=None,always_run:bool=False,inherit_states:Optional[Dict[str,Any]]=None,
    ):
        if inherit_states is None:
            inherit_states={}
        
        self._flush_params()
                
        stage_comment=self._get_comment(stage)
        # find previous file helper
        if stage and stage>0:
            prev_file_helpers=[self.file.get_exp_instance(f"{self._comment}({x})") for x in range(stage)]
            logger.debug("[BOOTSTRAP] found previous file helper")
        else:
            prev_file_helpers=None

        self.study_file=self.file.get_exp_instance(stage_comment)
        if not os.path.exists("opts"):
            os.mkdir("opts")
        storage_path = self._get_study_path(stage_comment)
        # when in diagnose mode, or the `always run` is true, do not save storage.
        if self._diagnose_mode or always_run:
            storage_path=None
        else:
            storage_path=f"sqlite:///{storage_path}"
        
        self._inf_score=inf_score
        self._best_trial_score=inf_score
        self._direction=direction.lower()
        assert direction in ['maximize','minimize'], f"direction must be whether `maximize` or `minimize`, but `{direction}`"
        study = optuna.create_study(study_name='deadbeef',direction=direction,storage=storage_path,load_if_exists=True)

        finished_trials=set()
        resumed_trials=set()
        for trial in study.trials:
            # if previously we have failed trials, resuming it with correct id
            if trial.state==optuna.trial.TrialState.FAIL:
                logger.info(f"[BOOTSTRAP] found failed trial {trial._trial_id} ({trial.user_attrs['trial_id']}), resuming")
                study.enqueue_trial(trial.params,trial.user_attrs)
                resumed_trials.add(trial.user_attrs['trial_id'])
            elif trial.user_attrs['num_epochs']<self._num_epoch:
                logger.info(f"[BOOTSTRAP] found unfinished trial {trial._trial_id} ({trial.user_attrs['trial_id']}), resuming")
                study.enqueue_trial(trial.params,trial.user_attrs)
                resumed_trials.add(trial.user_attrs['trial_id'])
            elif trial.state==optuna.trial.TrialState.COMPLETE or trial.state==optuna.trial.TrialState.PRUNED:
                logger.info(f"[BOOTSTRAP] found complete trial {trial._trial_id} ({trial.user_attrs['trial_id']})")
                if trial.user_attrs['trial_id'] in resumed_trials:          
                    resumed_trials.remove(trial.user_attrs['trial_id'])
                finished_trials.add(trial.user_attrs['trial_id'])
        logger.debug(f"[BOOTSTRAP] found finished trials {finished_trials}. the requested #trials is {n_trials}")
        if len(finished_trials)==n_trials:
            logger.warning(f"[BOOTSTRAP] this study `{stage_comment}` is already finished")
            logger.warning(
                f"[BOOTSTRAP] best hyperparameters\n{show_params_in_3cols(study.best_trial.user_attrs['params'])}"
            )
            logger.warning(f"[BOOTSTRAP] best score: {study.best_value}")
            return
        
        if num_epoch is None:num_epoch=self._num_epoch
        
        # in diagnose mode, run 1 times only
        if self._diagnose_mode:
            logger.info("[BOOTSTRAP] diagnose mode is enabled. run 1 trial and 2 epochs only.")
            n_trials=1
            num_epoch = 2
        # in always_run, do not load checkpoint
        load_checkpoint=self._load_checkpoint and not always_run
        assert num_epoch is not None

        # report bootstrap info
        self._report_info()

        try:
            for _ in range(n_trials):
                trial=study.ask()
                # process user attrs
                trial_id=trial._trial_id if 'trial_id' not in trial.user_attrs else trial.user_attrs['trial_id']
                trial.set_user_attr('trial_id',trial_id)
                trial.set_user_attr('num_epochs',self._num_epoch)

                self._hp_manager._set_trial(trial)
                logger.info(f"trial {trial._trial_id}, real id {trial_id}")
                trial_file=self.file.get_exp_instance(f"{stage_comment}_trial{trial_id}")
                
                # create checkpoint dir
                trial_file.prepare_checkpoint_dir()
                # assign hyperparamter
                recipe.ask_hyperparameter(self._hp_manager)


                try:
                    scores= 0.0 # TODO: find how to catch the score
                    mp.spawn(_wrapper_train,( # type: ignore
                        self._world_size,
                        trial,
                        recipe,
                        num_epoch,
                        load_checkpoint,
                        self._save_checkpoint,
                        self._enable_prune,
                        self._checkpoint_save_period,
                        trial_file,
                        prev_file_helpers,
                        direction,
                        inf_score,
                        self._dev,
                        self._diagnose_mode,
                        self._comment,
                        self._verbose
                    ),nprocs=self._world_size,join=True)
                    # _wrapper_train(
                    #     1,
                    #     trial,
                    #     recipe,
                    #     num_epoch,
                    #     load_checkpoint,
                    #     self._save_checkpoint,
                    #     self._enable_prune,
                    #     self._checkpoint_save_period,
                    #     trial_file,
                    #     prev_file_helpers,
                    #     direction,
                    #     inf_score,
                    #     self._dev,
                    #     self._diagnose_mode
                    # )

                    # append all params (suggested and fixed) into attrs
                    trial.set_user_attr('params',self._hp_manager.params)

                    assert type(scores)==float,"multiple target is not support now"
                    if scores>=self._best_trial_score:
                        self._best_trial_score=scores
                    study.tell(trial,scores)
                except optuna.TrialPruned:
                    study.tell(trial,state=optuna.trial.TrialState.PRUNED)
                gc.collect()
        finally:
            # post process
            logger.warning(f"[BOOTSTRAP] finish optimization of stage `{stage_comment}`")
            if len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))>0:
                best_trial=study.best_trial
                best_params = best_trial.user_attrs['params']
                best_value = study.best_value
                logger.warning(
                    f"[BOOTSTRAP] best hyperparameters\n{show_params_in_3cols(best_params)}"
                )
                logger.warning(f"[BOOTSTRAP] best score: {best_value}")

                best_file=self.file.get_exp_instance(f"{stage_comment}_trial{best_trial.user_attrs['trial_id']}")
                target_helper=self.file.get_exp_instance(stage_comment)
                shutil.copytree(best_file.default_board_dir,target_helper.default_board_dir,dirs_exist_ok=True)
                shutil.copytree(best_file.ckpt_dir,target_helper.ckpt_dir,dirs_exist_ok=True)
                logger.info("[BOOTSTRAP] copied best trial as the final result")
            else:
                logger.warning("[BOOTSTRAP] no trials are completed")
        
        logger.warning("[BOOTSTRAP] good luck!")
        return self._inherit_states


def _wrapper_train(
        rank:int,
        world_size:int,
        trial: optuna.Trial,
        recipe:ModuleRecipe,
        num_epoch:int,
        load_checkpoint:bool,
        save_checkpoint:bool,
        enable_prune:bool,
        checkpoint_save_period:int,
        trial_file:InstanceFileHelper,
        prev_file_helpers:Optional[List[InstanceFileHelper]],
        direction:str,
        inf_score:float,
        dev:Any,
        diagnose_mode:bool,
        study_comment:str,
        verbose:bool,
    ):
    if dev=='cpu':ddp_backend='gloo'
    elif dev=='cuda':ddp_backend='nccl'
    else:
        raise NotImplementedError(f"don't know what backend to use for device `{dev}`")
    if world_size>1:
        dist.init_process_group(ddp_backend,rank=rank,world_size=world_size)
    _init_logger(study_comment,verbose)

    best_score=inf_score
    # create board dir before training
    tbwriter = SummaryWriter(trial_file.default_board_dir)

    staff = ModelStaff(
        file_helper=trial_file,
        prev_file_helpers=prev_file_helpers,
        dev=dev,
    )

    recipe._total_epoch=num_epoch # this could be init before
    recipe._set_staff(staff)
    recipe.prepare(staff)
    staff.prepare(rank)
    # set optimizer
    staff._reg_optimizer(recipe.set_optimizers(staff._model))
    _scheduler=recipe.set_scheduler(staff._optimizer)
    if _scheduler is not None:
        staff._reg_scheduler(_scheduler)
    del _scheduler

    assert staff._get_flag('trainval_data_set'), "train or val set not set"
    if not staff._get_flag('test_data_set'):
        logger.warning("test set not set. test phase will be skipped.")
    
    recovered_epoch=None
    if load_checkpoint:
        latest_ckpt_path=trial_file.find_latest_checkpoint()
        if latest_ckpt_path is not None:
            logger.info(f"found latest checkpoint at `{latest_ckpt_path}`")
        else:
            logger.info(f"no checkpoint found")

        if load_checkpoint and latest_ckpt_path is not None:
            state=recipe.restore_ckpt(latest_ckpt_path)
            recovered_epoch=state['cur_epochi']
            best_score=state['best_score']

    # wait for all threads to load ckpt
    if dist.is_initialized():
        dist.barrier()

    logger.warning("ready to train model")
    recipe.before_start()
    for epochi in range(num_epoch):
        recipe._cur_epoch=epochi # set recipe progress reporter

        if dist.is_initialized():
            dist.barrier()

        if dist.is_main_process():
            logger.warning(f"=== START EPOCH {epochi} ===")
        
        recipe.before_epoch()
        if recovered_epoch is not None and epochi <= recovered_epoch:
            recipe.after_epoch()
            logger.info(f"fast pass epoch {recovered_epoch}")
            continue
        
        _prepare_dataloader_for_epoch(staff._dataloader_train,epochi)
        phase_train=PhaseEnv(
            "train",
            staff._data_train,
            staff._dataloader_train,
            dry_run=diagnose_mode,
            custom_probes=staff._custom_probes.copy(),
            dev=staff.dev
        )
        recipe.run_train_phase(phase_train)
        phase_train._check_update()
        _end_phase(staff,tbwriter,epochi,phase_train) # TODO

        if dist.is_initialized():
            logger.debug(f"rank {dist.get_rank()} reach barrier4")
            dist.barrier()
        
        logger.debug("ready to run val phase")
        _prepare_dataloader_for_epoch(staff._dataloader_val,epochi)
        phase_val=PhaseEnv(
            "val",
            staff._data_val,
            staff._dataloader_val,
            dry_run=diagnose_mode,
            custom_probes=staff._custom_probes.copy(),
            dev=staff.dev
        )
        recipe.run_val_phase(phase_val)
        phase_val._check_update()
        score=phase_val.score()
        _end_phase(staff,tbwriter,epochi,phase_val)

        if dist.is_initialized():
            dist.barrier()

        if staff._get_flag('test_data_set'):
            _prepare_dataloader_for_epoch(staff._dataloader_test,epochi)
            phase_test=PhaseEnv(
                "test",
                staff._data_test,
                staff._dataloader_test,
                dry_run=diagnose_mode,
                custom_probes=staff._custom_probes.copy(),
                dev=staff.dev
            )
            recipe.run_train_phase(phase_test)
            phase_test._check_update()
            score=phase_test.score()
            _end_phase(staff,tbwriter,epochi,phase_test)
        else: phase_test=None
        recipe.after_epoch()
        assert type(score)==float

        # output final result of this epoch
        loss_msg=[
            phase_train.loss_probe.average(),
            phase_val.loss_probe.average(),
            phase_test.loss_probe.average() if phase_test else 0
        ]
        score_msg=[
            phase_train.score(),
            phase_val.score(),
            phase_test.score() if phase_test else 0
        ]
        if dist.is_main_process():
            logger.warning(
                f"=== END EPOCH {epochi} - loss {'/'.join(map(lambda x: f'{x:.3f}',loss_msg))}, score {'/'.join(map(lambda x: f'{x:.3f}',score_msg))} ==="
            )

        # check if ckpt is need
        need_save_period = epochi % checkpoint_save_period == 0

        if (direction=='maximize' and score >= best_score) or (direction=='minimize' and score <=best_score):
            best_score=score
            need_save_best=True
        else:
            need_save_best=False
        if dist.is_main_process() and save_checkpoint:
            # force saving ckpt in diagnose mode
            if diagnose_mode:
                need_save_best=True
                need_save_period=True
            state={
                'cur_epochi':epochi,
                'best_score':best_score,
            }
            if need_save_period:recipe.save_ckpt(trial_file.get_checkpoint_slot(epochi),extra_state=state)
            if need_save_best:recipe.save_ckpt(trial_file.get_best_checkpoint_slot(),extra_state=state)

        if dist.is_initialized():
            dist.barrier()
        
        trial.report(score,epochi)
        # early stop
        if enable_prune and trial.should_prune():
            dist.barrier()
            # TODO: leave a flag for main thread to determine the status
            break
        
        instant_cmd=_check_instant_cmd()
        if instant_cmd=='prune':
            logger.warning("breaking epoch")
            break
        elif instant_cmd=='pdb':
            logger.warning("entering pdb")
            pdb.set_trace()
    
    recipe.end(staff)
    return best_score


def _prepare_dataloader_for_epoch(dataloader:DataLoader,cur_epochi:int):
    if isinstance(dataloader.sampler,DistributedSampler):
        dataloader.sampler.set_epoch(cur_epochi)

def _end_phase(staff:ModelStaff,tbwriter:SummaryWriter,epochi:int,phase:PhaseEnv):
    staff.update_tb(epochi,phase,tbwriter)
    for k,v in phase.custom_probes.items():
        average_v=v.average()
        if dist.is_main_process():
            logger.info(f"| {k}: {average_v}")
    
    if dist.is_main_process():
        logger.info(
            f"|| end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score()} ||"
        )
    else:
        logger.debug(
            f"|| RANK {dist.get_rank()} end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score()} ||"
        )

def _check_instant_cmd():
    logger.debug("checking instant cmd")
    if os.path.exists('instant_cmd'):
        with open('instant_cmd','r') as f:
            full_cmd=list(map(lambda x:x.strip(),f.readlines()))
        if len(full_cmd)==0:
            os.remove('instant_cmd')
        else:
            with open('instant_cmd','w') as f:
                f.writelines(full_cmd[1:])
            full_cmd=full_cmd[0]
            if full_cmd in ['prune','pdb']:
                logger.warning(f"received command `{full_cmd}`")
                return full_cmd
            else:
                logger.warning(f"unknown command '{full_cmd}'")
                return None
    else:
        return None


def _init_logger(comment:str,verbose:bool):
    # logger file
    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    if not dist.is_initialized():
        set_logger_file(f"logs/log_{comment}.log")
    else:
        set_logger_file(f"logs/log_{comment}@{dist.get_rank()}.log")

    if verbose or not dist.is_main_process():
        set_verbosity(logging.DEBUG)
    else:
        set_verbosity(logging.INFO)
    logger.info("[BOOTSTRAP] initialized logger")