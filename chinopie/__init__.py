name="chinopie"

import chinopie
from .bootstrap import TrainBootstrap
from .modelhelper import ModelStaff,PhaseEnv,HyperparameterManager
from .recipe import ModuleRecipe,EvaluationRecipe,TrainingRecipe,ModelStateKeeper
from .utils import copy_model,freeze_model,set_train,set_eval,any_to,validate_loss,validate_tensor,get_env,breakpoint
from .logging import get_logger
from .iddp import DistributedSampler

logger= get_logger(__name__)