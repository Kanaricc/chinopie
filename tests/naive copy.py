import sys
from typing import Dict
sys.path.append('..')
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from chinopie import HyperparameterManager, ModelStaff,ModuleRecipe,TrainBootstrap
from chinopie.datasets.fakeset import FakeRandomSet

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1=nn.Linear(10,10)
    
    def forward(self,x:Tensor):
        return self.fc1(x)
    

class Recipe2(ModuleRecipe):
    def __init__(self):
        super().__init__()
    
    def prepare(self, hp: HyperparameterManager, staff: ModelStaff):
        trainset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        valset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        testset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        trainloader=DataLoader(trainset,hp.suggest_int('batch_size',1,10))
        valloader=DataLoader(valset,hp.suggest_int('batch_size',1,10))
        testloader=DataLoader(testset,hp.suggest_int('batch_size',1,10))
        staff.reg_dataset(trainset,trainloader,valset,valloader)
        staff.reg_test_dataset(testset,testloader)

        model=Model()
        staff.reg_model(model)
    
    def set_optimizers(self, model, hp: HyperparameterManager) -> Optimizer:
        return torch.optim.AdamW(model.parameters(),lr=hp.suggest_float('lr',1e-5,1e-1,log=True))
    
    def forward(self, data) -> Tensor:
        return self.model(data['input'])
    
    def cal_loss(self, data, output) -> Tensor:
        return F.l1_loss(output,data['target'])
    
    def report_score(self, phase: str) -> float:
        return 0.0


if __name__=="__main__":
    tb=TrainBootstrap(disk_root='deps',num_epoch=10,load_checkpoint=True,save_checkpoint=True,comment=None)
    tb.hp.reg_float('lr')
    tb.hp.reg_int('batch_size')
    tb.optimize(Recipe2(),'maximize',0,1)