import sys
sys.path.append('..')
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from chinopie import TrainHelper,ModuleRecipe
from chinopie.modelhelper import TrainBootstrap
from chinopie.datasets.fakeset import FakeRandomSet

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1=nn.Linear(10,10)
    
    def forward(self,x:Tensor):
        return self.fc1(x)


class Recipe1(ModuleRecipe[Model,torch.optim.AdamW]):
    def __init__(self):
        super().__init__()
    
    
    def prepare(self, helper: TrainHelper):
        trainset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        valset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        testset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
        trainloader=DataLoader(trainset,helper.suggest_int('batch_size',1,10))
        valloader=DataLoader(valset,helper.suggest_int('batch_size',1,10))
        testloader=DataLoader(testset,helper.suggest_int('batch_size',1,10))
        helper.register_dataset(trainset,trainloader,valset,valloader)
        helper.register_test_dataset(testset,testloader)

        model=Model()
        opti=torch.optim.AdamW(model.parameters(),lr=helper.suggest_float('lr',1e-6,1e-1,log=True))
        helper.reg_model(model)
        helper.reg_optimizer(opti)
    
    def forward(self, data) -> Tensor:
        return self.model(data['input'])
    
    def cal_loss(self, data, output: Tensor) -> Tensor:
        return F.l1_loss(output,data['target'])
    
    def report_score(self, phase: str) -> float:
        return 0.0
    


if __name__=="__main__":
    tb=TrainBootstrap(disk_root='deps',epoch_num=10,load_checkpoint=True,save_checkpoint=True,comment=None)
    tb.reg_float('lr')
    tb.reg_int('batch_size')
    tb.optimize(Recipe1(),1)