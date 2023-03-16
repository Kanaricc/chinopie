import sys
sys.path.append('..')
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from chinopie import TrainHelper
from chinopie.datasets.fakeset import FakeRandomSet

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1=nn.Linear(10,10)
    
    def forward(self,x:Tensor):
        return self.fc1(x)

def train(batch_size:int=16):
    helper=TrainHelper('deps',10,batch_size,True,True,None,'114514',diagnose=True)

    trainset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
    valset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
    testset=FakeRandomSet(torch.zeros(10),torch.zeros(10))
    trainloader=DataLoader(trainset,batch_size)
    valloader=DataLoader(valset,batch_size)
    testloader=DataLoader(testset,batch_size)

    model=Model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)

    helper.register_dataset(trainset,trainloader,valset,valloader)
    helper.register_test_dataset(testset,testloader)

    helper.register_probe('a')

    with helper.section_checkpoint_load() as s:
        pass

    helper.ready_to_train()

    for epochi in helper.range_epoch():
        with helper.phase_train() as phase:
            model.train()
            for batchi,x in phase.range_data():
                inputs:Tensor=x['input']
                targets:Tensor=x['target']

                optimizer.zero_grad()
                outputs=model(inputs)
                loss=F.l1_loss(outputs,targets)
                phase.validate_loss(loss)
                loss.backward()
                optimizer.step()

                phase.update_loss(loss,batch_size)
            
            phase.update_probe('a',1.)
            phase.end_phase(0.)

        with helper.phase_val() as phase:
            model.eval()
            for batchi,x in phase.range_data():
                inputs:Tensor=x['input']
                targets:Tensor=x['target']

                with torch.no_grad():
                    outputs=model(inputs)
                loss=F.l1_loss(outputs,targets)
                phase.validate_loss(loss)

                phase.update_loss(loss,batch_size)
            phase.update_probe('a',2.)
            phase.end_phase(0.)
        with helper.phase_test() as phase:
            model.eval()
            for batchi,x in phase.range_data():
                inputs:Tensor=x['input']
                targets:Tensor=x['target']

                with torch.no_grad():
                    outputs=model(inputs)
                loss=F.l1_loss(outputs,targets)
                phase.validate_loss(loss)

                phase.update_loss(loss,batch_size)
            phase.update_probe('a',3.)
            phase.end_phase(0.)
        
        with helper.section_checkpoint_save():
            pass


if __name__=="__main__":
    TrainHelper.auto_bind_and_run(train)