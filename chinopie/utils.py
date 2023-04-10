from typing import Dict,Any,Optional,List
from prettytable import PrettyTable,PLAIN_COLUMNS
from datetime import datetime
from typing import Optional
from git.repo import Repo
from loguru import logger
from torch import nn,Tensor
import os

def copy_model(model:nn.Module):
    model_copy = type(model)() # get a new instance
    model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
    return model_copy

def freeze_model(model:nn.Module,eval:bool=True):
    for param in model.parameters():
        param.requires_grad_(False)
    if eval:
        model.eval()

def unfreeze_model(model:nn.Module):
    for param in model.parameters():
        param.requires_grad_(True)

def set_train(model:nn.Module,eval_on_nograd_module:bool=True):
    for mod in model.modules():
        has_require_grad=False
        for param in mod.parameters():
            has_require_grad|=param.requires_grad
        if not has_require_grad and eval_on_nograd_module:
            mod.eval()

def set_eval(model:nn.Module):
    model.eval()

def create_snapshot(comment:Optional[str]=None):
    date=datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name=f"{date}_{comment}"

    repo=Repo('.')
    base_branch=repo.active_branch.name
    g=repo.git
    
    g.add('.')
    g.stash()

    g.checkout('-b', branch_name)
    g.stash('apply')

    g.add('.')
    g.commit('-m', f'snapshot: {branch_name}')
    g.update_ref(f"refs/labrats/{branch_name}",branch_name)

    g.checkout(base_branch)
    g.branch('-D',branch_name)

    g.stash('pop')
    g.reset()

def check_gitignore(additional_list:List[str]=[]):
    if not os.path.exists('.gitignore'):
        logger.warning("no gitignore found. try creating one.")
        with open('.gitignore','w') as f:
            pass
    with open('.gitignore','r') as f:
        ignore_list=list(map(lambda x:x.strip(),f.readlines()))
    logger.trace(f"read .gitignore: {ignore_list}")
    
    check_list=[
        "/logs",
        "/opts"
    ]+additional_list
    has_newline=ignore_list[-1].strip()=='' if len(ignore_list)>0 else True
    with open('.gitignore','a') as f:
        for item in check_list:
            if item not in ignore_list:
                if not has_newline:
                    f.write("\n")
                    has_newline=True
                f.write(item)
                f.write("\n")
                logger.warning(f"`{item}` not found in .gitignore. appended it.")
    logger.info(".gitignore ignored logs and opts correctly")

def show_params_in_3cols(params:Optional[Dict[str,Any]]=None,name:Optional[List[str]]=None,val:Optional[List[Any]]=None):
    if params!=None:
        assert name==None and val==None
        name=list(params.keys())
        val=list(params.values())
    else:
        assert name!=None and val!=None
    while len(name)%3!=0:
        name.append('')
        val.append('')
    col_len=len(name)//3
    table=PrettyTable()
    table.set_style(PLAIN_COLUMNS)
    for i in range(3):
        table.add_column("params",name[i*col_len:(i+1)*col_len],"l")
        table.add_column("values",val[i*col_len:(i+1)*col_len],"c")
    return table


def any_to(data:Any,device:Any):
    if isinstance(data,Tensor):
        return data.to(device)
    elif isinstance(data,(list,tuple)):
        if len(data)==0:return type(data)()
        return type(data)(map(lambda x:any_to(x,device),data))
    elif isinstance(data,dict):
        res={}
        for k,v in data.items():
            res[k]=any_to(v,device)
        return res
    else:
        return data