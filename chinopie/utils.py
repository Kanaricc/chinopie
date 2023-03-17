from typing import Dict,Any,Optional,List
from prettytable import PrettyTable,PLAIN_COLUMNS
from datetime import datetime
from typing import Optional
from git.repo import Repo
from loguru import logger
import os

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

def check_gitignore():
    if not os.path.exists('.gitignore'):
        logger.warning("no gitignore found. try creating one.")
        with open('.gitignore','w') as f:
            pass
    ignore_list=[]
    with open('.gitignore','r') as f:
        ignore_list=f.readlines()
    
    check_list=[
        "/logs",
        "/opts"
    ]
    with open('.gitignore','a') as f:
        for item in check_list:
            if item not in ignore_list:
                f.writelines([item])
                logger.warning("`/logs` not found in .gitignore. appended it.")
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