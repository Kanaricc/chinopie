import subprocess
import shutil
from datetime import datetime
from typing import Optional
from git.repo import Repo
import git

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