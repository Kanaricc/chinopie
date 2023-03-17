import subprocess
import shutil
from datetime import datetime
from git.repo import Repo
import git

def create_snapshot(comment:str=''):
    rat_name=datetime.now().strftime("%Y%m%d%H%M%S")
    repo=Repo('.')
    base_branch=repo.active_branch.name
    g=repo.git
    
    g.add('.')
    g.stash()

    g.checkout('-b', rat_name)
    g.stash('apply')

    g.add('.')
    g.commit('-m', f'rat {rat_name}: {comment}')
    g.update_ref(f"refs/labrats/{rat_name}",rat_name)

    g.checkout(base_branch)
    g.branch('-D',rat_name)

    g.stash('pop')
    g.reset()