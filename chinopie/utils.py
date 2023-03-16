from typing import Dict,Any,Optional,List
from prettytable import PrettyTable,PLAIN_COLUMNS

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