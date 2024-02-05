
import itertools
import numpy as np
import os
import pandas as pd
import os.path
from datetime import datetime, timedelta
import process_utils


p=os.getcwd()
p1=os.path.split(os.path.split(p)[0])[0]
path_raw=p1
#path_raw=os.path.join(p1,'data','dataset1','raw')


#tuple_zones=list(itertools.product(['T','Tset'],[tuple[1] for tuple in room_filenames]))+list(itertools.product(['T'],[tuple[1] for tuple in external_filenames]))


def makesplit(row):
    string=row['Datetime']
    string.strftime('%y-%m-%d'), string.strftime('%H:%M:%S')

def makedf():

    """
    ici je crée le dataframe
    """
    DF=pd.DataFrame(columns=pd.MultiIndex.from_tuples(tuple_zones, names=["data", "zone"]))
    df_tmp = pd.read_csv(path_raw)


    DF['timestep']=DF['time'].diff().dt.total_seconds().shift(-1)
    


    return DF

    #DF.to_csv('dataset1.csv',index=False)
    #DF.to_csv('dataset1.csv',index=True,index_label=True)