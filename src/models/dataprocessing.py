import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import random

import sys
#sys.path.append('C:/Users/DimiP/Documents/GitHub/ULG/COURS/basic-MPC/data/raw')
#from features import process_utils
#reload_ext autoreload

import process_utils

p=os.getcwd()
p1=os.path.split(os.path.split(p)[0])[0]
path_raw=os.path.join(p1,'data','raw')

#création des dicts de dataframe

def makedf():
    p=os.getcwd()
    p1=os.path.split(os.path.split(p)[0])[0]
    path_raw=os.path.join(p1,'data','raw')

    room_filenames = [("temperature_bathroom.csv", "bathroom"), 
                      ("temperature_bedroom_1.csv", "bedroom_1"), 
                      ("temperature_bedroom_2.csv", "bedroom_2"), 
                      ("temperature_bedroom_3.csv", "bedroom_3"),
                      ("temperature_diningroom.csv", "diningroom"), 
                      ("temperature_kitchen.csv", "kitchen"), 
                      ("temperature_livingroom.csv", "livingroom"),
                      ("temperature_heating_system.csv","heating_system"),
                      ("temperature_outside.csv","outside"),
                      ("pv_production_load.csv","pv_production_load")]

    #misc_filenames = []

    data={}
    for file,roomid in room_filenames:
        print(roomid)
        data[roomid]=pd.read_csv(os.path.join(path_raw,file))
        data[roomid].index=data[roomid]['time'].apply(lambda x: process_utils.convertdate(x))
    print(data.keys())

    data['pv_production_load']=pd.read_csv(os.path.join(path_raw,"pv_production_load.csv",))
    data['pv_production_load'].columns = ['time', 'L1_PV', 'L2_PV', 'L3_PV', 'L1_Load', 'L2_Load','L3_Load']

    pv=data['pv_production_load'].to_numpy()
    heating=data['heating_system'].to_numpy()

    big_df = pd.DataFrame(index=data['kitchen'].index)

    date_array = [process_utils.datetimeobj(x) for x in big_df.index]

    df_add = pd.DataFrame(data=date_array,columns=['perc_D','week_D','number_D','year'],index=big_df.index)

    big_df = pd.concat([big_df,df_add], axis=1)



    mesT_df=pd.concat([data[key]['current_value'] for key in ['bathroom', 'bedroom_1', 'bedroom_2', 'bedroom_3',
    'diningroom', 'kitchen', 'livingroom']],axis=1)
    setT_df=pd.concat([data[key]['setpoint'] for key in ['bathroom', 'bedroom_1', 'bedroom_2', 'bedroom_3',
    'diningroom', 'kitchen', 'livingroom']],axis=1)


    big_df['multizone_mesT']=mesT_df.apply(list,axis=1)
    big_df['multizone_setT']=setT_df.apply(list,axis=1)
    big_df['outside_mesT']=data['outside']['current_value']
    #finaldf=big_df[big_df.index>pd.to_datetime('2020-05-24 19:40:03')]

    def sum_nonneg(x):
        return sum(max(i,0) for i in x)

    pv_df=data['pv_production_load'].reindex(index=big_df.index,method='nearest')
    heating_df=data['heating_system'].reindex(index=big_df.index,method='nearest')

    pv_df.drop_duplicates(subset='time')
    heating_df.drop_duplicates(subset='time')
    big_df['PV'] = pv_df[['L1_PV','L2_PV','L3_PV']].apply(lambda x: sum_nonneg(x),axis=1)
    big_df['PV_load'] = pv_df[['L1_Load','L2_Load','L3_Load']].apply(lambda x: sum_nonneg(x),axis=1)
    big_df['waterP']=heating_df['water_pressure']
    big_df['waterT']=heating_df['water_temperature']

    lst=big_df.index

    big_df['singlezone_mesT']=big_df['multizone_mesT'].apply(lambda x:[sum(x)/7])
    big_df['singlezone_setT']=big_df['multizone_setT'].apply(lambda x:[sum(x)/7])

    big_df['singlezone_nextT']=big_df['singlezone_mesT'].shift(-1)
    big_df['multizone_nextT'] = big_df['multizone_mesT'].shift(-1)


    emptydf=pd.DataFrame(index=big_df.index)
    emptydf['timestamp']=big_df.index
    big_df['timestep']=emptydf['timestamp'].diff().dt.total_seconds().shift(-1)


    big_df["Id_D"] = str('Y')+big_df['year'].astype(str) + str('D') + big_df['number_D'].astype(str)
    listeId=big_df["Id_D"].to_list()
    #df=big_df[big_df['timestep']<500].set_index(['Id_D','time'])

    #pd.DataFrame.to_csv
    return big_df
#train_ix, test_ix = train_test_split(df.index.levels[0], test_size=0.3)
#train = df.loc[train_ix]
#test = df.loc[test_ix]



