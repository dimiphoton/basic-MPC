
import itertools
import numpy as np
import os
import pandas as pd
import os.path
from datetime import datetime, timedelta
import process_utils


room_filenames = [("temperature_bathroom.csv", "bathroom"), 
                      ("temperature_bedroom_1.csv", "bedroom_1"), 
                      ("temperature_bedroom_2.csv", "bedroom_2"), 
                      ("temperature_bedroom_3.csv", "bedroom_3"),
                      ("temperature_diningroom.csv", "diningroom"), 
                      ("temperature_kitchen.csv", "kitchen"), 
                      ("temperature_livingroom.csv", "livingroom")]
rooms=[tuple[1] for tuple in room_filenames]
    
heating_filenames=[("temperature_heating_system.csv","heating_system")]
solar_filenames = [("pv_production_load.csv","pv_production_load")]
external_filenames = [("temperature_outside.csv", "outside")]

p=os.getcwd()
p1=os.path.split(os.path.split(p)[0])[0]
path_raw=os.path.join(p1,'data','dataset1','raw')

df_bath = pd.read_csv(os.path.join(path_raw, "temperature_bathroom.csv"))
df_bath[["uniqueday_Id","intraday_Id"]]=df_bath.apply(process_utils.row_to_dayANDmin,axis=1, result_type="expand")
df_bath.set_index(["uniqueday_Id","intraday_Id"],inplace=True)

myindex=df_bath.index
del df_bath

tuple_zones=list(itertools.product(['T','Tset'],[tuple[1] for tuple in room_filenames]))+list(itertools.product(['T'],[tuple[1] for tuple in external_filenames]))




def makedf():

    """
    ici je crée le dataframe
    """
    DF=pd.DataFrame(columns=pd.MultiIndex.from_tuples(tuple_zones, names=["data", "zone"]))
    for room_filename, name in room_filenames+external_filenames+external_filenames:
        df_tmp = pd.read_csv(os.path.join(path_raw, room_filename))
        df_tmp[["uniqueday_Id","intraday_Id"]]=df_tmp.apply(process_utils.row_to_dayANDmin,axis=1, result_type="expand")
        df_tmp.set_index(["uniqueday_Id","intraday_Id"],inplace=True)
        #df_tmp = df_tmp.rename(columns={"current_value": name+"_temperature", "setpoint": name+"_setpoint"})
        #df = df.merge(df_tmp, how="inner", left_on='time', right_on='time')

        DF.loc[:,('T',name)]=df_tmp.loc[:,'current_value']
        if (room_filename,name) in room_filenames:
            DF.loc[:,('Tset',name)]=df_tmp.loc[:,'setpoint']
        del df_tmp


 

    
    ### ici j'ajoute Twater,Pwater
    df_tmp = pd.read_csv(os.path.join(path_raw, heating_filenames[0][0]))
    df_tmp[["uniqueday_Id","intraday_Id"]]=df_tmp.apply(process_utils.row_to_dayANDmin,axis=1, result_type="expand")
    df_tmp.set_index(["uniqueday_Id","intraday_Id"],inplace=True)
    DF.loc[:,'Twater']=df_tmp.loc[:,'water_temperature']
    DF.loc[:,'Pwater']=df_tmp.loc[:,'water_pressure']
    del df_tmp


    
    #ici j'ajoute time,timestep
    DF['time']=pd.to_datetime([tuple[0]+" "+tuple[1] for tuple in DF.index])
    #DF['timestep']=DF['time'].diff().dt.total_seconds().shift(-1)

    #ici j'ajout nextT
    for room in DF['T'].columns:
        DF.loc[:,('Tnext',room)]=DF.loc[:,('T',room)].shift(-1)

    DF.dropna(inplace=True)
    #ici j'ajoute les données moyennes
    for data in ['T','Tnext','Tset']:
        DF.loc[:,(data,'singleroom')]=DF[data].mean(axis=1)

    
    DF=DF.reset_index()
    #DF.set_index(["uniqueday_Id","sequence_Id","intraday_Id"],inplace=True)

    #ici j'ajoute l'activité solaire

    DF.index=DF['time']


    PVDF=pd.read_csv(os.path.join(path_raw,"pv_production_load.csv",))
    PVDF.columns = ['time', 'L1_PV', 'L2_PV', 'L3_PV', 'L1_Load', 'L2_Load','L3_Load']
    #PVDF[["uniqueday_Id","intraday_Id"]]=PVDF.apply(process_utils.row_to_dayANDmin,axis=1, result_type="expand")
    #PVDF.set_index(["uniqueday_Id","intraday_Id"],inplace=True)

    def sum_nonneg(x):
        return sum(max(i,0) for i in x)

    PVDF['solar']=PVDF[['L1_PV','L2_PV','L3_PV']].apply(lambda x: sum_nonneg(x),axis=1)
    PVDF['time']=PVDF['time'].apply(lambda x:process_utils.convertdate(x))
    PVDF.index=PVDF['time']
    PVDF=PVDF.reindex(index=DF.index,method='nearest')
    DF['solar']=PVDF['solar']

    DF['timestep']=DF['time'].diff().dt.total_seconds().shift(-1)


    #return DF

    DF.to_csv('dataset1.csv',index=False)
    #DF.to_csv('dataset1.csv',index=True,index_label=True)