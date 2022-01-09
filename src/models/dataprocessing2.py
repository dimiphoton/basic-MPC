import numpy as np
import os
import pandas as pd
import os.path
from datetime import datetime, timedelta
import process_utils




def makedf():
    p=os.getcwd()
    p1=os.path.split(os.path.split(p)[0])[0]
    path_raw=os.path.join(p1,'data','raw')
    room_filenames = [("temperature_bathroom.csv", "bath"), 
                      ("temperature_bedroom_1.csv", "bed_1"), 
                      ("temperature_bedroom_2.csv", "bed_2"), 
                      ("temperature_bedroom_3.csv", "bed_3"),
                      ("temperature_diningroom.csv", "dining"), 
                      ("temperature_kitchen.csv", "kitchen"), 
                      ("temperature_livingroom.csv", "living")]

    df = pd.read_csv(os.path.join(path_raw, "temperature_heating_system.csv"))

    df_tmp = pd.read_csv(os.path.join(path_raw,"temperature_outside.csv"))
    df_tmp = df_tmp.rename(columns={"current_value": "outside_temperature"})
    df = df.merge(df_tmp, how="inner", left_on='time', right_on='time')

    for room_filename, name in room_filenames:
        df_tmp = pd.read_csv(os.path.join(path_raw, room_filename))
        df_tmp = df_tmp.rename(columns={"current_value": name+"_temperature", "setpoint": name+"_setpoint"})
        df = df.merge(df_tmp, how="inner", left_on='time', right_on='time')
    
    #
    #df['Id_D']=df.apply(process_utils.row_to_dayId,axis=1)
    
    #df['dayId,intradayId']=df.apply(process_utils.row_to_dayANDmin,axis=1)
    
    #for index,row in df.iterrows():
    #    ratio, jour, numjour,annee = process_utils.datetimeobj(row['time'])
    #    row['Id_D'] = str('Y')+str(annee) + str('D') + str(numjour)

    #df=df.set_index(['dayId','intradayId'])
    df[["uniqueday_Id","intraday_Id"]]=df.apply(process_utils.row_to_dayANDmin,axis=1, result_type="expand")
    df['time']=df['time'].apply(lambda x: process_utils.convertdate(x))
    df['timestep']=df['time'].diff().dt.total_seconds().shift(-1)
    df["singlezone_temperature"] = (df["bath_temperature"] + df["bed_1_temperature"]+
                                df["bed_2_temperature"] + df["bed_3_temperature"]+ 
                                df["dining_temperature"] + df["kitchen_temperature"] + 
                                df["living_temperature"])/7
    df["singlezone_setpoint"] = (df["bath_setpoint"] + df["bed_1_setpoint"]+
                                    df["bed_2_setpoint"] + df["bed_3_setpoint"]+ 
                                    df["dining_setpoint"] + df["kitchen_setpoint"] + 
                                    df["living_setpoint"])/7


    df['dailyrecord_Id'] = pd.Series(dtype='int')
    df.set_index(["uniqueday_Id","intraday_Id"],inplace=True)
    for index0 in df.index.levels[0]:
        recordId=int(0)
        for index1 in df.loc[index0].index:
            myindex=(index0,index1)
            if df.loc[myindex,'timestep']<1800:
                df.loc[myindex,'dailyrecord_Id']=recordId
            else:
                recordId +=1
                df.loc[myindex,'dailyrecord_Id']=recordId


    df.to_csv('processed.csv',index=["uniqueday_Id","dailyrecord_Id","intraday_Id"])
    

def fill(df,myindex,maxtimestep,recordId):
    if df.loc[myindex,'timestep']<maxtimestep:
        df.loc[myindex,'dailyrecord_Id']=recordId
    elif df.loc[df.loc[index0].index[num+1],'timestep']<maxtimestep:
        fill(df,myindex,maxtimestep,recordId+1)

    