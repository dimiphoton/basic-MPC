import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def infodate(string):
    """
    convert string to array of time infos
    """
    newstring = string.replace("T"," ").replace("Z","")
    date_time_obj = datetime.strptime(newstring[2:], '%y-%m-%d %H:%M:%S')
    date_time_obj2 = date_time_obj - timedelta(hours=3)

    ratio = (date_time_obj.hour + date_time_obj.minute/60 + date_time_obj.second/3600)/24
    jour=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date_time_obj2.weekday()]
    numjour=date_time_obj2.timetuple().tm_yday
    annee=date_time_obj.year


    return [ratio, jour, numjour,annee]

def convertdate(string):
    """
    convert string to timeobj
    """
    newstring = string.replace("T"," ").replace("Z","")
    return pd.to_datetime(newstring)


def datetimeobj(date_time_obj):
    """
    convert time object to array of time infos
    """
    date_time_obj2 = date_time_obj - timedelta(hours=3)

    ratio = (date_time_obj.hour + date_time_obj.minute/60 + date_time_obj.second/3600)/24
    jour=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date_time_obj2.weekday()]
    numjour=date_time_obj2.timetuple().tm_yday
    annee=date_time_obj.year


    return [ratio, jour, numjour,annee]

def time_to_dayId(obj):
    ratio, jour, numjour,annee=datetimeobj(obj)
    return str('Y')+str(annee) + str('D') + str(numjour)

def row_to_dayId(row):

    ratio, jour, numjour,annee=datetimeobj(obj)
    return str('Y')+str(annee) + str('D') + str(numjour)

def row_to_dayANDmin(row):
    badstring=row['time']
    goodstring=badstring.replace("T"," ").replace("Z","")
    return goodstring.split(" ")

def load_df(filename):
    df=pd.read_csv(filename,header=[0,1])
    MI=pd.MultiIndex.from_tuples(df,names=['data','zone'])
    MI2=[]
    for tuple in MI:
        if tuple[1].startswith('Unnamed:'):
            MI2.append((tuple[0],''))
        else:
            MI2.append(tuple)
    df.columns=pd.MultiIndex.from_tuples(MI2,names=['data','zone'])
    
    df.set_index(["uniqueday_Id","intraday_Id"],inplace=True)
    
    return df


