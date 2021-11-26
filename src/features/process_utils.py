import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def infodate(string):
    newstring = string.replace("T"," ").replace("Z","")
    date_time_obj = datetime.strptime(newstring[2:], '%y-%m-%d %H:%M:%S')
    date_time_obj2 = date_time_obj - timedelta(hours=3)

    ratio = (date_time_obj.hour + date_time_obj.minute/60 + date_time_obj.second/3600)/24
    jour=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date_time_obj2.weekday()]
    numjour=date_time_obj2.timetuple().tm_yday
    annee=date_time_obj.year


    return [ratio, jour, numjour,annee]

def convertdate(string):
    newstring = string.replace("T"," ").replace("Z","")
    return pd.to_datetime(newstring)


def datetimeobj(date_time_obj):
    date_time_obj2 = date_time_obj - timedelta(hours=3)

    ratio = (date_time_obj.hour + date_time_obj.minute/60 + date_time_obj.second/3600)/24
    jour=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'][date_time_obj2.weekday()]
    numjour=date_time_obj2.timetuple().tm_yday
    annee=date_time_obj.year


    return [ratio, jour, numjour,annee]