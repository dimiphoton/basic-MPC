import numpy as np
import os
import pandas as pd

def load_data_multi_room():
    room_filenames = [("temperature_bathroom.csv", "bath"), 
                      ("temperature_bedroom_1.csv", "bed_1"), 
                      ("temperature_bedroom_2.csv", "bed_2"), 
                      ("temperature_bedroom_3.csv", "bed_3"),
                      ("temperature_diningroom.csv", "dining"), 
                      ("temperature_kitchen.csv", "kitchen"), 
                      ("temperature_livingroom.csv", "living")]

    df = pd.read_csv(os.path.join("data",os.path.join("raw_data", "temperature_heating_system.csv")))

    df_tmp = pd.read_csv(os.path.join("data", os.path.join("raw_data", "temperature_outside.csv")))
    df_tmp = df_tmp.rename(columns={"current_value": "outside_temperature"})
    df = df.merge(df_tmp, how="inner", left_on="time", right_on="time")

    for room_filename, name in room_filenames:
        df_tmp = pd.read_csv(os.path.join("data", os.path.join("raw_data", room_filename)))
        df_tmp = df_tmp.rename(columns={"current_value": name+"_temperature", "setpoint": name+"_setpoint"})
        df = df.merge(df_tmp, how="inner", left_on="time", right_on="time")

    return df

def load_data_toy():
    df = load_data_multi_room()
    df["house_temperature"] = (df["bath_temperature"] + df["bed_1_temperature"] + df["bed_2_temperature"] + df["bed_3_temperature"]+
                               df["dining_temperature"] + df["kitchen_temperature"] + df["living_temperature"])/7

    return df[["house_temperature", "outside_temperature"]]