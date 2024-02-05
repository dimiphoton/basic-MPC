
from sklearn.base import BaseEstimator
import modelsubfunctions as modelsub
from scipy.integrate import odeint
from stateclasses import *
import numpy as np
import pandas as pd
import os
import os.path
import shutil

class toymodel:
    def __init__(self):
        self.name='toymodel'
    def predict(self,inputvalue):
        return inputvalue

class model0(BaseEstimator):
    def __init__(self,
        PVload2heat=1,
        PVProd2heat=1,
        C=1000,
        Crad=60,
        Rwall=1,
        Rrad=1,
        valverange=3,
        valvetrigger=1,
        Prad=1000
        ):
        self.PVload2heat=PVload2heat
        self.PVProd2heat=PVProd2heat
        self.C=C
        self.Crad=Crad
        self.Rwall=Rwall
        self.Rrad=Rrad
        self.valverange=valverange
        self.valvetrigger=valvetrigger
    
    def test(self,state):
        state.T=state.T+1

    def valveopen(self,T,Tset):
        """""
        returns the valve status as boolean
        T = measured
        Tset = set
        T2 = margin below which flow > 0.0
        T1 = margin below which flow = 1.0 
        """""
        if T >= (Tset-self.valvetrigger):
            return False
        else:
            return True

    def valveratio(self,T,Tset):
        """
        returns the flow command as a float
        T = measured
        Tset = set
        T2 = margin below which flow > 0.0
        T1 = margin below which flow = 1.0
        """
        if self.valveopen(T,Tset)==False:
            return 0.0
        else:
            return min(1, -(T-Tset+self.valvetrigger)/self.valverange)

    def initialize(self):
        return 0

    def heatflow(self,T,Tset,Text):
        """
        returns a list of flows
        from rad to the room 
        from ext to the room
        """

        #heat flow 1: radiator
        flow1 = self.valveratio(T,Tset)*(Trad-T)/self.Rrad
        flow2 = (Text-T)/self.Rwall

        return flow1, flow2

    


    def updateTrad(self,T,Trad,Tboiler,Tset):
        if self.valveopen(T,Tset)==True:
            return Tboiler
        else:
            return Trad
    def deriv(self,T,Trad,Tset,Text):
        """"
        dy/dt

        """

        flow1, flow2 = self.heatflow(T,Trad,Tset,Text)

        derivT=(flow1+flow2)/self.C
        
        if self.valveopen(T,Tset)==True:
            derivTrad=0
        else:
            derivTrad=-flow1/self.Crad


        return [derivT, derivTrad]

    def simpleintegrate(self,initialstate,Tboilerarray,Tsetarray,Textarray,timesteparray):
        """
        returns array Tzone, Trad
        """
        result=np.empty((len(timesteparray),2))

        [T,Trad]=initialstate
        for index,value in enumerate(timesteparray):
            result[index]=[T,Trad]
            T=result[index][0]+ value*self.deriv(T,Trad,Tsetarray[index],Textarray[index])[0]
            Trad=result[index][1]+ value*self.deriv(T,Trad,Tsetarray[index],Textarray[index])[1]
            print((T,Trad))
            Trad=self.updateTrad(T,Trad,Tboilerarray[index],Tsetarray[index])

        return result[:,0]

    def squarred_error(self,initialstate,Tmeasured,Tboilerarray,Tsetarray,Textarray,timesteparray):
        computed=self.simpleintegrate(initialstate,Tmeasured,Tboilerarray,Tsetarray,Textarray,timesteparray)
        return sum((computed-Tmeasured)**2)
    
    def train(self):
        #cur_path = os.path.dirname(__file__)
        #new_path = os.path.relpath('..\\data\\processed', cur_path)
        #with open("bigdf.csv") as file:
        #    df=pd.read_csv(file)

        os.getcwd()
        os.chdir("..") 
        os.chdir("..")
        os.chdir('data\\processed\\')
        df=pd.read_csv('bigdf.csv',index_col=[0,1])
        return df