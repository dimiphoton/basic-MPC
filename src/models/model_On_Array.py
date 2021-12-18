from sklearn.base import BaseEstimator
import os
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split

class mainmodel():
    def __init__(self,maison='singleroom') -> None:

        self.valvetrigger=1
        self.temperature_noise=0
        self.obs_noise=0.1
        self.Prad=1000
        self.log_f =1


        self.T_25 = 1
        self.T_50 = 4
        self.T_100 = 7


        if maison == 'singlezone':
            self.zones=['room','exterior']
                        
            self.C=100.0*np.ones(1)
            self.edges=[]
            self.edges=[(1,2)]
            self.Lambda=0.01*np.ones(len(self.edges))

            self.state=20*np.ones(1)

        elif maison=='multizone':
            self.zones=['bath','bed1','bed2','bed3','dining','kitchen','living','hall','exterior']
            self.C=100.0*np.ones(8)

            self.edges=[]
            [self.edges.append((i,7)) for i in range(6) ]
            [self.edges.append((i,8)) for i in range(7) ]
            self.edges.append((6,7))
            self.edges.append((7,8))
            self.edges.append((4,5))
            self.edges.append((2,3))
            self.Lambda=0.01*np.ones(len(self.edges))


     def setupdata(self,relative_data_path='..\\..\\data\\processed\\bigdf.csv',random_state=42):
        """
        this function takes processed dataframe and creates train/test dataframe attributes
        """
        df=self.retrievebigdf(relative_data_path)
        days=np.unique([index[0] for index in df.index])
        traindays,testdays=train_test_split(days,test_size=0.6, random_state=random_state)
        self.trainset=df.loc[traindays]
        self.testset=df.loc[testdays]
        self.traindays=traindays
        self.testdays=testdays

    def toy(self):
        return list(20*np.ones(len(self.zones)-1)), list(10*np.ones(1))
    
    def test(self, **params):
        for key,value in params.items():
            if key not in self.__dict__.keys():
                raise Exception('paramètre inconnu')
            
        

    def computeLambda(self):
        self.matrix=np.zeros((len(self.zones), len(self.zones)))
        for index,tuple in enumerate(self.edges):
            self.matrix[tuple[0]][tuple[1]]=self.Lambda[index]
            self.matrix[tuple[1]][tuple[0]]=self.Lambda[index]

    def heating(self,house_temperature,set_temperature):
        array=np.subtract(set_temperature,house_temperature)
        a=np.zeros(len(array))
        for index,value in enumerate(array):
            if value < self.T_25:
                a[index]=0.0
            elif self.T_25 <= value < self.T_50:
                a[index]=0.25
            elif self.T_50 <= value < self.T_100:
                a[index]=0.50
            else:
                a[index]=1
        
        return self.Prad*a
            


    
    def thermalmodel(self, house_temperature, exterior_temperature,timestep,set_temperature=None):

        zones=house_temperature+exterior_temperature
        dH=np.zeros(len(house_temperature))
        dQ=np.zeros(len(self.zones))


        # PARTIE CONDUCTION
        self.computeLambda()
        for idx1,T1 in enumerate(zones):
            for idx2,T2 in enumerate(zones):
                dQ[idx1] += (T2-T1)*self.matrix[idx1,idx2]
        dH = dH + np.multiply(dQ[:-1],timestep)


        #PARTIE RADIATEURS
        if set_temperature !=None:
            dH = dH + np.multiply(self.heating(house_temperature,set_temperature),timestep)


        #ENTHALPIE
        #dH=timestep*np.add(dQ,dQheating)

        #conversion température
        house_temperature += dH[:-1]*np.reciprocal(self.C)

        return house_temperature

    def simulate(self,*theta, out_temperature, **params):
        for key,value in params.items():
            if key not in self.__dict__.keys():
                raise Exception('paramètre inconnu')

        self.Lambda, self.C, self.log_f = theta
            
        #for key,value in params.items():
        #        self.key=value
         house_temperature = np.zeros(len(out_temperature)+1)


        #INITIALIZATION
        



        #house_temperature = house_temperature + 300 * (exterior_temperature - house_temperature) / (R * C)







