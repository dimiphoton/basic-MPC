#from sklearn.base import BaseEstimator
from datetime import *
import os
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import dataprocessing2
from sklearn.metrics import mean_squared_error

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



        if maison == 'singleroom':
            self.diczones={'heated':{'T':['singlezone_temperature'],'Tset':['singlezone_setpoint']},
                            'external':{'T':['outside_temperature']}}
            self.nbzones=2
                        
            self.C=100.0*np.ones(1)
            self.edges=[]
            self.edges=[(1,2)]
            self.Lambda=0.01*np.ones(len(self.edges))
            self.colnames=['singlezone_temperature','singlezone_setpoint']
            self.state=20*np.ones(1)

        elif maison=='multiroom':
            self.diczones={'heated':{'T':['bath_temperature','bed_1_temperature','bed_2_temperature',
                                        'bed_3_temperature','dining_temperature','kitchen_temperature','living_temperature'],
                                    'Tset':['bath_setpoint','bed_1_setpoint','bed_2_setpoint','bed_3_setpoint',
                                    'dining_setpoint','kitchen_setpoint','living_setpoint']},
                            'external':{'T':['outside_temperature']}}
            self.nbzones=9
            self.C=100.0*np.ones(8)

            self.edges=[]
            [self.edges.append((i,7)) for i in range(6) ]
            [self.edges.append((i,8)) for i in range(7) ]
            self.edges.append((6,7))
            self.edges.append((7,8))
            self.edges.append((4,5))
            self.edges.append((2,3))
            self.Lambda=0.01*np.ones(len(self.edges))
            self.colnames=['bath_temperature', 'bath_setpoint', 'bed_1_temperature',
                        'bed_1_setpoint', 'bed_2_temperature', 'bed_2_setpoint',
                        'bed_3_temperature', 'bed_3_setpoint', 'dining_temperature',
                        'dining_setpoint', 'kitchen_temperature', 'kitchen_setpoint',
                        'living_temperature', 'living_setpoint']
        



     #def setupdata(self,relative_data_path='..\\..\\data\\processed\\bigdf.csv',random_state=42):
        """
        this function takes processed dataframe and creates train/test dataframe attributes
        """
     #   df=self.retrievebigdf(relative_data_path)
     #   days=np.unique([index[0] for index in df.index])
     #   traindays,testdays=train_test_split(days,test_size=0.6, random_state=random_state)
     #   self.trainset=df.loc[traindays]
     #   self.testset=df.loc[testdays]
     #   self.traindays=traindays
     # self.testdays=testdays

    def setupdata(self,seed=42):
        #df=dataprocessing2.makedf()

        df=pd.read_csv('processed.csv',index_col=["uniqueday_Id","intraday_Id"])
        X=df[self.diczones['heated']['T']+self.diczones['heated']['Tset']+self.diczones['external']['T']+['timestep']]
        y=df[self.diczones['heated']['T']].shift(periods=-1)
        for dayIndex in df.index.levels[0]:
            temp=df.loc[dayIndex].index[(df.loc[dayIndex]['timestep']>1800) | (df.loc[dayIndex]['timestep'].isna())]
            if temp.size != 0:
                df.loc[dayIndex].drop(df.loc[dayIndex].index[df.loc[dayIndex].index >= temp[0]])
                X.loc[dayIndex]=X.loc[dayIndex].drop(X.loc[dayIndex].index[X.loc[dayIndex].index >= temp[0]])
                y.loc[dayIndex]=y.loc[dayIndex].drop(y.loc[dayIndex].index[y.loc[dayIndex].index >= temp[0]])
            
        days=np.unique([index[0] for index in df.index])
        self.split=train_test_split(days,test_size=0.6, random_state=seed)
        self.Xtrain=X.loc[self.split[0]]
        self.Xtest=X.loc[self.split[1]]
        self.ytrain=y.loc[self.split[0]]
        self.ytest=y.loc[self.split[1]]
        self.summerdays=None
        self.df=df



    def toy(self):
        return list(20*np.ones(len(self.zones)-1)), list(10*np.ones(1))
    
    def test(self, **params):
        for key,value in params.items():
            if key not in self.__dict__.keys():
                raise Exception('paramètre inconnu')
            
        

    def computeLambda(self):
        """compute the conductivity matrix between rooms"""
        self.matrix=np.zeros((self.nbzones, self.nbzones))
        for index,tuple in enumerate(self.edges):
            self.matrix[tuple[0]][tuple[1]]=self.Lambda[index]
            self.matrix[tuple[1]][tuple[0]]=self.Lambda[index]

    def heating(self,house_temperature,set_temperature):
        """
        returns the instant heating array given instant temperature states and commands
        """
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
            


    
    def thermalmodel(self, house_fullstate, exterior_temperature,timestep,set_temperature):
        """
        house_temperature: observed+obsered house rooms
        returns next state given current state, Text,timestep
        """

        zones=np.concatenate((house_fullstate,exterior_temperature))
        measured_temperature=house_fullstate[:-1]
        dH=np.zeros(self.nbzones)
        dQ=np.zeros(self.nbzones)


        # PARTIE CONDUCTION
        self.computeLambda()
        for idx1,T1 in enumerate(zones):
            for idx2,T2 in enumerate(zones):
                dQ[idx1] += (T2-T1)*self.matrix[idx1,idx2]
        dH += np.multiply(dQ,timestep)


        #PARTIE RADIATEURS
        dH[:-2] += np.multiply(self.heating(measured_temperature,set_temperature),timestep)


        #ENTHALPIE
        #dH=timestep*np.add(dQ,dQheating)

        #conversion température
        house_fullstate = house_fullstate + dH[:-1]*np.reciprocal(self.C)

        return house_fullstate

    def daysimulate(self, Text_array, Tset_array,timestep_array):
        """
        returns a temperature array given Text,Tset and timestep arrays
        """
        # ESSAYER IF IS NOT NONE


            
        #for key,value in params.items():
        #        self.key=value
        house_temperature=np.empty(((Text_array.size,self.nbzones-1)))
        house_temperature[0]=[ 15 for i in range(self.nbzones-1) ]
        for i, step in enumerate(timestep_array):
            house_temperature[i] = self.thermalmodel(house_temperature[i], Text_array[i], step, set_temperature=Tset_array[i])

    # Observation model:
    # house temperature + N(0.25, noise_var)
        obs_noise = np.exp(self.log_f) * self.obs_noise
        obs_temperature = house_temperature + 0.25+(obs_noise ** 0.5) * np.random.randn(*house_temperature.shape)

        return house_temperature[:], obs_temperature[:]

    def plotndays(n):
        """
        returns a plot with some trajectories
        """
        pass


    def day_squarred_error(self,X,y,daylabel):

        """
        this function returns the squarred error between prediction and measured value
        """
        try:
            Text_array=X.loc[daylabel][self.diczones['external']['T']].to_numpy()
            timestep_array=X.loc[daylabel]['timestep'].to_numpy()
            recorded=y.loc[daylabel][self.diczones['heated']['T']].to_numpy()
            Tset_array=X.loc[daylabel][self.diczones['heated']['Tset']].to_numpy()

            predicted=self.daysimulate(Text_array,Tset_array,timestep_array)[0][:,0:len(self.diczones['heated']['T'])]

            return mean_squared_error(predicted,recorded)*np.shape(predicted)[0]/1000000

        except:
            print("probleme avec l'index "+daylabel)

    def set_squarred_error(self,X,y):
        error=0
        for daylabel,daydf in X.iterrows():
            error+=self.day_squarred_error(X,y,index)
        return error


    def log_likelihood(self,theta, out_temperature, obs_temperature):
        self.R, self.Lambda, self.log_f = theta
        house_temperature, _ = self.simulate(theta, out_temperature)
        noise_var = np.exp(self.log_f) * 0.1
        self.MLE=np.sum(norm.logpdf(obs_temperature, loc=house_temperature+0.25, scale=noise_var ** 0.5))
        return self.MLE



    








