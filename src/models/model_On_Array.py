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


        self.obs_model=False
 
        self.param=dict()
        self.magnitude=dict()
        self.param['temperature_noise']=np.array([0])
        self.param['obs_noise']=np.array([0.1])
        self.param['Qfamily']=np.array([0.5])
        self.magnitude['Qfamily']=1000
        self.param['log_f'] =np.array([1])
        self.param['valvetrigger']=np.array([2])
        self.param['valverange']=np.array([5])


        self.guessPrad=1.0
        self.guessC=1.0
        self.guessLambda=1.0
        



        if maison == 'singleroom':
            self.diczones={'heated':{'T':['singlezone_temperature'],'Tset':['singlezone_setpoint']},
                            'external':{'T':['outside_temperature']}}
            self.nbheated=1
            self.nbextra=0
            self.nbzones=2
                        
            self.edges=[]
            self.edges=[(0,1)]
            self.colnames=['singlezone_temperature','singlezone_setpoint']

        elif maison=='multiroom':
            self.diczones={'heated':{'T':['bath_temperature','bed_1_temperature','bed_2_temperature',
                                        'bed_3_temperature','dining_temperature','kitchen_temperature','living_temperature'],
                                    'Tset':['bath_setpoint','bed_1_setpoint','bed_2_setpoint','bed_3_setpoint',
                                    'dining_setpoint','kitchen_setpoint','living_setpoint']},
                            'external':{'T':['outside_temperature']}}
            self.nbheated=7
            self.nbextra=1
            self.nbzones=9
            

            self.edges=[]
            [self.edges.append((i,7)) for i in range(6) ]
            [self.edges.append((i,8)) for i in range(7) ]
            self.edges.append((6,7))
            self.edges.append((7,8))
            self.edges.append((4,5))
            self.edges.append((2,3))
            self.colnames=['bath_temperature', 'bath_setpoint', 'bed_1_temperature',
                        'bed_1_setpoint', 'bed_2_temperature', 'bed_2_setpoint',
                        'bed_3_temperature', 'bed_3_setpoint', 'dining_temperature',
                        'dining_setpoint', 'kitchen_temperature', 'kitchen_setpoint',
                        'living_temperature', 'living_setpoint']
        


        self.param['Lambda']=self.guessLambda*np.ones(len(self.edges))
        self.magnitude['Lambda']=0.001
        self.param['C']=self.guessC*np.ones(self.nbheated+self.nbextra)
        self.magnitude['C']=1000000
        self.param['Prad']=self.guessPrad*np.ones(self.nbheated)
        self.magnitude['Prad']=1000
        self.param2optimize=['log_f','Lambda','C','Prad']
        self.update_theta()



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

    def update_theta(self):
        """
        updates the theta attribute (array) from param dic and param2optimize liste
        """
        tuple = [self.param[key] for key in self.param2optimize]
        self.theta= np.concatenate(tuple)

            

    def set_theta(self,theta):
        """
        takes theta array and updates the param attributes
        """
        size_list = [len(self.param[key]) for key in self.param2optimize]
        split = np.split(theta,np.cumsum(size_list)[:-1])

        for index,key in enumerate(self.param2optimize):
            self.param[key] = split[index]


    def setupdata(self,seed=42):
        #df=dataprocessing2.makedf()

        df0=pd.read_csv('processed.csv')
        df0.drop(df0[df0['timestep']>=1800].index,inplace=True)
        df=df0.set_index(["uniqueday_Id","dailyrecord_Id","intraday_Id"])
        df.sort_index
        X=df[self.diczones['heated']['T']+self.diczones['heated']['Tset']+self.diczones['external']['T']+['timestep']]
        y=df[self.diczones['heated']['T']].shift(periods=-1)
        
        df.drop(df.tail(1).index,inplace=True)
        X.drop(X.tail(1).index,inplace=True)
        y.drop(y.tail(1).index,inplace=True)

        days=np.unique([index[0] for index in df.index])
        self.split=train_test_split(days,test_size=0.3, random_state=seed)
        self.Xtrain=X.loc[self.split[0]].sort_index()

        self.Xtest=X.loc[self.split[1]].sort_index()
        self.ytrain=y.loc[self.split[0]].sort_index()
        self.ytest=y.loc[self.split[1]].sort_index()
        self.summerdays=None
        self.df=df.sort_index()
        self.X=X.sort_index()
        self.y=y.sort_index()



    def toy(self):
        return list(20*np.ones(len(self.zones)-1)), list(10*np.ones(1))
    
    def test(self, **params):
        for key,value in params.items():
            if key not in self.__dict__.keys():
                raise Exception('paramètre inconnu')
            
        

    def computeLambda(self):
        """compute the conductivity matrix between rooms"""
        self.Lmatrix=np.zeros((self.nbzones, self.nbzones))
        for index,tuple in enumerate(self.edges):
            self.Lmatrix[tuple[0]][tuple[1]]=self.magnitude['Lambda'] * self.param['Lambda'][index]
            self.Lmatrix[tuple[1]][tuple[0]]=self.magnitude['Lambda'] * self.param['Lambda'][index]

    def heating(self,house_temperature,set_temperature):
        """
        returns the instant heating array given instant temperature states and commands
        """
        diff_array=np.subtract(set_temperature,house_temperature)-self.param['valvetrigger'][0]
        command_array=np.ones(len(diff_array))
        for index,value in enumerate(diff_array):
            if value < 0:
                command_array[index]=0.0
            elif value > self.param['valverange'][0]:
                command_array[index]=1.0
            else:
                command_array[index]=1* value/(self.param['valverange'][0])
        
        return self.magnitude['Prad']*self.param['Prad']*command_array
            


    
    def thermalmodel(self, state, Text, Tset, timestep,debug=False):
        """
        house_temperature: observed+obsered house rooms
        returns next state given current state, Text,timestep
        """

        zones=np.concatenate((state, Text))

        dH=np.zeros(self.nbzones)
        dQ=np.zeros(self.nbzones)

        if debug==True:
            print(zones)
            print(dH)
            print(dQ)

        # PARTIE CONDUCTION
        self.computeLambda()
        for idx1,T1 in enumerate(zones):
            for idx2,T2 in enumerate(zones):
                dQ[idx1] += (T2-T1)*self.Lmatrix[idx1,idx2]

        for i in range(self.nbheated+self.nbextra):
            dH[i] += timestep*dQ[i]


        #PARTIE RADIATEURS

        heating_array=self.heating(state[:self.nbheated],Tset)

        for i in range(self.nbheated):
            dH[i] += timestep*heating_array[i]



        #ENTHALPIE
        #dH=timestep*np.add(dQ,dQheating)

        #conversion température
        fulldT=dH[0:self.nbheated+self.nbextra]*np.reciprocal(self.magnitude['C'] * self.param['C'])
        state[0:self.nbheated+self.nbextra] += fulldT

        #ajout bruit thermique
        if self.param['temperature_noise'][0] >0.0:
            state += (self.param['temperature_noise'][0]**0.5)*np.random.randn(self.nbzones)



        
        if debug==True:
            return state,dQ,dH
        if debug==False:
            return state

    

    def daysimulate(self, initialState,Text_array, Tset_array,timestep_array,save=False):
        """
        returns a temperature array given Text,Tset and timestep arrays
        """
        # ESSAYER IF IS NOT NONE


            
        #for key,value in params.items():
        #        self.key=value

        state=np.zeros(((Text_array.size+1,self.nbheated+self.nbextra)))
        #Q=np.zeros(((Text_array.size,self.nbzones)))
        #dH=np.zeros(((Text_array.size,self.nbzones)))
        state[0]=initialState
        for i in range(len(timestep_array)):
            state[i+1] = self.thermalmodel(state[i], Text_array[i], Tset_array[i], timestep_array[i])

        np.delete(state,0)

        if save==True:
            np.savetxt('T.csv', state, fmt='%1.1f')

        if self.obs_model==False:
            return state[1:]

        if self.obs_model==True:
    # Observation model:
    # house temperature + N(0.25, noise_var)
            obs_noise = np.exp(self.param['log_f'][0]) * self.param['obs_noise'][0]
            obs_temperature = state + 0.25+(obs_noise ** 0.5) * np.random.randn(*state.shape)

        return obs_temperature[1:]

    def plotndays(n):
        """
        returns a plot with some trajectories
        """
        pass

    


    def level1_pred(self,X,y,recordingIndex=('2020-05-24', 0.0), debug=False):

        """
        this function takes a recording and returns a tuple (record mse, record size)
        the squarred error between prediction and measured value
        """
        try:
            #initial state
            initial_heated=np.array(X.loc[recordingIndex][self.diczones['heated']['T']].to_numpy()[0])
            initial_notheated=np.mean(initial_heated)*np.ones(self.nbextra)
            initialState=np.concatenate((initial_heated,initial_notheated))
            #Text array
            Text_array=X.loc[recordingIndex][self.diczones['external']['T']].to_numpy()
            #Tset array
            Tset_array=X.loc[recordingIndex][self.diczones['heated']['Tset']].to_numpy()
            #timestep
            timestep_array=X.loc[recordingIndex]['timestep'].to_numpy()
            recorded=y.loc[recordingIndex][self.diczones['heated']['T']].to_numpy()
            
            predicted=self.daysimulate(initialState, Text_array, Tset_array, timestep_array, save=False)[:,0:len(self.diczones['heated']['T'])]
            #predicted=self.daysimulate(initialState, Text_array, Tset_array, timestep_array, save=False)[:,0:len(self.diczones['heated']['T'])]

            #return mean_squared_error(predicted,recorded)*len(predicted)/(np.shape(predicted)[0] * 10000)

            if debug==False:
                return predicted,recorded
        #except:
        #    if debug==True:
        #        return initialState, Text_array, Tset_array, timestep_array
        #    print('ERREUR au niveau de ', recordingIndex)
        except Exception as e:
            print(e)

    def level0_pred(self,X,y):
        """
        returns predicted and recorded arrays for the whole dataset
        """
        myindex=[]
        for idx in list(X.index):
            temp=(idx[0],idx[1])
            if temp not in myindex:
                myindex.append(temp)

        error_array=np.empty((len(myindex),1), dtype=np.float64)
        
        #for index,recordingIndex in enumerate(myindex):
        #    if self.level1_error(X,y,recordingIndex) != None:
        #        error_array[index]=self.level1_error(X,y,recordingIndex)
        
        predicted=np.empty(shape=(0,self.nbheated))
        recorded=np.empty(shape=(0,self.nbheated))
        for index,recordingIndex in enumerate(myindex):
            p,r = self.level1_pred(X,y,recordingIndex)
            predicted=np.append(predicted,p,axis=0)
            recorded=np.append(recorded,r,axis=0)
            

        return predicted,recorded
        #return np.sum(error_array)

    def mse(self,theta,X,y):

        self.set_theta(theta)
        p,r=self.level0_pred(X,y)

        return mean_squared_error(p,r)



    def log_likelihood(self,theta, out_temperature, obs_temperature):
        self.R, self.Lambda, self.log_f = theta
        house_temperature, _ = self.simulate(theta, out_temperature)
        noise_var = np.exp(self.log_f) * 0.1
        self.MLE=np.sum(norm.logpdf(obs_temperature, loc=house_temperature+0.25, scale=noise_var ** 0.5))
        return self.MLE



    








