#from sklearn.base import BaseEstimator
from datetime import *
import os
from pyexpat import features
from statistics import mean
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import dataprocessing2
from sklearn.metrics import mean_squared_error
import itertools
import process_utils
import random



class mainmodel():
    def __init__(self,maison='singleroom',extrafeatures=dict()) -> None:

        self.noisymeasure=False
        self.param=dict()
        self.magnitude=dict()
        self.param['temperature_noise']=np.array([0.0])
        self.param['obs_noise']=np.array([0.1])
        self.param['log_f'] = np.array([1.0])
        self.param['valverange']=1
        self.guessPrad=1.0
        self.guessC=1.0
        self.guessLambda=1.0
        



        if maison == 'singleroom':

            self.diczones={'floor0':['singleroom'],
                            'floor1':[],
                            'outside':['outside'],
                            'ground':[]}
            self.edges=dict()
            self.edges['floor0-outside']=[(0,1),(0,1)]
            self.param['Lambda']=self.guessLambda*np.ones(1)


        elif maison=='multiroom1':

            self.diczones={'floor1':['bathroom', 'bedroom_1', 'bedroom_2', 'bedroom_3'],
                            'floor0':['diningroom','kitchen','livingroom'],
                            'outside':['outside'],
                            'ground':[]}
            self.nbrooms=7
            self.nbexternal=1
            self.nbzones=8
            self.rooms=['bathroom', 'bedroom_1', 'bedroom_2', 'bedroom_3','diningroom','kitchen','livingroom']

            self.edges=dict()
            self.edges['floor0-floor0']=list(itertools.permutations([4, 5, 6], 2))
            self.edges['floor1-floor1']=list(itertools.permutations([0, 1, 2, 3], 2))
            self.edges['floor0-floor1']=list(itertools.product([0,1,2,3],[4,5,6]))+list(itertools.product([4,5,6],[0,1,2,3]))
            self.edges['floor0-outside']=list(itertools.product([7],[4,5,6]))+list(itertools.product([4,5,6],[7]))
            self.edges['floor1-outside']=list(itertools.product([7],[0,1,2,3]))+list(itertools.product([0,1,2,3],[7]))

            self.param['Lambda']=self.guessLambda*np.ones(5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

        self.rooms=self.diczones['floor1']+self.diczones['floor0']
        self.external=self.diczones['outside']+self.diczones['ground']
        self.nbrooms=len(self.rooms)
        self.nbexternal=len(self.external)
        self.nbzones=self.nbrooms+self.nbexternal
        
        self.magnitude['Lambda']=0.0010
        self.param['C']=self.guessC*np.ones(self.nbrooms)
        self.magnitude['C']=1000000.0
        self.param['Prad']=self.guessPrad*np.array([1])
        self.magnitude['Prad']=1
        #self.param['ThetaRad1']=5.0*np.ones(self.nbrooms)
        #self.param['ThetaRad2']=1.0*np.ones(self.nbrooms)
        self.param['Tadd']=np.ones(self.nbrooms)
        self.magnitude['Tadd']=3
        #self.param['valverange']=np.array([5])
        self.magnitude['Pocc']=0
        self.param['Pocc']=np.ones(self.nbrooms)

        self.param2optimize=['Lambda','C','Prad']
        self.theta= np.concatenate([self.param[key] for key in self.param2optimize])   
        
    




    def update_theta(self):
        """
        updates the theta attribute (array) from param dic and param2optimize liste
        """
        #print('liste à concaténer',[self.param[key] for key in self.param2optimize])
        #print('concatenration',np.concatenate([self.param[key] for key in self.param2optimize]))
        #print('theta avant',self.theta)
        self.theta= np.concatenate([self.param[key] for key in self.param2optimize])
        #print('theta après',self.theta)

            



    def set_theta(self,theta):
        """
        takes theta array and updates the param attributes AND Lmatrix
        """
        size_list = [len(self.param[key]) for key in self.param2optimize]
        #print('size_list',size_list)
        split = np.split(theta,np.cumsum(size_list)[:-1])
        #print('split',split)
        for index,key in enumerate(self.param2optimize):
            self.param[key] = split[index]
        
        
        """computes the conductivity matrix between rooms"""
        self.Lmatrix=np.zeros((self.nbzones, self.nbzones))
        for index,key in enumerate(self.edges):
            
            for tuple in self.edges[key]:
                self.Lmatrix[tuple[0]][tuple[1]]=self.magnitude['Lambda'] * self.param['Lambda'][index]

        for i in range(self.nbzones):
            self.Lmatrix[i][i]=-np.sum(self.Lmatrix[i])

    def make_trajectory_dict(self,df,days):
        """
        returns dictionary 'data': list(sequence) of 2darray data
        """
        D=dict()
        for day in days:
            D[day]=dict()
            for nom, liste in self.features.items():
                D[day][nom]=df.loc[day,liste].to_numpy()
        return D

    def make_day_trajectory_dict(self,df,random_day=True):
        """
        returns dictionary 'data': 2darray data
        """
        if random_day:
            day=random.choice(df.index.levels[0])
        D=dict()
        for nom, liste in self.features.items():
            D[nom]=df.loc[day,liste].to_numpy()
        return D
        


    def make_transition_dict(self,df):
        """
        dictionary data:2d list
        """
        datadic=dict()
        for nom, liste in self.features.items():
            datadic[nom]=df.loc[:,liste].to_numpy()
        return datadic
        
            

    def setupdata(self,addswitch=True, addoccupancy=True, filename='dataset1.csv',seed=42):

        self.raw=process_utils.load_df(filename)

        self.features={'state':[('T',i) for i in self.rooms]\
                      ,'Tset':[('Tset',i) for i in self.rooms]\
                      ,'Twater':[('Twater','')]\
                      ,'switch':[('Rad',i) for i in self.rooms]\
                      ,'external':[('T',i) for i in self.external]\
                      ,'timestep':[('timestep','')]\
                      ,'occupancy':[('occupancy','')]\
                      ,'next_state':[('Tnext',i) for i in self.rooms]}

        self.mapping=dict()
        temp=0
        for key in self.features:
            self.mapping[key]=list(np.arange(temp,temp+len(self.features[key])))
            temp=temp+len(self.features[key])
        
        #self.raw['T']=self.raw['T'][self.rooms]
        if addswitch:
            for r in self.rooms:
                self.raw.loc[:,('Rad',r)]=(self.raw.loc[:,('Tset',r)]-self.raw.loc[:,('T',r)])>1
        
        if addoccupancy:
            self.raw.loc[:,('occupancy','')]=True

        self.traindays,self.testdays=train_test_split(np.unique(self.raw.index.get_level_values(0))[40:65],test_size=0.3,random_state=seed)
        self.traindic=self.make_trajectory_dict(self.raw,self.traindays)
        self.testdic=self.make_trajectory_dict(self.raw,self.testdays[0:10])

        lines=self.raw.index
        self.split_array=train_test_split(lines,test_size=0.5, random_state=seed)

        self.trainarray=self.make_transition_dict(self.raw.loc[self.split_array[0]].sort_index())
        self.testarray=self.make_transition_dict(self.raw.loc[self.split_array[1]].sort_index())

        #self.sample={key:value[0] for (key,value) in self.trainarray.items()}





        



    def heating2(self,state,Tset,Twater,switch):
        """
        returns the instant heating array given instant temperature states and commands
        """
        command=np.zeros([self.nbrooms])
        for i,s in enumerate(switch):
            if s == False:
                command[i]=0
            if s == True:
                #print('state[i]',state[i])
                #print('Twater',Twater)
                #print('param',self.param['ThetaRad1'])
                if state[i]-Twater+self.param['ThetaRad1'][i]>0:

                    coeff=min(self.param['valverange']*np.subtract(Tset[i],state[i]),1)
                    coeff=1
                    try:
                        heating_array=self.param['Prad'][i] *(np.log((state[i]-Twater+self.param['ThetaRad1'][i])/(state[i]-Twater+self.param['ThetaRad2'][i])))**(-1.33)
                        #heating_array=self.param['Prad'][i] * (state[i]-Twater+self.param['ThetaRad1'][i])/(state[i]-Twater+self.param['ThetaRad2'][i])
                    except:
                        heating_array=0
                    command[i]=coeff*heating_array
                else:
                    command[i]=0
        
        return command

    def heating(self,Tset,Twater,state):

        """
        returns instant heating array
        """
        Q=np.zeros_like(state)
        for index,value in enumerate(state):
            if Tset[index]>(state[index]-1):
                Q[index]=self.param['Prad']*(Twater-state[index])
            else:
                Q[index]= 0
        return Q

    def heating_occupancy(self,occupancy):
        return 0

            


    
    def thermalmodel(self,state=0,Tset=0,external=0,Twater=0 \
                    , occupancy=False, switch=True, timestep=300, debug=False):
        """
        returns next state given current state, Text,timestep
        """
        #zones=np.concatenate((state, Text,Tbasement))
        #print('state',state)
        dH=np.zeros_like(state)
        #print('dH ligne 264',dH)

        #print('dH au début',dH)

        # PARTIE CONDUCTION
        
        #print(np.matmul(self.Lmatrix,np.concatenate((state,Text,Tbasement)))[0:self.nbrooms])
        #print('state',state)
        #print('external',external)
        #print('axis',state.ndim-1)
        
        #state_extended=np.concatenate((state,external),axis=state.ndim-1)
        state_extended=np.concatenate((state.T,external.T)).T
        #print('state+ext ligne 277',state_extended)
        dH+=timestep*np.matmul(state_extended,self.Lmatrix).T[0:self.nbrooms].T
        #print('dH ligne 279',dH)

        #PARTIE RADIATEURS

        #print(state)
        #print(Tset)
        #print(Twater)
        #print(switch)
        dH+=timestep*self.heating(Tset,Twater,state)
        #print('dH ligne 288',dH)
        #PARTIE occupation
        #print(occupancy)
        if occupancy:
            dH+=timestep*self.magnitude['Pocc']*self.param['Pocc']*np.ones_like(state)
        #print('dH ligne 294', dH)


        #conversion température
        fulldT=dH*np.reciprocal(self.magnitude['C'] * self.param['C'])
        #print('fulldT ligne 298',fulldT)
        result= state + fulldT

        #ajout bruit thermique
        if self.noisymeasure:
            if self.param['temperature_noise'][0] >0.0:
                result += (self.param['temperature_noise'][0]**0.5)*np.random.randn(self.nbrooms)
        
        return result


    def predict1(self,dic):
        dic['predicted']=np.empty_like(dic['state'])
        for line,vector in enumerate(dic['state']):
            dic['predicted'][line]=self.thermalmodel(state=dic['state'][line]\
                                                     ,Tset=dic['Tset'][line]\
                                                     ,external=dic['external'][line]\
                                                     ,Twater=dic['Twater'][line]\
                                                     ,switch=dic['switch'][line]\
                                                     ,occupancy=dic['occupancy'][line]\
                                                     ,timestep=dic['timestep'][line])
        #print('rms',mean_squared_error(dic['predicted'],dic['next_state']))

    
    def res1(self,theta,dic):
        self.set_theta(theta)
        self.update_theta
        self.predict1(dic)
        return (dic['predicted']-dic['next_state']).flatten()

    def metric1(self,theta,dic):
        self.set_theta(theta)
        self.update_theta()
        #print(self.theta)
        #print(self.param)
        self.predict1(dic)
        

        return mean_squared_error(dic['next_state'],dic['predicted'])

    def dic_simulate(self,dic):
        """
        simulate on a dic of datadic
        """

        for day,data in dic.items():
            self.simulate(data)

    def metric2(self,theta,dic):
        """
        returns mse of double dic
        """
        self.set_theta(theta)
        self.update_theta()
        self.dic_simulate(dic)
        mse=[mean_squared_error(data['predicted'],data['state']) for day,data in dic.items()]
        return np.linalg.norm(mse)/len(mse)


    

    def simulate(self,dic):
        """
        simulate on a datadic
        """
        temp=np.zeros(shape=(dic['state'].shape[0]+1,dic['state'].shape[1]))
        temp[0]=(dic['external'][0]+7)*np.ones(self.nbrooms)
        for i, out in enumerate(dic['external']):
            temp[i+1] = self.thermalmodel(temp[i]\
                                          ,dic['Tset'][i]\
                                          ,dic['external'][i]\
                                          ,dic['Twater'][i]\
                                          ,dic['occupancy'][i]\
                                          ,dic['switch'][i]\
                                          ,dic['timestep'][i])
                                                      
        dic['predicted']=temp[1:]



    def errorplot(self):
        liste=(self.trainarray['next_state']-self.trainarray['predicted'])**2


    def log_likelihood(self,theta, out_temperature, obs_temperature):
        self.R, self.Lambda, self.log_f = theta
        house_temperature, _ = self.simulate(theta, out_temperature)
        noise_var = np.exp(self.log_f) * 0.1
        #self.MLE=np.sum(norm.logpdf(obs_temperature, loc=house_temperature+0.25, scale=noise_var ** 0.5))
        #return self.MLE
        pass



    








