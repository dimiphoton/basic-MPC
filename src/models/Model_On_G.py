from sklearn.base import BaseEstimator
import os
import pandas as pd
import ast
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split



class model(BaseEstimator):
    def __init__(self,
        G,
        valverange=3,
        valvetrigger=1):
        self.state=G
        self.singlezone=(len([node for node,d in self.state.nodes().items() if d['category'] == 'room'])==1)


        #parametres généraux
        self.valvetrigger=valvetrigger
        self.valverange=valverange
        self.familyQ=500
        self.PV2activity=1

        self.trainset=None
        self.testset=None
        self.traindays=None
        self.testdays=None

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

    def getparams(self):
        """
        IMPORTANT: this function returns several numpy arrays to be read in further optimization
        """
        #parametres globaux
        #param_valve=np.array([self.valvetrigger])
        #parametres C
        param_C=np.array([d['C']['value'] for node,d in self.state.nodes().items() if d['category'] != 'external'])
        #parametres R
        param_R=np.array([d['R']['value'] for edge,d in self.state.edges().items()])
        #parametres de production de chaleur
        #param_fam=np.array([self.familyQ])
        #param_PVrad=np.array([self.PV2activity])
        #param_PVQ=np.array([d['Solarcoeff']['value'] for node,d in self.state.nodes().items() if d['category'] != 'external'])

        return param_C,param_R


    def setparams(self,param_C,param_R):
        """
        this function reads several parameter numpy arrays an updates the model
        """

        for index,value in enumerate([node for node,d in self.state.nodes().items() if d['category'] != 'external']):
            self.state.nodes[value]['C']['value']=param_C[index]

        for index,value in enumerate([edge for edge,d in self.state.edges().items()]):
            self.state.edges[value]['R']['value']=param_R[index]

        #self.familyQ=param_fam[0]

        #self.PV2activity=param_PVrad[0]

        #for index,value in enumerate([node for node,d in self.state.nodes().items() if d['category'] != 'external']):
        #    self.state.nodes[value]['Solarcoeff']['value']=param_PVQ[index]










    def valveopen(self,T,Tset):
        """""
        returns the valve status as boolean
        T = measured
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

    def retrievebigdf(self,relative_data_path):
        """
        this function returns the bigdf
        """
        cur_path = os.path.dirname(__file__)
        new_path = os.path.relpath(relative_data_path, cur_path)
        with open(new_path, 'r') as f:
            df=pd.read_csv(f,index_col=[0,1])
        return df

    def set_radstatus(self):
        """
        sets heater (bool) status 
        """
        for Hroom in [node for node,d in self.state.nodes().items() if d['category'] == 'heater']:
                zone=next(self.state.neighbors(Hroom)) #zone chauffée, seul voisin dans le graphe
                self.state.nodes[Hroom]['ON']=self.valveopen(self.state.nodes[zone]['T']['value'],self.state.nodes[zone]['Tset']['value'])


    def read_bigdf_row(self,row):
        """
        reads a PD.Serie row from bigdf['day'] and returns a dic to use for update methods
        """
        datadic={}
        if self.singlezone:
            keys=['room']
            measuredT=row['multizone_mesT']
            setT=row['multizone_setT']
        else:
            keys=['bath', 'bed1', 'bed2', 'bed3', 'dining', 'kitchen', 'living']
            measuredT=ast.literal_eval(row['multizone_mesT'])
            setT=ast.literal_eval(row['multizone_setT'])

        datadic['exterior']={}
        datadic['exterior']['T']=row['outside_mesT']
        datadic['boiler']={}
        datadic['boiler']['T']=row['waterT']

        for index,room in enumerate(keys):
            datadic[room]={}
            datadic[room]['T']=measuredT[index] #fonction initializeT à écrire ?
            datadic[room]['Tset']=setT[index]

        return datadic
        
        

        


    def set_data_from_dic(self,dict,initialize=False):
        """
        this function reads a dictionary key=node, value={data:value}
        and assigns Text,Tset and T (option) to the state attributes
        """
            
        #update Text
        self.state.nodes['exterior']['T']['value']=dict['exterior']['T']
        self.state.nodes['boiler']['T']['value']=dict['boiler']['T']

        #update T,setT
        #probleme avec hall
        for room in [node for node,d in self.state.nodes().items() if d['category'] == 'room']:

            if initialize==True:
                if 'Tset' in self.state.nodes[room].keys():
                    self.state.nodes[room]['T']['value']=dict[room]['T']
                else:
                    self.state.nodes[room]['T']['value']=20.0
            
            if 'Tset' in self.state.nodes[room].keys():
                self.state.nodes[room]['Tset']['value']=dict[room]['Tset']
        

            #update heaters
        
        if initialize==True:
            for Hroom in [node for node,d in self.state.nodes().items() if d['category'] == 'heater']:
                #print(Hroom, self.state.nodes[next(self.state.neighbors(Hroom))]['T']['value'])

                self.state.nodes[Hroom]['T']['value']=self.state.nodes[next(self.state.neighbors(Hroom))]['T']['value'] #zone chauffée, seul voisin dans le graphe
        self.set_radstatus()

    

    def prod_solar(self,row,method='simple'):
        if method =='simple':

            return 1

    def prod_occupancy(self,timeofday):
        return 1



    def conduction(self):
        """
        this function computes the dQ given T
        """
        for edge in self.state.edges:
            self.state.edges[edge]['dQ']['value']=abs(self.state.nodes[edge[0]]['T']['value'] - self.state.nodes[edge[1]]['T']['value'])/self.state.edges[edge]['R']['value']


            
                
        
    def initialization(self,row):
        dict=self.read_bigdf_row(row)
        self.set_data_from_dic(dict,initialize=True)

    def graph_predict(self,row,method='conduction',familyheating=False,solarheating=False):
        """
        predict the next state (room temperatures), given the up-to-date data in the graph
        """
        timestep =row['timestep']
        dict=self.read_bigdf_row(row)
        self.set_data_from_dic(dict,initialize=False)

        if method=='conduction':
            
            
            for node in self.state.nodes:
                
                
                if self.state.nodes[node]['category'] not in ['external']:

                    self.state.nodes[node]['dH']['value']=0 #initialisation de l'incrément

                    for voisin in self.state.neighbors(node):
                        #print(voisin,node)
                        Tv=self.state.nodes[voisin]['T']['value'] #temperature du voisin
                        #print('Tv',Tv)
                        Tn=self.state.nodes[node]['T']['value'] #temperature du noeud
                        #print('Tn',Tn)
                        #print(node, voisin, self.state.edges[node,voisin])
                        self.conduction()
                        dQ=self.state.edges[node,voisin]['dQ']['value'] #puissance conduction
                        self.state.nodes[node]['dH']['value'] += np.sign(Tv-Tn)*dQ*timestep #calcul
                    self.state.nodes[node]['T']['value']+= self.state.nodes[node]['dH']['value'] / self.state.nodes[node]['C']['value']
                


            self.set_radstatus()
            #actualisation radiateurs
            for node in self.state.nodes:
                if self.state.nodes[node]['category']=='heater':
                    if self.state.nodes[node]['ON']==True:
                        self.state.nodes[node]['T']['value'] = self.state.nodes['boiler']['T']
                    elif self.state.nodes[node]['ON']==False:
                        self.state.nodes[node]['T']['value'] = self.state.nodes[node]['dH']['value'] / self.state.nodes[node]['C']['value']

                
    def error_squarred_day(self,df,log=False):
        """
        returns an error metric given a day dataframe
        """
        liste=df.index[1:-1]
        error=0.0
        self.initialization(df.iloc[0])
        for timestamp in liste:
            row=df.loc[timestamp]
            dict=self.read_bigdf_row(row)
            for room,data in dict.items():
                if 'Tset' in data.keys():
                    if log==True:
                        error += np.log((data['T']-self.state.nodes[room]['T']['value'])**2)
                    else:
                        error += (data['T']-self.state.nodes[room]['T']['value'])**2
            self.graph_predict(row)
        return error

    def objRC(self,param_C,param_R,log=False):
        """
        returns an error metric given parameters
        """
        df=self.trainset

        self.setparams(param_C,param_R)
        error=0
        for day in self.traindays:
            error += self.error_squarred_day(df,log)
        
        return error
        


            

        


                        
                