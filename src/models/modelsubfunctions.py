import numpy as np

    

def updateTrad(Trad,Twater,Q,C):
    if Trad < Twater:
        return Twater
    else:
        return Trad-Q/C

def flowdispo():
    pass


def flow_dispo(P,etage):
    return np.sqrt(P-etage)

def Qsun(PVProd,PVProd2heat):
    return PVProd*PVProd2heat

def Qhab(PVLoad, PVLoad2heat,):
    return PVLoad*PVLoad2heat




