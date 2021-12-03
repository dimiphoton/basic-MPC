import numpy as np

def valve(T,Tset,T1,T2):
    """
    returns the flow command as a float
    T = measured
    Tset = set
    T2 = margin below which flow > 0.0
    T1 = margin below which flow = 1.0
    """

    if T >= (Tset-T2):
        return 0
    else:
        return min (1, (-T-T2)/(T1-T2))


def flow_dispo(P,etage):
    return np.sqrt(P-etage)

def Qsun(PVProd,PVProd2heat):
    return PVProd*PVProd2heat

def Qhab(PLlLoad, PLLoad2heat,)

