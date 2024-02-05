import networkx as nx
import itertools

def myedge(G,node1,node2):
    G.add_edge(node1,node2,
    R={'value':100,'uncertainty':None},
    dQ={'value':None,'uncertainty':None})
    
def mynode(G,nodename='noname',category='room',heated=False):
    G.add_node(nodename,
    category=category,
    T={'value':None,'uncertainty':None},)
    if category!='external':
        G.nodes[nodename]['C']={'value':100,'uncertainty':None}
        G.nodes[nodename]['dH']={'value':None,'uncertainty':None}
        G.nodes[nodename]['dT']={'value':None,'uncertainty':None}
        G.nodes[nodename]['Solarcoeff']={'value':0.0001,'uncertainty':None}
    if category=='room':
        if heated==True:
            G.nodes[nodename]['Tset']={'value':None}
            mynode(G,'H'+nodename,'heater',heated=False)
            G.nodes['H'+nodename]['ON']=None
            #G.nodes['H'+nodename]['valveratio']={'value':None,'uncertainty':None}
            myedge(G,nodename,'H'+nodename)


def makesingle():
    G=nx.Graph()

    mynode(G,'room','room',heated=True)
    mynode(G,'exterior','external',heated=False)
    mynode(G,'boiler','external',heated=False)
    G['room']['C']['value']=100
    return G


def makemulti(rooms=['bath','bed1','bed2','bed3','dining','kitchen','living'],
            extrarooms=['hall'],
            external=['exterior','boiler']):
    G=nx.Graph()
    
    
    

    #créer les zones chauffées, les heaters, les edges
    for zone in rooms:
        mynode(G,zone,'room',heated=True)
    
    #créer les zones non chauffées
    if extrarooms != None:
        for zone in extrarooms:
            mynode(G,zone,'room',heated=False)
    
    #créer l'exterieur
    for zone in external:
        mynode(G,zone,'external',heated=False)

    #relier les zones à l'extérieur
    for zone in rooms+extrarooms:
        myedge(G,zone,'exterior')
    
    #relier les rooms entre elles
    myedge(G,'kitchen','dining')
    myedge(G,'bed2','bed3')
    if 'hall' not in extrarooms:
        for combi in list(itertools.combinations(rooms, 2)):
            myedge(G,combi[0],combi[1])


    elif 'hall' in extrarooms:
        for zone in rooms:
            myedge(G,'hall',zone)
        



    

    return G

