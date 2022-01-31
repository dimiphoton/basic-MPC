

def thermalmodel(model, state, Tset, Twater, Text, switch, timestep,debug=False):
    """
    house_temperature: observed+obsered house rooms
    returns next state given current state, Text,timestep
    A CORRIGER: AJOUTER Tbasement optionnel
        """

    zones=np.concatenate((state, Text))

    dH=np.zeros(model.nbzones)

        # PARTIE CONDUCTION

    dH+=timestep*np.matmul(model.Lmatrix,state+Text)

        #PARTIE RADIATEURS

    dH+=timestep*model.heating(state,Tset,Twater,switch)

        #PARTIE occupation


        #conversion température
    fulldT=dH[0:model.nbheated+self.nbext]*np.reciprocal(model.magnitude['C'] * model.param['C'])
        

        #ajout bruit thermique
    if model.param['temperature_noise'][0] >0.0:
        state += (model.param['temperature_noise'][0]**0.5)*np.random.randn(model.nbzones)