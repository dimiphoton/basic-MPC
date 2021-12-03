

class toymodel:
    def __init__(self):
        self.name='toymodel'
    def predict(self,inputvalue):
        return inputvalue

class model0:
    def __init__(self):
        self.PVload2heat=None
        self.PVProd2heat=None
        self.C=None
        self.R=None
        self.valverange=3
        self.valvetrigger=1
    
    def predict(self):
        