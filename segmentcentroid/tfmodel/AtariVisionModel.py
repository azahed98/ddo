from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *
import tensorflow as tf
import numpy as np
import ray

class AtariVisionModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self,  
                 k,
                 statedim=(42,42,1), 
                 actiondim=(6,1),
                 JSD_weight=0,
                 entropy_weight=0):
        
        super(AtariVisionModel, self).__init__(statedim, actiondim, k, [0,0],'all', JSD_weight, entropy_weight)


    def createPolicyNetwork(self):

        return conv2a3c(self.statedim, self.actiondim[0])  

        #return continuousTwoLayerReLU(self.statedim[0],
                                      #self.actiondim[0],
                                      #self.variance) 

    def createTransitionNetwork(self):

        #return logisticRegression(self.statedim[0], 2)
        return  conv2a3c(self.statedim, 2)




