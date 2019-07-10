import tensorflow as tf
import numpy as np
from .TFModel import TFModel
from segmentcentroid.inference.forwardbackward import ForwardBackward
from tensorflow.python.client import timeline
import tensorflow_probability as tfp

class TFSeparableModel(TFModel):
    """
    This class defines a common instantiation of the abstract class TFModel
    where all of the policies and transitions are of an identical type.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 boundary_conditions,
                 prior,
                 JSD_weight=0,
                 entropy_weight=0):
        """
        Create a model from the parameters

        Positional arguments:
        statedim -- numpy.ndarray defining the shape of the state-space
        actiondim -- numpy.ndarray defining the shape of the action-space
        k -- float defining the number of primitives to learn
        """

        self.policy_networks = []
        self.transition_networks = []
        self.JSD_weight = JSD_weight
        self.entropy_weight = entropy_weight
        super(TFSeparableModel, self).__init__(statedim, actiondim, k, boundary_conditions, prior)


    def getLossFunction(self):

        """
        Returns a loss function that sums over policies and transitions
        """

        loss_array = []

        pi_vars = []
        logits = 0
        entropies = []
        for i in range(0, self.k):
            logits = logits + self.policy_networks[i]['logits']
            loss_array.append(self.policy_networks[i]['wlprob'])
            entropies.append(self.policy_networks[i]['entropy'])
            pi_vars.append((self.policy_networks[i]['state'], 
                            self.policy_networks[i]['action'], 
                            self.policy_networks[i]['weight']))

        psi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.transition_networks[i]['wlprob'])
            psi_vars.append((self.transition_networks[i]['state'], 
                            self.transition_networks[i]['action'], 
                            self.transition_networks[i]['weight']))

        logits /= self.k
        distr = tfp.distributions.Categorical(logits=logits)
        entropy = distr.entropy()
        entropy_reg = -1 * (self.JSD_weight * tf.reduce_sum(entropy) + (self.entropy_weight - self.JSD_weight / self.k) * tf.reduce_sum(entropies) )
        return tf.reduce_sum(loss_array) + entropy_reg, pi_vars, psi_vars, loss_array



    def initialize(self):
        """
        Initializes the internal state
        """

        for i in range(0, self.k):
            self.policy_networks.append(self.createPolicyNetwork())

        for i in range(0, self.k):
            self.transition_networks.append(self.createTransitionNetwork())


    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):

        feed_dict = {self.policy_networks[index]['state']: s, 
                     self.policy_networks[index]['action']: a}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        dist = self.sess.run(self.policy_networks[index]['prob'], feed_dict, options=run_options, run_metadata=run_metadata)

        if self.policy_networks[index]['discrete']:
            return np.sum(np.multiply(dist,a), axis=1)


        dista = self.sess.run(self.policy_networks[index]['lprob'], feed_dict)

        #print("predpi", dista)

        #print("predpi", dist)
        
        return dist
            

    #returns a probability distribution over actions
    def _evalpsi(self, index, s):

        dummyAction = np.zeros((s.shape[0], 2))
        dummyAction[:,1] = 1

        feed_dict = {self.transition_networks[index]['state']: s, 
                     self.transition_networks[index]['action']: dummyAction}
        #print(s)
        dist = self.sess.run(self.transition_networks[index]['prob'], feed_dict)

        #dista = self.sess.run(self.transition_networks[index]['lprob'], feed_dict)

        #print("predpsi", np.argwhere(s==1), dist)

        #if not self.transition_networks[index]['discrete']:
        #    raise ValueError("Transition function must be discrete")

        if len(dist.shape) == 1:
            return dist
        else:
            return dist[:,1]



        
