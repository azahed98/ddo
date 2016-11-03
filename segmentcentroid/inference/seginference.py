
import numpy as np
import copy
from sklearn.preprocessing import normalize

"""
This class defines the main inference logic
"""

class SegCentroidInferenceDiscrete(object):

    def __init__(self, policy_class, k):
        self.policy_class = policy_class
        self.k = k


    #X is a list of segmented trajectories
    def fit(self, X, statedims, actiondims, max_iters=100, learning_rate=0.001):
        
        #create k initial policies
        policies = [copy.copy(self.policy_class(statedims, actiondims)) for i in range(0,self.k)]


        #initialize q and P for iterations
        q = np.matrix(np.ones((len(X),self.k)))/self.k
        P = np.matrix(np.ones((self.k,1)))/self.k

        
        #Outer Loop For Gradient Descent
        for it in range(0, max_iters):

            #print("Iteration", it, q, P)
            #print(P)

            q, P = self._updateQP(X, policies, q, P)


            for seg in range(0, self.k):
                policies[seg].descent(self._batchGrad(X, policies[seg],seg, q), learning_rate)

        return q, P, policies

            
    """
    Defines the inner loops
    """

    def _batchGrad(self, X, policy, policy_index, q):

        gradSum = None

        m = len(X)

        for plan in range(0, m):

            traj = X[plan]

            pointGrad = q[plan, policy_index]*self._trajLogDeriv(traj, policy)

            if gradSum is None:
                gradSum = pointGrad
            else:
                gradSum = gradSum + pointGrad

        return gradSum*1.0/m



    def _trajLogDeriv(self, traj, policy):
        gradSum = None

        for t in range(0, len(traj)):
            obs = np.matrix(traj[t][0])
            action = traj[t][1]
            deriv = policy.log_deriv(obs, action)

            if gradSum is None:
                gradSum = deriv
            else:
                gradSum = gradSum + deriv

        return gradSum


    def _updateQP(self, X, policies, q, P):

        #how many trajectories are there in X
        m = len(X)

        qorig = copy.copy(q)
        Porig = copy.copy(P)

        #for each trajectory
        for plan in range(0, m):

            #for each segments
            for seg in range(0, self.k):

                q[plan, seg] = P[seg]*self._segLikelihood(X[plan], policies[seg])

            #print q[plan, :], plan, P[seg]
      
        normalization = np.matrix(np.sum(q, axis=1))

        normalization_matrix = np.tile(1/normalization, [1,self.k])
        q = np.multiply(q, normalization_matrix)
        P = np.matrix(np.sum(q, axis=0)).T/m


        #test if nan
        if np.product(1-np.isnan(q)) == 0:
            return qorig, Porig

            
        return q,P



    def _segLikelihood(self, traj, policy):
        product = 1

        for t in range(0, len(traj)):

            obs = np.matrix(traj[t][0])

            pred = np.squeeze(policy.eval(obs))

            action = traj[t][1]
            
            preda = pred[action]

            product = preda * product

        return product





class JointSegCentroidInferenceDiscrete(object):

    def __init__(self, policy_class, transition_class, k):
        self.policy_class = policy_class
        self.transition_class = transition_class
        self.k = k


    #X is a trajectory
    def fit(self, X, statedims, actiondims, max_iters=10, learning_rate=0.01):
        
        #create k initial policies
        policies = [copy.copy(self.policy_class(statedims, actiondims, unnormalized=True)) for i in range(0,self.k)]

        #create k initial transitions
        transitions = [copy.copy(self.transition_class(statedims, 1, unnormalized=True)) for i in range(0,self.k)]

        #hint transition matrix, defaulted to sequential (bidiagonal)
        Ph = np.tri(self.k,self.k, 1) - np.tri(self.k,self.k, -1)
        Ph = normalize(Ph, axis=1, norm='l1')
        q = np.matrix(np.ones((len(X),self.k)))/self.k
        psi = np.matrix(np.ones((len(X),self.k)))/self.k
       
        #print(Ph)
        #print(self._forward(20, 1, X, policies, transitions, Ph))

        for it in range(0, max_iters):
            q = self._updateQ(q, X, policies, transitions, Ph)
            psi = self._updatePsi(psi, X, policies, transitions, Ph)

            print("Iteration", it)

            for seg in range(0, self.k):
                policies[seg].descent(self._batchGrad(X, policies[seg],seg, q), learning_rate)
                transitions[seg].descent(self._batchGrad(X, transitions[seg],seg, psi), learning_rate)

        return q, psi, policies


    def _updateQ(self, q, X, policies, transitions, Ph):
        
        newq = copy.copy(q)

        for h in range(self.k):
            for t in range(len(X)-1):
                newq[t,h] = self._forward(t, h, X, policies, transitions, Ph)*\
                            self._backward(t, h, X, policies, transitions, Ph)


        newq = normalize(newq, axis=1, norm='l1')

        return newq


    def _updatePsi(self, psi, X, policies, transitions, Ph ):

        newpsi = copy.copy(psi)

        for t in range(len(X)-1):
            for h in range(self.k):
                
                total = 0

                for hp in range(self.k):
                    total = total + \
                            self._backwardTerm(t,h,X, policies, transitions, Ph)* \
                            Ph[hp, h] * \
                            self._forward(t, hp, X, policies, transitions, Ph)

                    #print(self._backwardTerm(t,h,X, policies, transitions, Ph))

                newpsi[t,h] = total

        newpsi = normalize(newpsi, axis=1, norm='l1')

        return newpsi


    def _backwardTerm(self, t, h, X, policies, transitions, Ph):

        s = X[t][0]
        sp = X[t+1][0]
        a = X[t][1]

        #print(self._backward(t, h, X, policies, transitions, Ph),self._pi_a_giv_s(s,a, policies[h]), self._pi_term_giv_s(sp, policies[h]))

        return self._backward(t, h, X, policies, transitions, Ph)*\
               self._pi_a_giv_s(s,a, policies[h])*\
               self._pi_term_giv_s(sp, transitions[h])



    def _backward(self, 
                  #args
                  t,
                  hp,
                  #control state
                  traj, 
                  policies,
                  transitions,
                  Ph):
        
        state = traj[t][0]
        next_state = traj[t+1][0]
        action = traj[t][1]

        #base case
        if t == 0:
            
            return np.sum([ self._pi_a_giv_s(state,action, policies[h])* \
                            (self._pi_term_giv_s(next_state, transitions[h])* \
                            Ph[hp,h])\
                           for h in range(self.k) if Ph[hp,h] > 0\
                         ])

        else:
            return np.sum([ self._backward(t-1, h, traj, policies, transitions, Ph)* \
                            self._pi_a_giv_s(state,action, policies[h])* \
                            (self._pi_term_giv_s(next_state, transitions[h])* \
                            Ph[hp,h])\
                           for h in range(self.k) if Ph[hp,h] > 0\
                         ])

    def _forward(self, 
                  #args
                  t,
                  h,
                  #control state
                  traj, 
                  policies,
                  transitions,
                  Ph):
        
        state = traj[t][0]
        next_state = traj[t+1][0]
        action = traj[t][1]

        #base case
        if t+2 >= len(traj):
            
            return self._pi_a_giv_s(next_state,action, policies[h])

        else:
            return self._pi_a_giv_s(next_state,action, policies[h]) * \
                   np.sum([ self._pi_term_giv_s(state, transitions[h])* \
                            Ph[hp,h]* \
                            self._forward(t +1,hp, traj, policies, transitions, Ph) \
                            for hp in range(self.k) if Ph[hp,h] > 0])



    def _pi_a_giv_s(self, s,a, policy):
        obs = np.matrix(s)
        pred = np.squeeze(policy.eval(obs))
        action = a
        return pred[action]


    def _pi_term_giv_s(self, s, transition):
        obs = np.matrix(s)
        pred = np.squeeze(transition.eval(obs))
        return pred



    def _batchGrad(self, X, policy, policy_index, q):

        gradSum = None

        m = len(X)

        for t in range(0, m):

            obs = np.matrix(X[t][0])
            action = X[t][1]

            pointGrad = q[t, policy_index]*policy.log_deriv(obs, action)

            if gradSum is None:
                gradSum = pointGrad
            else:
                gradSum = gradSum + pointGrad

        return gradSum*1.0/m







        