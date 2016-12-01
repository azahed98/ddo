#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.tfmodel.MLSoftMaxModel import MLSoftMaxModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

import tensorflow as tf

"""
This example demonstrates a proof of concept for manual segmentation
"""

m  = MLSoftMaxModel((6,1), (4,1), 4)

#first we load the gridworld map and initialize the environment
MAP_NAME = 'resources/GridWorldMaps/Roundabout.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
full_traj = []
vis_traj = []

for i in range(0,100):
    g = GridWorldEnv(copy.copy(gmap), noise=0.0)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    
    new_traj = []
    for t in traj:
        ns = np.ndarray(shape=(6))
        ns[0:2] = t[0]
        ns[2:4] = t[0]
        ns[4:6] = t[0]#np.argwhere(g.map == g.START)[0]

        a = np.zeros(shape=(4,1))
        a[t[1]] = 1

        new_traj.append((ns,a))

    full_traj.append(new_traj)
    vis_traj.extend(new_traj)

g.visualizePlan(vis_traj,blank=True)


opt = tf.train.AdamOptimizer(learning_rate=1e-4)
train = opt.minimize(m.getLossFunction()[0])
init = tf.initialize_all_variables()

#with m.sess as sess:
m.sess.run(init)

for it in range(500):
    print("Iteration",it)
    m.sess.run(train, m.sampleBatch(full_traj))


actions = np.eye(4)


g = GridWorldEnv(copy.copy(gmap), noise=0.0)

for i in range(m.k):
    states = g.getAllStates()
    policy_hash = {}
    trans_hash = {}

    for s in states:
        
        ns = np.ndarray(shape=(6))
        ns[0:2] = s
        ns[2:4] = s
        ns[4:6] = s#np.argwhere(g.map == g.START)[0]

        #print([m.evalpi(i,ns, actions[:,j]) for j in range(4)])
        l = [m.evalpi(i,ns, actions[j,:]) for j in range(4)]

        print(i, s,l, m.evalpsi(i,ns))
        action = np.argmax(l)

        policy_hash[s] = action

        #print(transitions[i].eval(np.array(ns)))
        trans_hash[s] = m.evalpsi(i,ns)

    g.visualizePolicy(policy_hash, trans_hash)


#m.fit()

#g = GridWorldEnv(copy.copy(gmap), noise=0.0)
#g.visualizePlan(vis_traj,blank=True)




"""
seg = JointSegCentroidInferenceDiscrete(TabularModel, TabularModel, 2, 6, 4)
#s.fit(full_traj)

policies, transitions = seg.fit(full_traj, learning_rate=0.01, max_iters=30, max_liters=1)

#seg.fitTraj(full_traj[0], learning_rate=0.3, max_iters=10)


g = GridWorldEnv(copy.copy(gmap), noise=0.0)
g.generateRandomStartGoal()

#g.visualizePlan(full_traj, blank=True)

#Visualization Code

"""

"""

#We can query the planner by running
full_traj = []
for i in range(0,50):
    g = GridWorldEnv(copy.copy(gmap), noise=0.1)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    full_traj.append(traj)

#g.visualizePlan(full_traj, blank=True)

s = SegCentroidInferenceDiscrete(LogitModel, 6)
q,p, policies = s.fit(full_traj,2,4)
print(p)


#Visualization Code
for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)
"""



"""

#Next, let's collect a dataset
#We can manually segment the data into each room's data
segments = []

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)
"""

#We can fit the model with a logistic regression policy class

"""
s = JointSegCentroidInferenceDiscrete(LogitModel, LogitModel, 2)
q,p, policies = s.fit(traj,2,4)


#Visualization Code
for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)
"""




