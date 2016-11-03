#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.LogitModel import LogitModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete
from segmentcentroid.planner.traj_utils import *

import numpy as np

MAP_NAME = 'resources/GridWorldMaps/11x11-Rooms.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
g = GridWorldEnv(gmap, noise=0.3)

#g.visualize(policy)

v = ValueIterationPlanner(g)

#g.visualizePolicy(v.policy)

segments = []

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

s = SegCentroidInferenceDiscrete(LogitModel, 2)

q,p, policies = s.fit(segments,2,4)

for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)








##Example 1##
#Initialize GridWorld and query a state
"""

g.init()
print g.getState()
g.play(3)
print g.getState()
"""

