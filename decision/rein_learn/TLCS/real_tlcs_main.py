# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime
import tensorflow as tf
import numpy as np
import math
import timeit
from numpy import random
from .Memory import Memory
from .Model import Model





class RL_Agent:
    def __init__(self):
        self.m_config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            # device_count = {'GPU': 1}
        )
        self.m_config.gpu_options.allow_growth = True
        self.num_states = 80
        self.num_actions = 2
        self.batch_size = 128
        self.memory_size = 80000
        self.model = Model(self.num_states, self.num_actions, self.batch_size)
        #self.sess.run(self.model.var_init)#is it needed?
        self.memory = Memory(self.memory_size)
        self.sess = tf.Session(config=self.m_config)
        self.saver = tf.train.Saver()
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "model_recent3")
        print(path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
    def calculate_state(self,east_jo, west_jo):
        state = np.zeros(80)
        for res in east_jo['reses']:
            lane_cell = self.get_lane_cell(res['distance'])
            if lane_cell == -1 : 
                continue
            lane_group = 4 #east
            veh_position = int(str(lane_group) + str(lane_cell))
            state[veh_position] = 1
        for res in west_jo['reses']:
            lane_cell = self.get_lane_cell(res['distance'])
            if lane_cell == -1 : 
                continue
            lane_group = 0 #west
            veh_position = lane_cell
            state[veh_position] = 1
        if east_jo['n_person'] > 1 :
            lane_group = 2 # north
            lane_cell = 0
            veh_position = int(str(lane_group) + str(lane_cell))
            state[veh_position] = 1
        if west_jo['n_person'] > 1 :
            lane_group = 6 # south
            lane_cell = 0
            veh_position = int(str(lane_group) + str(lane_cell))
            state[veh_position] = 1
        return state

    def decide(self, east_jo, west_jo):
        prediction = -1
        state = self.calculate_state(east_jo, west_jo)
        print("----- Start time:", datetime.datetime.now())
        prediction = np.argmax(self.model.predict_one(state, self.sess))
        print("prediction : ", prediction)
        return prediction

    def get_lane_cell(self, lane_pos):
        if lane_pos < 0:
            lane_cell = -1
        elif lane_pos < 2:
            lane_cell = 0
        elif lane_pos < 3:
            lane_cell = 1
        elif lane_pos < 5:
            lane_cell = 2
        elif lane_pos < 6:
            lane_cell = 3
        elif lane_pos < 8:
            lane_cell = 4
        elif lane_pos < 12:
            lane_cell = 5
        elif lane_pos < 20:
            lane_cell = 6
        elif lane_pos < 32:
            lane_cell = 7
        elif lane_pos < 80:
            lane_cell = 8
        elif lane_pos <= 150:
            lane_cell = 9
        return lane_cell
    
'''

if __name__ == "__main__":
    A = RL_Agent()
    print(type(A))

'''
