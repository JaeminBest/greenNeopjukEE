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
from RealSimRunner import SimRunner
from TrafficGenerator import TrafficGenerator
from Memory import Memory
from Model import Model



m_config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
m_config.gpu_options.allow_growth = True


def calculate_state(east_jo, west_jo):
    state = np.zeros(80)
    for res in east_jo['reses']:
        lane_cell = get_lane_cell(res['distance'])
        lane_group = 4 #east
        veh_position = int(str(lane_group) + str(lane_cell))
        state[veh_position] = 1
    for res in west_jo['reses']:
        lane_cell = get_lane_cell(res['distance'])
        lane_group = 0 #west
        veh_position = int(str(lane_group) + str(lane_cell))
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

def rl_decide(east_jo, west_jo):
    # --- TRAINING OPTIONS ---
    prediction = -1
    gui = False
    total_episodes = 300
    gamma = 0.75
    batch_size = 128
    memory_size = 80000
    path = "./model/model_recent3/"  # nn = 5x400, episodes = 300, gamma = 0.75
    # ----------------------

    # attributes of the agent
    num_states = 80
    num_actions = 2
    max_steps = 5400  # seconds = 1 h 30 min each episode
    green_duration = 10
    yellow_duration = 2


    # initializations
    # TODO: restore model
    model = Model(num_states, num_actions, batch_size)
    memory = Memory(memory_size)
    with tf.Session(config=m_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('model_recent3/'))
        state = calculate_state(east_jo, west_jo)#random.randint(0, 2, num_states)
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        sess.run(model.var_init)#restore model
       # sim_runner = SimRunner(sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd)
        #TODO: define state
        prediction = np.argmax(model.predict_one(state, sess))
        print("prediction : ", prediction)
    return prediction

def get_lane_cell(lane_pos):
    if lane_pos < 2:
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
    rl_decide()
'''

