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



# PLOT AND SAVE THE STATS ABOUT THE SESSION
if __name__ == "__main__":

    # --- TRAINING OPTIONS ---
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
        state = random.randint(0, 2, num_states)
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        sess.run(model.var_init)#restore model
       # sim_runner = SimRunner(sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd)
        #TODO: define state
        prediction = np.argmax(model.predict_one(state, sess))
        print("prediction : ", prediction)
        
