# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from sumolib import checkBinary
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import numpy as np
import math
import timeit

from SimRunner import SimRunner
from TrafficGenerator import TrafficGenerator
from Memory import Memory
from Model import Model


m_config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
m_config.gpu_options.allow_growth = True


# sumo things - we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def save_graphs(sim_runner, total_episodes, plot_path):

    plt.rcParams.update({'font.size': 24})  # set bigger font size

    # reward
    data = sim_runner.reward_store
    plt.plot(data)
    plt.ylabel("Cumulative negative reward")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'reward_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

    # cumulative wait
    data = sim_runner.cumulative_wait_store
    plt.plot(data)
    plt.ylabel("Cumulative delay (s)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'delay_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

    # average number of cars in queue
    data = sim_runner.avg_intersection_queue_store
    plt.plot(data)
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'queue_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)



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

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # initializations
    model = Model(num_states, num_actions, batch_size)
    memory = Memory(memory_size)
    traffic_gen = TrafficGenerator(max_steps)
    sumoCmd = [sumoBinary, "-c", "intersection/custom.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]#TODO:change cfg file
    saver = tf.train.Saver()

    with tf.Session(config=m_config) as sess:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        sess.run(model.var_init)
        sim_runner = SimRunner(sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd)
        episode = 0

        while episode < total_episodes:
            print('----- Episode {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()
            sim_runner.run(episode)  # run the simulation
            stop = timeit.default_timer()
            print('Time: ', round(stop - start, 1))
            episode += 1

        os.makedirs(os.path.dirname(path), exist_ok=True)
        saver.save(sess, path + "my_tlcs_model.ckpt") 
        print("----- End time:", datetime.datetime.now())
        print("PATH:", path)
        save_graphs(sim_runner, total_episodes, path)
