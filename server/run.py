import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision.rein_learn.TLCS.real_tlcs_main as rl
import pickle
import requests
import json
Agent = rl.RL_Agent()

f = open('east.pkl2.txt', 'rb')
east_jo = pickle.load(f) 
f = open('west.pkl2.txt', 'rb')
west_jo = pickle.load(f) 

i=0
while True:
    try:
        sign = Agent.decide(east_jo[i], west_jo[i])
        i+=1
        print(sign)
    except Exception as e:
        print(e)
        break


