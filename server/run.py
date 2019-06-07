import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import application.big_trafficlight as app
import decision.rein_learn.TLCS.real_tlcs_main as rl
import pickle
import requests
import cv2
import json
Agent = rl.RL_Agent()

f = open('east.pkl2.txt', 'rb')
east_jo = pickle.load(f) 
f = open('west.pkl2.txt', 'rb')
west_jo = pickle.load(f) 

green = cv2.imread('green.jpg', cv2.IMREAD_COLOR)
red = cv2.imread('red.jpg', cv2.IMREAD_COLOR)
cv2.imshow('light', green)

i=0
sign=-1
iv=10
while True:
    try:
        signTemp = Agent.decide(east_jo[i], west_jo[i])
        if(sign==signTemp):
            iv+=1
        else:
            if iv>=10:
                sign = signTemp
                iv=0
            else:
                iv+=1

        i+=1
        print(sign)
    except Exception as e:
       # print(e)
        break


