import pickle
import requests
import json
f = open('east.pkl.txt', 'rb')
east_jo = pickle.load(f) 
f = open('west.pkl.txt', 'rb')
west_jo = pickle.load(f) 
i=0
while True:
    try:
        jo = json.dumps(east_jo[i])
        print("joeast", jo)
        url = "http://localhost:3000/east"
        r = requests.post(url, json=jo)
        jo = json.dumps(west_jo[i])
        print("jowest", jo)
        url = "http://localhost:3000/west"
        r = requests.post(url, json=jo)
        break
    except Exception as e:
        print(e)
        break

