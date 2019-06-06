from flask import Flask, request, jsonify
import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision.rein_learn.TLCS.real_tlcs_main as rl
app = Flask(__name__)

east_jo = None
west_jo = None
sign = '0'
Agent = rl.RL_Agent()

@app.route('/')
def hell():
    return 'Hell World!'

@app.route('/east',methods=['POST'])
def east_post():
    global east_jo
    east_jo = json.loads(request.json)
   
    print("east sign",sign)
    return sign

@app.route('/west',methods=['POST'])
def west_post():
    global sign
    global west_jo
    west_jo = json.loads(request.json)
    sign = Agent.decide(east_jo, west_jo)
    print("west sign",sign)
    return sign


if __name__ == '__main__':
    app.run(debug=True, port=3000)