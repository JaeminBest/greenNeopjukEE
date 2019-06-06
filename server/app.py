from flask import Flask, request, jsonify
import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decision.rein_learn.TLCS as rl
app = Flask(__name__)

east_jo = None
west_jo = None
sign = '0'
Agent = rl.real_tlcs_main.RL_Agent()

@app.route('/')
def hell():
    return 'Hell World!'

@app.route('/east',methods=['POST'])
def east_post():
    east_jo = json.loads(request.json)
    return sign

@app.route('/west',methods=['POST'])
def west_post():
    global sign
    west_jo = json.loads(request.json)
    sign = Agent.decide(east_jo, west_jo)
    return sign


if __name__ == '__main__':
    app.run(debug=True, port=3000)