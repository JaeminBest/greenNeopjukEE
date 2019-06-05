from flask import Flask, request, jsonify
import json
app = Flask(__name__)

east_jo = None
west_jo = None
sign = '0'

@app.route('/')
def hell():
    return 'Hell World!'

@app.route('/east',methods=['POST'])
def east_post():
    east_jo = json.loads(request.json)
    return sign

@app.route('/west',methods=['POST'])
def west_post():
    west_jo = json.loads(request.json)
    simulate()
    return 'Hello World!'

def simulate

if __name__ == '__main__':
    app.run(debug=True, port=3000)