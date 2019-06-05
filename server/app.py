from flask import Flask, request, jsonify
import json
app = Flask(__name__)



@app.route('/')
def hell():
    return 'Hell World!'

@app.route('/east',methods=['POST'])
def east_post():
    jo = json.loads(request.json)
    print(jo['n_person'])
    #jo = request.form['n_person']
    return 'Hello World!'

@app.route('/west',methods=['POST'])
def west_post():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True, port=3000)