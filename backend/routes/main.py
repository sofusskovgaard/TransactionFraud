from flask import jsonify
from server import app

@app.route('/hello-world')
def hello_world():
    return jsonify(msg="Herro Worl")