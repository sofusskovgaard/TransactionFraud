from server import socketio

@socketio.on('hello')
def handle_message(data):
    print('received message: ' + data)
    return "got it"