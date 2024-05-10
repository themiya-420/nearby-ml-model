from flask import Flask, request, jsonify

app = Flask(__name__)


# test function to test the flask server.
@app.route('/print', methods=['GET'])
def test():
    return jsonify({'message': 'Hello World!'})


# run the flask server.
if __name__ == '__main__':
    app.run(debug=True)
