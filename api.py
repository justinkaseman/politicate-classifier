from flask import Flask, jsonify, make_response, request

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    features = request.json['input']
    return make_response(jsonify({'class': 'Test response'}))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
