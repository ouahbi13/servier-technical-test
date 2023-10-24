from flask import Flask, request, jsonify
import json
from model_1 import predict_model_1
from model_2 import predict_model_2
from model_3 import predict_model_3

app = Flask(__name__)

@app.route('/predict/<path:model>', methods=['POST'])
def predict(model):
    """
        Flask route to predict a smiles binary property depending on the model type (1 or 2)
    """
    try:
        # Parse input data
        data = json.loads(request.data)
        smiles = data["smiles"]

        # Predict property P1
        if model == 'model_1':
            prediction, probability = predict_model_1(smiles)
        elif model == 'model_2':
            prediction, probability = predict_model_2(smiles)
        elif model == 'model_3':
            prediction, probability = predict_model_3(smiles)
        
        # print(prediction)
        
        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
