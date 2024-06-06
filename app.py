from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/alzheimer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        cognitive_score = float(request.form['cognitive_score'])
        # Add other required features here
        
        features = np.array([[age, gender, cognitive_score]])
        prediction = model.predict(features)
        
        if prediction == 0:
            result = 'Low risk of Alzheimer\'s'
        else:
            result = 'High risk of Alzheimer\'s'
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
