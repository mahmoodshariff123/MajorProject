from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
heart_model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))
diabetes_model = pickle.load(open('models/Diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))
scaler1 = pickle.load(open('models/scaler1.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

# Routes for each disease
@app.route('/heart')
def heart():
    return render_template('heart/index.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney/index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes/index.html')

# Prediction routes
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        # Get all 9 features in EXACT ORDER used during training:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['slope'])  # ADD THIS
        ]
        
        # Scale and predict
        final_features = scaler1.transform([features])
        prediction = heart_model.predict(final_features)
        result = "healthy heart" if prediction[0] == 1 else "heart problem"
        return render_template('heart/index.html', prediction_text=result)
    except Exception as e:
        return str(e)

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        input_data = [float(request.form[key]) for key in request.form.keys()]
        prediction = kidney_model.predict([input_data])[0]
        result = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"
        return render_template('kidney/index.html', prediction_text=result)
    except Exception as e:
        return str(e)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        input_data = [float(request.form[key]) for key in request.form.keys()]
        scaled_input = scaler.transform([input_data])  # Apply scaling
        prediction = diabetes_model.predict(scaled_input)[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        return render_template('diabetes/index.html', prediction_text=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
