from flask import Flask, render_template, request
import joblib
from sklearn import svm

app = Flask(__name__)

# Load the pre-trained scaler and classifier
scaler = joblib.load('./model/scaler.lb')
classifier = joblib.load('./model/classifier.lb')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            Pregnancies = int(request.form['pregnancies'])
            Glucose = int(request.form['glucose'])
            Blood_Pressure = int(request.form['bloodPressure'])
            Skin_Thickness = int(request.form['skinThickness'])
            Insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])  # Changed to float as BMI is a decimal value
            Diabetes_Pedigree_Function = float(request.form['diabetesPedigreeFunction'])  # Changed to float
            age = int(request.form['age'])
            
            # Prepare the input data for prediction
            UNSEEN_DATA = [[Pregnancies, Glucose, Blood_Pressure, Skin_Thickness, Insulin, bmi, Diabetes_Pedigree_Function, age]]
            
            # Transform the data using the scaler
            transformed_data = scaler.transform(UNSEEN_DATA)
            
            # Predict the outcome
            PREDICTION = classifier.predict(transformed_data)[0]  # Correctly pass the 2D array
            
            # Map prediction to label
            label_dict = {0: 'No Diabetes', 1: 'Diabetes'}
            result = label_dict[PREDICTION]
            
            return render_template('home.html', output=result)
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
