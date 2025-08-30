🩺 Breast Cancer Prediction Web App

An interactive Streamlit web application that predicts whether a breast tumor is Benign (B) or Malignant (M) using Machine Learning models (K-Nearest Neighbors and Support Vector Machines).

🚀 Features

Upload your own CSV dataset and run predictions.

Choose between KNN and SVM classifiers.

Tune model hyperparameters interactively.

Visualize results with:

Confusion Matrix

ROC Curve

View accuracy, precision, recall, and classification report.

git clone https://github.com/YOUR_USERNAME/breast-cancer-streamlit.git
cd breast-cancer-streamlit

Install the required dependencies:

pip install -r requirements.txt
Run the app:

streamlit run app.py

📂 Project Structure
breast-cancer-streamlit/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── breast_cancer.csv      # Dataset (optional, for default use)
└── README.md              # Project documentation

📊 Dataset

This app uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which classifies tumors as benign or malignant based on fine-needle aspirate (FNA) features.

Diagnosis column: M = Malignant, B = Benign.

Features: Mean, Standard Error, and Worst values for radius, texture, perimeter, area, etc.

📌 Dataset Source (UCI Machine Learning Repository)

🎯 Usage

Launch the app with streamlit run app.py.

On the sidebar:

Choose a classifier (KNN or SVM).

Adjust hyperparameters.

Click Classify to train the model.

Upload a CSV file to test predictions.

View results in the main window with metrics and plots.

🌐 Deployment

You can deploy this app for free on Streamlit Cloud
:

Fork this repository.

Go to Streamlit Cloud → New App.

Connect your repo and select app.py.

Deploy and share the link 🎉

📌 Requirements

Python 3.8+

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib


👨‍💻 Author
Abraham Pulickakudiyil
abrahampulickakudiyil@gmail.com
