**Diabetes Prediction App using Machine Learning**
Overview
This is a Diabetes Prediction Web Application built using Streamlit and Random Forest Classifier. The application predicts whether an individual is likely to have diabetes based on health metrics such as glucose levels, BMI, age, and more. The model is trained using the famous Pima Indians Diabetes Dataset.

**Key Features**
Interactive UI: The app provides a user-friendly interface where users can input personal health metrics to get a prediction.
Machine Learning Model: Uses the Random Forest Classifier for prediction, trained on the Pima Indians Diabetes dataset.
Data Visualization: Displays summary statistics and data visualizations for better understanding of the dataset.
Real-Time Predictions: User input values are evaluated against the trained model to predict diabetes risk.
Technologies Used
Streamlit: A framework for building and deploying machine learning applications with ease.
Pandas: For data manipulation and analysis.
Scikit-learn: Used for building the machine learning model (Random Forest Classifier).
Python: Core programming language for development.
**Dataset**
The application uses the Pima Indians Diabetes Dataset. The dataset consists of several health-related metrics, such as:

Pregnancies
Glucose level
Blood Pressure
Skin Thickness
Insulin level
BMI (Body Mass Index)
Diabetes Pedigree Function
Age
Outcome (Target: 0 = No Diabetes, 1 = Diabetes)
You can download the dataset from Kaggle.

How to Run the Project
Prerequisites
Ensure you have the following installed on your machine:

Python 3.x
pip (Python package installer)
Libraries/Packages
The required Python libraries are:
streamlit==1.x.x
pandas==1.x.x
scikit-learn==1.x.x
You can install them using the following command:
pip install streamlit pandas scikit-learn
**Running the App**
Clone this repository:
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app
Download the diabetes.csv dataset and place it in the same directory.
Run the Streamlit app:
streamlit run app.py
A new tab will open in your default web browser with the app interface.
**ScreenShot**
![image](https://github.com/user-attachments/assets/066c766a-4455-4a3d-bfc9-83123d88e5d9)
![image](https://github.com/user-attachments/assets/39ced0e9-c422-4ea5-8c67-92f7156f0e08)
![image](https://github.com/user-attachments/assets/450a363b-9784-4739-9f5c-b3a93674086f)

**Code Explanation**
Data Preparation: The Pima Indians Diabetes dataset is read and preprocessed using Pandas. We split the data into features (X) and the target (y), and then further split into training and testing sets.

Model Training: A RandomForestClassifier is trained on the training dataset, and the accuracy of the model is evaluated on the test set.

User Input: The app provides a sidebar with sliders for users to input their personal health metrics, such as glucose levels, BMI, and age.

Prediction: Based on the user’s input, the app runs the input through the trained Random Forest model and predicts whether the user is likely to have diabetes.

**Application Interface**
Data Statistics: The app shows descriptive statistics of the dataset.
Data Visualization: A bar chart is used to visualize the entire dataset.
User Input Panel: A sidebar allows the user to input their health metrics for real-time prediction.
Prediction Output: After submitting the inputs, the app displays whether the user is healthy or likely to have diabetes based on the prediction model.
Screenshots
You can add screenshots of the app interface here to showcase its features.

**Conclusion**
This project demonstrates how machine learning models can be integrated into interactive web applications to make real-time predictions. By using Streamlit, you can quickly prototype and deploy such applications for practical use cases like diabetes prediction.

License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as proper credit is given.

Acknowledgments
Dataset: Pima Indians Diabetes Dataset from Kaggle.
Libraries: Streamlit, Pandas, Scikit-learn.
