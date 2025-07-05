# 🎓 Student Performance Predictor using Machine Learning

An end-to-end Machine Learning web application that predicts student performance (Pass/Fail) based on various academic and behavioral factors. This project covers everything from data preprocessing and model training to deployment with a modern Flask-based interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat) ![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange) ![Deployment](https://img.shields.io/badge/Deployed-Local%20Server-green)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)
- [Connect with Me](#connect-with-me)

---

## 📖 Overview

This classification project predicts whether a student will pass based on key attributes like:

- Study hours per week

- Attendance rate (%)

- Previous academic grades

- Participation in extracurricular activities

- Parent education level

The project uses a Random Forest model trained on a labeled dataset and is integrated into a sleek Flask-based frontend for real-time predictions.

---

## 🎥 Demo

Watch the full walkthrough video on [YouTube]() 

## Screenshots

![Demo Screenshot](https://i.postimg.cc/VNRh2KVq/studentss.png)
![Demo Screenshot](https://i.postimg.cc/502rJ8hZ/studentss2.png)

---

## 📁 Project Structure
```bash
Student Performence/
├── app.py                              # Flask web application
├── train_model.py                      # Model training script
├── student_performance_model_optimized.joblib  # Optimized model (0.26 MB)
├── student_performance_prediction.csv   # Dataset
├── requirements.txt                    # Dependencies
├── Student_Performance_Predictor_Workflow.docx  # Documentation
└── .git/                              # Git repository
```

---

## 🚀 Features

- Predicts student performance (Pass/Fail) using Random Forest

- Handles categorical encoding and missing value imputation

- Clean and interactive frontend with progress tracking and form validation

- Real-time predictions via Flask web application

- Model and encoders saved using Joblib for efficient reuse

---

## 🧠 Tech Stack

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, joblib, flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Model:** Random Forest Classifier

---

## ⚙️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
````

3. **Train the model**

```bash
python train_model.py
````
4. **Run the Flask app**

```bash
python app.py
````
5. **Open in your browser**

```bash
http://127.0.0.1:5000/

````

---

## 📈 Model Performance

```
Metric	Score
Accuracy	~85%
Model Size	~0.2 MB
Algorithm	Random Forest Classifier
```

---

## 🔧 Future Improvements

- Deploy app on Render or Hugging Face Spaces

- Add charts to visualize student insights

- Expand dataset with more features (e.g., sleep, internet access)

- Add login/auth system for multi-user access

---

## 🤝 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/parthiv-majumdar-524046238/)

- 🧠 [YouTube](https://www.youtube.com/@parthivmajumdar6805)

- 💻 [Portfolio](https://portfolio-parthiv.vercel.app/)
