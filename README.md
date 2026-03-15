# Resume Classification using Machine Learning

## Project Overview

This project builds a machine learning model that automatically classifies resumes into different job categories based on the resume text.

The system analyzes resume content and predicts the most relevant job role such as Data Science, Java Developer, HR, Web Designing, and others.

## Dataset

The dataset contains **962 resumes** across **25 job categories**.

Each record includes:

* **Resume** – Full resume text
* **Category** – Job role label

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Machine Learning Pipeline

1. Load the resume dataset
2. Clean and preprocess the resume text
3. Convert text into numerical features using TF-IDF
4. Split dataset into training and testing sets
5. Train a Naive Bayes classification model
6. Evaluate model accuracy
7. Predict the category of new resumes

## Model Used

Naive Bayes Classifier

## Model Performance

Accuracy achieved: **96%**

## Project Structure

resume-classifier
│
├── data
│   └── resume_dataset.csv
│
├── src
│   └── main.py
│
├── requirements.txt
└── README.md

## Example Prediction

Input Resume:
Experienced Python developer with knowledge of machine learning, pandas, numpy, and data analysis.

Predicted Category:
Data Science

## Future Improvements

* Trying additional models such as SVM or Logistic Regression
* Building a web interface for uploading resumes
* Improving text preprocessing
