# Spam Email Classifier

## Experiment: Spam Email Detection using Machine Learning

## Objective

The objective of this project is to classify emails as Spam or Not Spam (Ham) using machine learning techniques based on email content.

## Description

This project uses Natural Language Processing (NLP) and machine learning classification algorithms to analyze email messages and detect unwanted spam emails.

The model learns from previous email data and predicts whether a new email is spam or safe.

## Dataset

Dataset used:

Email Spam Dataset

The dataset contains:

- Email message text
- Email category label

Labels:
- Spam
- Ham (Not Spam)

## Machine Learning Algorithms Used

1. Multinomial Naive Bayes

2. Support Vector Machine (SVM)

## NLP Technique Used

TF-IDF Vectorization

TF-IDF converts email text into numerical values so that machine learning models can understand and classify messages.

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Import required libraries

2. Load email dataset

3. Select required columns

4. Convert labels into numerical values

5. Convert text data into TF-IDF features

6. Split dataset into training and testing data

7. Train classification models

8. Predict spam or not spam emails

9. Evaluate model performance


## Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report


## Output

The program provides:

- Naive Bayes accuracy
- SVM accuracy
- Confusion matrix
- Spam vs Non-spam graph
- Custom email spam prediction


## How to Run

Install required libraries:

pip install -r requirements.txt


Run the program:
email_classifier.py


## Project Structure

email_classifier.py

README.md

email.csv


## Conclusion

This project successfully detects spam emails using NLP and machine learning classification algorithms. The model helps filter unwanted emails and improves email security.
