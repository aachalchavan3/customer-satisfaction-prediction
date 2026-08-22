# Customer Satisfaction Prediction

A beginner-friendly Machine Learning mini project using Logistic Regression and Flask.

## Dataset
`customer_satisfaction.csv` contains 1000 sample customer records.

## ML Model
Logistic Regression with:
- One-hot encoding for Gender
- Standard scaling for numerical features
- 80/20 train-test split

## Model result on the included dataset
Accuracy: 77.00%

## How to run

1. Open the project folder in VS Code.
2. Create a virtual environment:
   `python -m venv venv`
3. Activate it on Windows PowerShell:
   `venv\Scripts\Activate.ps1`
4. Install packages:
   `pip install -r requirements.txt`
5. Run:
   `python app.py`
6. Open the local Flask address shown in the terminal.

## Important
The included CSV is a synthetic educational dataset created for this mini project. For a real-world project, replace it with a properly collected/public customer survey dataset and clearly cite its source.
