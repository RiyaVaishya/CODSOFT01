## TASK-01

## Titanic Survival Prediction

This repository contains a machine learning project that predicts whether a passenger on the Titanic survived or not based on various features such as age, gender, ticket class, fare, and more. This is a classic beginner project using the Titanic dataset.

## Introduction:-

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project uses data science and machine learning techniques to predict the survival of passengers based on various features.

## Dataset:-

The dataset used in this project is the Titanic dataset, which is available on [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset). It contains information about the passengers, such as:

- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Fare paid for the ticket
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Steps:-

1. Load and Explore the Dataset-
Load the dataset and display the first few rows to understand its structure.

2. Data Preprocessing-
Handle missing values, encode categorical variables, and perform feature engineering.

3. Visualize the Data-
(Optional) Visualize the data to gain insights into distributions and relationships between features.

4. Feature Selection-
Select relevant features for training the model.

5. Train-Test Split-
Split the dataset into training and testing sets to evaluate model performance.

6. Model Training-
Train a Random Forest classifier on the training data.

7. Model Evaluation-
Evaluate the model using metrics such as accuracy, precision, recall, and the confusion matrix.

## Dependencies:-

This project requires the following dependencies:

- Python (version 3.x)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Results:-

The script will preprocess the data, train a Random Forest classifier, and evaluate its performance. The results, including accuracy, precision, recall, confusion matrix, and classification report, will be displayed.

## Conclusion:-

This project demonstrates the application of machine learning techniques to predict the survival of passengers on the Titanic. By analyzing various features such as age, gender, ticket class, and fare, the Random Forest classifier achieves an accuracy of 84%, indicating that the model performs reasonably well in predicting survival outcomes.

Through data preprocessing, feature selection, model training, and evaluation, this project provides a comprehensive example of a typical machine learning workflow. However, there is always room for improvement. Future enhancements could include trying different algorithms, fine-tuning hyperparameters, or incorporating additional features for better predictive performance.

Overall, this project serves as a valuable learning experience for beginners in data science and machine learning, offering insights into data analysis, model building, and evaluation methodologies. Contributions and feedback are welcome to further refine and improve this project.

## My Linkedin Profile:- www.linkedin.com/in/riya-vaishya-537731278
