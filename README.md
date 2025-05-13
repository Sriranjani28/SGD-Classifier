# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.
2.Data preprocessing 
3.Cleanse data,handle missing values,encode categorical variables.
4.Model Training:Fit logistic regression model on preprocessed data.
5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6.Prediction: Predict placement status for new student data using trained model.
7.End the program

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SRIRANJANI.M
RegisterNumber:  212224040327
*/
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
iris = load_iris()

# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
x = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)
y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)

<img width="621" alt="1" src="https://github.com/user-attachments/assets/d0a6bb57-a0b3-4b13-99bc-191e684abbc1" />

<img width="505" alt="2" src="https://github.com/user-attachments/assets/9b3b5ce1-2da7-401b-896f-e675d1c69dbf" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
