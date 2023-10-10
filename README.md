# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.THIVAKAR
RegisterNumber: 212222240109
```
```
python
import pandas as pd
data = pd.read_csv('dataset/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![Screenshot from 2023-09-01 07-38-22](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/4fb6dea0-5d8c-4d3c-a258-0a745f881461)

![Screenshot from 2023-09-01 07-38-33](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/2a6d05a2-a7c8-4841-96dc-fa4ece7f2733)

![Screenshot from 2023-09-01 07-38-42](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/17b88370-9854-4ccd-9c98-c43767c149ed)

![Screenshot from 2023-09-01 07-38-54](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/c7eaf312-8c93-468a-99a4-b906f43a0f61)

![Screenshot from 2023-09-01 07-39-13](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/aea32f5f-34ad-4efa-b276-f1ea2ddf76cc)

![Screenshot from 2023-09-01 07-39-22](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/224bbbc5-43a4-427e-8d3a-cc02c8753f72)

![Screenshot from 2023-09-01 07-39-33](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/93fbe087-ce21-4441-abab-5d32788b8cb2)

![Screenshot from 2023-09-01 07-39-42](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/be444b18-03ba-4f7d-9eef-a6e109714d9b)

![Screenshot from 2023-09-01 07-39-52](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/959f3943-bd7f-41ac-8b66-b92d2cd6e1d6)

![Screenshot from 2023-09-01 07-40-00](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/3affa85e-ae23-4c75-8543-bbfba2e8f7ba)

![Screenshot from 2023-09-01 07-40-53](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/782b3895-90a9-43b3-be3e-2314a1580d44)

![Screenshot from 2023-09-01 07-41-04](https://github.com/Gchethankumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118348224/4dcdfb3c-1021-4165-969e-0c2178c5b5fc)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
