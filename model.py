###################################
# Importing the dependecies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#######################################
# Data collection and Analysis

#loading the data from csv file to a Pandas dataframe
parkinsons_data = pd.read_csv(r"C:\Users\whora\Desktop\ParkinPredict\parkinsons.csv")

#printing the first 5 rows of the dataset
# parkinsons_data.head()

#number of rows and columns in the dataframe
# parkinsons_data.shape

#getting more information about the dataset
# parkinsons_data.info()

# checking for missing values in each column
# parkinsons_data.isnull().sum()

# getting some statistical measures about the data
# parkinsons_data.describe()


# distribution of target
# parkinsons_data['status'].value_counts()

######### 1 --> Parkinson's Positive

######### 0 --> Healthy

# goruping the data based on the target variable
# parkinsons_data.groupby('status').mean()

###########################
# Data Pre-Processing

# Separating the features and Target
X = parkinsons_data.drop(columns=['name','status'], axis = 1)
Y = parkinsons_data['status']

# Splitting the data to training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


################################
# Data Standardization
scaler  = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#################################
# Model Training

# Support Vector Machine Model

model = svm.SVC(kernel='linear')

# Trainig the SVM Model with Training data 
model.fit(X_train, Y_train)

#################################
# Model Evaluation

# Accuracy Score

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data: ', test_data_accuracy)

##################################
# Building a Predictive System

input_data = (162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)


if (prediction[0]==0):
  print("The Person does not have Parkinsons Disease")
else:
  print("The Person has Parkinsons")
