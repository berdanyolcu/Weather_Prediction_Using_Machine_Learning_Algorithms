import pd as pd


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import pandas as pd

# Define the path to the CSV file
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'

# Read the CSV file directly from the URL into a pandas DataFrame
df = pd.read_csv(path)

# Show the first 5 rows of the DataFrame
print(df.head())



# DATA PREPROCESSING

# Performing one hot encoding
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


df_sydney_processed.drop('Date',axis=1,inplace=True)


# Now, we set our 'features' or x values and our Y or target variable.
df_sydney_processed = df_sydney_processed.astype(float)

features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

# KNN
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size= 0.2, random_state=10)

Tree = DecisionTreeClassifier()
Tree.fit(X_train,y_train)

prediction = Tree.predict(X_test)

Tree_Accuracy_Score = accuracy_score(y_test, prediction)


print(f"Accuracy: {Tree_Accuracy_Score}")

Tree_JaccardIndex = jaccard_score(y_test, prediction)
print(f"Jaccard Index): {Tree_JaccardIndex}")

Tree_F1_Score = f1_score(y_test, prediction)
print(f"F1 Score): {Tree_F1_Score}")
