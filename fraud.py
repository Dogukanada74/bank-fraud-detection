import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('C:/Users/dogukan/PycharmProjects/bankfraud/creditcard_2023.csv')
bank_data = data.head(100000)

X = bank_data.drop(columns=['Class'])
y = bank_data.Class


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_valid = my_imputer.transform(X_valid)

our_model = RandomForestClassifier(class_weight='balanced')
our_model.fit(imputed_X_train, y_train)


y_pred = our_model.predict(imputed_X_valid)


accuracy = accuracy_score(y_valid, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


print(classification_report(y_valid, y_pred))