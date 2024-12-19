import numpy as np
import pandas as pd
import joblib as jb
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler ,LabelEncoder

#Load the dataset
data = pd.read_csv("E:\\Ayush\\MIT_all\\MIT TY\\Sem_5\\MDM\\dataset\\assign_me\\Student_Performance.csv")

#Extract featurs and target
features = ['Hours Studied', 'Previous Score', 'Extracurricular Activities', 'Sleep Hours', 'Sample Papers Practiced']
target = 'Performance Index'

X = data[features]
Y = data[target]

label_encoder = LabelEncoder()
X['Extracurricular Activities'] = label_encoder.fit_transform(X['Extracurricular Activities'])

#scale the features to ensure that they are in a similar range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#plot the previous score vs performance index
plt.figure(figsize=(8, 5))
plt.scatter(data['Previous Score'], data[target], alpha=0.5, c='blue')
plt.xticks(np.arange(min(data['Previous Score']), max(data['Previous Score']) + 5, 5))
plt.xlim(min(data['Previous Score']) - 1, max(data['Previous Score']) + 1)
plt.xlabel('Previous Score')
plt.ylabel('Performance Index')
plt.title('Previous Score vs Performance Index')
plt.show()

#split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

#evaluate linear regression
print("\nLinear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("MSE:", mean_squared_error(y_test, lr_preds))
print("R²:", r2_score(y_test, lr_preds))

#random forest regressor model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

#evaluate random forest model
print("\nRandom Forest Performance:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("MSE:", mean_squared_error(y_test, rf_preds))
print("R²:", r2_score(y_test, rf_preds))

#feature importance for random forest model
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

#save the model
jb.dump(rf_model, 'random_forest_model.pkl')

#create a dataframe for the new data
new_data = pd.DataFrame({
    'Hours Studied': [2.5],
    'Previous Score': [79],
    'Extracurricular Activities': [1],
    'Sleep Hours': [5],
    'Sample Papers Practiced': [7]
})

#predict the performance index
prediction = rf_model.predict(new_data)
print("\nPredicted Performance Index: ", prediction[0])
print("\n")

#loadthe model
rf_model = jb.load('random_forest_model.pkl')

#get user input for prediction
hours_studied = float(input("Enter hours studied: "))
previous_score = float(input("Enter previous score: "))
extracurricular_activities = float(input("Enter if performed extracurricular actvities: "))
sleep_hours = float(input("Enter no of hours slept: "))
sample_papers = float(input("Enter the no. of sample papers practiced: "))

#create a dataframe for the new data
new_data = pd.DataFrame({
    'Hours Studied': [hours_studied],
    'Previous Score': [previous_score],
    'Extracurricular Activities': [extracurricular_activities],
    'Sleep Hours': [sleep_hours],
    'Sample Papers Practiced': [sample_papers]
})

#predict the performance index
prediction = rf_model.predict(new_data)
print("\nPredicted Performance Index: ", prediction[0])
