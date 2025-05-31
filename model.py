import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('E:/minor/pattern/Crop_recommendation.csv')

X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Example usage: Predict crop for a new set of features
new_features = [[117 ,32,34,26.2724184,52.12739421,6.758792552,127.1752928,]]  # Replace with your own set of features
predicted_crop = model.predict(new_features)
print("Predicted crop:", predicted_crop)

import pickle

pickle.dump(model, open("model.pkl", "wb"))
