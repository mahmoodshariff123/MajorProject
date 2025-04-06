import pandas as pd
import numpy as np
import pickle
import os

# Load the dataset
data = pd.read_csv('heart.csv')

# Sampling balanced dataset
one_df = data[data['target'] == 1].sample(138)
zero_df = data[data['target'] == 0]

# Concatenate both to create a balanced dataset
heart_df = pd.concat([one_df, zero_df], axis=0)

# Drop unnecessary columns (KEEP 'slope' since your error shows it's expected)
heart_df.drop(columns=['exang', 'oldpeak', 'ca', 'thal'], axis=1, inplace=True)

# Define features (X) and target (Y) - NO CHANGES
X = heart_df.drop(['target'], axis=1)
Y = heart_df['target']

# Scaling the features - NO CHANGES
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
x_std = scaler.transform(X)

# Train-test split - NO CHANGES
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_std, Y, test_size=0.2, random_state=42)

# Random Forest Classifier - NO CHANGES
from sklearn.ensemble import RandomForestClassifier
mrf = RandomForestClassifier()
mrf.fit(x_train, y_train)

# Make predictions - NO CHANGES
y_pred = mrf.predict(x_test)

# Accuracy - NO CHANGES
from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save model - NO CHANGES
os.makedirs('models', exist_ok=True)
with open('models/heart_disease_model.pkl', 'wb') as f:
    pickle.dump(mrf, f)
with open('models/scaler1.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    