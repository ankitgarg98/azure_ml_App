# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate dummy data
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [40, 50, 60, 65, 70, 80, 85, 90, 95, 98],
    'assignments_done': [1, 1, 2, 3, 3, 4, 4, 5, 5, 5],
    'sleep_hours': [8, 7, 7, 6, 6, 6, 5, 5, 4, 4],
    'result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 1 = pass, 0 = fail
}

df = pd.DataFrame(data)

# Split data
X = df[['hours_studied', 'attendance', 'assignments_done', 'sleep_hours']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
