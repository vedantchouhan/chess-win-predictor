import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("chess_data.csv")

# Remove draws for simplicity — binary classification
df = df[df['outcome'] != 0.5]
df['outcome'] = df['outcome'].astype(int)

# Features
df['color_encoded'] = (df['our_color'] == 'white').astype(int)
df['time_encoded'] = df['time_class'].map({
    'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3
})

X = df[['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded']]
y = df['outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy*100:.1f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Loss', 'Win']))

# Feature importance
print("\nFeature Importance:")
features = ['our_rating', 'opp_rating', 'rating_diff', 'color', 'time_class']
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp*100:.1f}%")