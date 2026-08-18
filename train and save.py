import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv("chess_data.csv")

# 1. Clean data
df = df[df['our_rating'] >= 500]
df = df[df['outcome'] != 0.5]
df['outcome'] = df['outcome'].astype(int)

# 2. Encode features
df['color_encoded'] = (df['our_color'] == 'white').astype(int)
df['time_encoded'] = df['time_class'].map({'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3})

le = LabelEncoder()
df['opening_encoded'] = le.fit_transform(df['opening_family'])

# 3. Define Features (No data leakage!)
X = df[['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded', 'opening_encoded']]
y = df['outcome']

# 4. Split the data to check REAL accuracy (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Gradient Boosting Model
print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Check Accuracy on the 20% "hidden" test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\n====================================")
print(f"✅ True Model Accuracy: {accuracy:.1f}%")
print(f"====================================\n")
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=["Loss", "Win"]))

# 7. Retrain on ALL data so the final API has the most experience possible
print("\nRetraining on all data for production deployment...")
model.fit(X, y)

# 8. Save the final "brain" and encoders
joblib.dump(model, 'chess_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Saved 'chess_model.pkl' and 'label_encoder.pkl'. Ready for the API!")