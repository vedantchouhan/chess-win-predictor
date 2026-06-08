import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("chess_data.csv")

# Remove low rating noisy games
df = df[df['our_rating'] >= 500]
print(f"Games after filtering: {len(df)}")

# Remove draws
df = df[df['outcome'] != 0.5]
df['outcome'] = df['outcome'].astype(int)

# Encode features
df['color_encoded'] = (df['our_color'] == 'white').astype(int)
df['time_encoded'] = df['time_class'].map({
    'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3
})

# Opening encoding
le = LabelEncoder()
df['opening_encoded'] = le.fit_transform(df['opening_family'])

X = df[['our_rating', 'opp_rating', 'rating_diff',
        'color_encoded', 'time_encoded',
        'opening_encoded', 'move_count']]
y = df['outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy*100:.1f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Loss', 'Win']))

# Feature importance
print("\nFeature Importance:")
features = ['our_rating', 'opp_rating', 'rating_diff',
            'color', 'time_class', 'opening', 'move_count']
for feat, imp in sorted(
    zip(features, model.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    print(f"{feat}: {imp*100:.1f}%")