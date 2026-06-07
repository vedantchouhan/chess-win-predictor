import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model again
df = pd.read_csv("chess_data.csv")
df = df[df['outcome'] != 0.5]
df['outcome'] = df['outcome'].astype(int)
df['color_encoded'] = (df['our_color'] == 'white').astype(int)
df['time_encoded'] = df['time_class'].map({
    'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3
})

X = df[['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded']]
y = df['outcome']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("=== Chess Win Probability Predictor ===")
print("Based on Vedant's 4714 real games\n")

while True:
    print("Enter game details:")
    our_rating = int(input("Your rating: "))
    opp_rating = int(input("Opponent rating: "))
    color = input("Your color (white/black): ").strip().lower()
    time_class = input("Time class (bullet/blitz/rapid/daily): ").strip().lower()

    color_encoded = 1 if color == 'white' else 0
    time_encoded = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}.get(time_class, 1)
    rating_diff = our_rating - opp_rating

    features = pd.DataFrame([[our_rating, opp_rating, rating_diff, color_encoded, time_encoded]], 
                        columns=['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded'])  
  
    win_prob = model.predict_proba(features)[0][1] * 100
    loss_prob = model.predict_proba(features)[0][0] * 100

    print(f"\n{'='*40}")
    print(f"Win Probability:  {win_prob:.1f}%")
    print(f"Loss Probability: {loss_prob:.1f}%")
    
    if win_prob > 65:
        print("Verdict: You are FAVORED to win")
    elif win_prob > 45:
        print("Verdict: CLOSE game, could go either way")
    else:
        print("Verdict: Opponent is FAVORED")
    print(f"{'='*40}\n")

    again = input("Predict another? (yes/no): ").strip().lower()
    if again != 'yes':
        break

print("Model trained on your real chess.com games.")