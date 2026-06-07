import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def update_games():
    username = "vedantchouhan"
    headers = {"User-Agent": "vedantt-chess-project"}
    
    # Get all archives
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    archives = requests.get(archives_url, headers=headers).json()['archives']
    
    # Load existing games if file exists
    if os.path.exists("my_games.json"):
        with open("my_games.json", "r") as f:
            all_games = json.load(f)
        
        # Only fetch last 2 months — current and previous
        archives = archives[-2:]
        print(f"Fetching only latest 2 months...")
    else:
        all_games = []
        print("First time — downloading all games...")
    
    # Download only recent archives
    new_games = []
    for archive in archives:
        response = requests.get(archive, headers=headers).json()
        new_games.extend(response.get('games', []))
    
    # Merge — remove duplicates by uuid
    existing_uuids = {g['uuid'] for g in all_games}
    added = 0
    for game in new_games:
        if game['uuid'] not in existing_uuids:
            all_games.append(game)
            added += 1
    
    # Save
    with open("my_games.json", "w") as f:
        json.dump(all_games, f)
    
    print(f"Added {added} new games.")
    return len(all_games)

def prepare_data():
    with open("my_games.json", "r") as f:
        games = json.load(f)

    our_username = "vedantchouhan"
    data = []

    for game in games:
        if game['rules'] != 'chess':
            continue
        white = game['white']['username'].lower()
        black = game['black']['username'].lower()

        if white == our_username.lower():
            our_color = 'white'
            our_rating = game['white']['rating']
            opp_rating = game['black']['rating']
            our_result = game['white']['result']
        elif black == our_username.lower():
            our_color = 'black'
            our_rating = game['black']['rating']
            opp_rating = game['white']['rating']
            our_result = game['black']['result']
        else:
            continue

        if our_result == 'win':
            outcome = 1
        elif our_result in ['drawn', 'stalemate', 'agreed', 'repetition', 'insufficient']:
            outcome = 0.5
        else:
            outcome = 0

        data.append({
            'our_color': our_color,
            'our_rating': our_rating,
            'opp_rating': opp_rating,
            'rating_diff': our_rating - opp_rating,
            'time_class': game['time_class'],
            'outcome': outcome
        })

    df = pd.DataFrame(data)
    df.to_csv("chess_data.csv", index=False)
    return len(df)

def train_model():
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
    return model

# ── Auto update on startup ──────────────────────────────
print("Checking for new games...")
total = update_games()
prepare_data()
model = train_model()
print(f"Model updated with {total} games. Ready.\n")

# ── Prediction loop ─────────────────────────────────────
print("=== Chess Win Probability Predictor ===")
print("Based on your real chess.com games\n")

while True:
    print("Enter game details:")
    our_rating = int(input("Your rating: "))
    opp_rating = int(input("Opponent rating: "))
    color = input("Your color (white/black): ").strip().lower()
    time_class = input("Time class (bullet/blitz/rapid/daily): ").strip().lower()

    color_encoded = 1 if color == 'white' else 0
    time_encoded = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}.get(time_class, 1)
    rating_diff = our_rating - opp_rating

    features = pd.DataFrame(
        [[our_rating, opp_rating, rating_diff, color_encoded, time_encoded]],
        columns=['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded']
    )

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

print("Done.")