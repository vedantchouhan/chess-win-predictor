import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

def update_games():
    username = "vedantchouhan"
    headers = {"User-Agent": "vedantt-chess-project"}
    
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    archives = requests.get(archives_url, headers=headers).json()['archives']
    
    if os.path.exists("my_games.json"):
        with open("my_games.json", "r") as f:
            all_games = json.load(f)
        archives = archives[-2:]
        print("Fetching only latest 2 months...")
    else:
        all_games = []
        print("First time — downloading all games...")
    
    new_games = []
    for archive in archives:
        response = requests.get(archive, headers=headers).json()
        new_games.extend(response.get('games', []))
    
    existing_uuids = {g['uuid'] for g in all_games}
    added = 0
    for game in new_games:
        if game['uuid'] not in existing_uuids:
            all_games.append(game)
            added += 1
    
    with open("my_games.json", "w") as f:
        json.dump(all_games, f)
    
    print(f"Added {added} new games.")
    return len(all_games)

def get_pgn_field(pgn, field):
    match = re.search(rf'\[{field} "(.+?)"\]', pgn)
    return match.group(1) if match else None

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

        pgn = game.get('pgn', '')
        eco_url = get_pgn_field(pgn, 'ECOUrl')
        if eco_url:
            opening_full = eco_url.split('/')[-1]
            opening_family = opening_full.split('-')[0]
        else:
            opening_family = 'Unknown'

        moves = re.findall(r'\d+\.', pgn)
        move_count = len(moves) if moves else 0

        data.append({
            'our_color': our_color,
            'our_rating': our_rating,
            'opp_rating': opp_rating,
            'rating_diff': our_rating - opp_rating,
            'time_class': game['time_class'],
            'opening_family': opening_family,
            'move_count': move_count,
            'outcome': outcome
        })

    df = pd.DataFrame(data)
    df.to_csv("chess_data.csv", index=False)
    return len(df)

def train_model():
    df = pd.read_csv("chess_data.csv")
    df = df[df['our_rating'] >= 500]
    df = df[df['outcome'] != 0.5]
    df['outcome'] = df['outcome'].astype(int)

    df['color_encoded'] = (df['our_color'] == 'white').astype(int)
    df['time_encoded'] = df['time_class'].map({
        'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3
    })

    le = LabelEncoder()
    df['opening_encoded'] = le.fit_transform(df['opening_family'])

    X = df[['our_rating', 'opp_rating', 'rating_diff',
            'color_encoded', 'time_encoded',
            'opening_encoded', 'move_count']]
    y = df['outcome']

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    return model, le

# ── Auto update on startup ──────────────────────────────
print("Checking for new games...")
total = update_games()
prepare_data()
model, le = train_model()
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
    opening = input("Opening (e.g. Sicilian, French, Kings, Queens): ").strip().capitalize()

    color_encoded = 1 if color == 'white' else 0
    time_encoded = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}.get(time_class, 1)
    rating_diff = our_rating - opp_rating

    # Handle unknown openings
    if opening not in le.classes_:
        opening = 'Unknown'

    opening_encoded = le.transform([opening])[0]

    features = pd.DataFrame(
        [[our_rating, opp_rating, rating_diff,
          color_encoded, time_encoded,
          opening_encoded, 0]],
        columns=['our_rating', 'opp_rating', 'rating_diff',
                 'color_encoded', 'time_encoded',
                 'opening_encoded', 'move_count']
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