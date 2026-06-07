import json
import pandas as pd

with open("my_games.json", "r") as f:
    games = json.load(f)

our_username = "vedantchouhan"
data = []

for game in games:
    # Skip non-standard games
    if game['rules'] != 'chess':
        continue

    white = game['white']['username'].lower()
    black = game['black']['username'].lower()

    # Determine our color
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

    # Win = 1, Loss = 0, Draw = 0.5
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
print(df.head(10))
print(f"\nTotal usable games: {len(df)}")
print(f"\nWin rate: {(df['outcome'] == 1).sum() / len(df) * 100:.1f}%")
print(f"Loss rate: {(df['outcome'] == 0).sum() / len(df) * 100:.1f}%")
print(f"Draw rate: {(df['outcome'] == 0.5).sum() / len(df) * 100:.1f}%")
print(f"\nTime class distribution:\n{df['time_class'].value_counts()}")

df.to_csv("chess_data.csv", index=False)
print("\nSaved to chess_data.csv")