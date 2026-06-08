import json
import pandas as pd
import re

with open("my_games.json", "r") as f:
    games = json.load(f)

our_username = "vedantchouhan"
data = []

def get_pgn_field(pgn, field):
    match = re.search(rf'\[{field} "(.+?)"\]', pgn)
    return match.group(1) if match else None

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

    # Extract opening info from PGN
    pgn = game.get('pgn', '')
    eco = get_pgn_field(pgn, 'ECO')
    eco_url = get_pgn_field(pgn, 'ECOUrl')
    
    # Extract opening family — first word before any dash
    if eco_url:
        opening_full = eco_url.split('/')[-1]  # e.g. Scandinavian-Defense-Mieses
        opening_family = opening_full.split('-')[0]  # e.g. Scandinavian
    else:
        opening_family = 'Unknown'

    # Extract move count from PGN
    moves = re.findall(r'\d+\.', pgn)
    move_count = len(moves) if moves else 0

    data.append({
        'our_color': our_color,
        'our_rating': our_rating,
        'opp_rating': opp_rating,
        'rating_diff': our_rating - opp_rating,
        'time_class': game['time_class'],
        'eco': eco if eco else 'Unknown',
        'opening_family': opening_family,
        'move_count': move_count,
        'outcome': outcome
    })

df = pd.DataFrame(data)

print(f"Total games: {len(df)}")
print(f"\nTop 10 openings you play:")
print(df['opening_family'].value_counts().head(10))
print(f"\nWin rate by opening family (min 20 games):")
opening_stats = df.groupby('opening_family').agg(
    games=('outcome', 'count'),
    win_rate=('outcome', lambda x: (x==1).sum()/len(x)*100)
).query('games >= 20').sort_values('win_rate', ascending=False)
print(opening_stats.head(10))

df.to_csv("chess_data.csv", index=False)
print("\nSaved to chess_data.csv")