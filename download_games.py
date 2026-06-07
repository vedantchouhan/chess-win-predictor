import requests
import json

username = "vedantchouhan"
headers = {"User-Agent": "vedantt-chess-project"}

# Get all archive URLs
archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
archives = requests.get(archives_url, headers=headers).json()['archives']

# Download all games
all_games = []
for archive in archives:
    print(f"Fetching: {archive}")
    response = requests.get(archive, headers=headers).json()
    all_games.extend(response.get('games', []))

print(f"\nTotal games downloaded: {len(all_games)}")

# Save to file
with open("my_games.json", "w") as f:
    json.dump(all_games, f)

print("Saved to my_games.json")