import requests

username = "vedantchouhan"
url = f"https://api.chess.com/pub/player/{username}/games/archives"
headers = {"User-Agent": "vedantt-chess-project"}

response = requests.get(url, headers=headers)
archives = response.json()

print(f"Total months: {len(archives['archives'])}")
for archive in archives['archives']:
    print(archive)