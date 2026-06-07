import requests

username = "vedantchouhan"
headers = {"User-Agent": "vedantt-chess-project"}

url = f"https://api.chess.com/pub/player/{username}/stats"
stats = requests.get(url, headers=headers).json()

print("RAPID:", stats.get('chess_rapid', {}).get('last', {}).get('rating', 'N/A'))
print("BLITZ:", stats.get('chess_blitz', {}).get('last', {}).get('rating', 'N/A'))
print("BULLET:", stats.get('chess_bullet', {}).get('last', {}).get('rating', 'N/A'))
print("DAILY:", stats.get('chess_daily', {}).get('last', {}).get('rating', 'N/A'))
