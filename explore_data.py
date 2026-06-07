
import json

with open("my_games.json", "r") as f:
    games = json.load(f)

# Look at one game structure
print("Total games:", len(games))
print("\nOne game keys:", games[0].keys())
print("\nSample game:")
print("White:", games[0]['white']['username'])
print("Black:", games[0]['black']['username'])
print("Result:", games[0]['white']['result'])
print("White rating:", games[0]['white']['rating'])
print("Black rating:", games[0]['black']['rating'])
print("Time class:", games[0]['time_class'])
print("Rules:", games[0]['rules'])


print(games[0]['pgn'][:500])
print("\nECO:", games[0].get('eco', 'N/A'))