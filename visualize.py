import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("chess_data.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Vedant's Chess Analysis", fontsize=16)

axes[0,0].pie(
    [df['outcome'].eq(1).sum(), df['outcome'].eq(0).sum(), df['outcome'].eq(0.5).sum()],
    labels=['Win', 'Loss', 'Draw'],
    colors=['green', 'red', 'grey'],
    autopct='%1.1f%%'
)
axes[0,0].set_title('Overall Results')

# 2. Win rate by time class
time_classes = df.groupby('time_class')['outcome'].apply(
    lambda x: (x == 1).sum() / len(x) * 100
)
axes[0,1].bar(time_classes.index, time_classes.values, color=['blue','orange','green','red'])
axes[0,1].set_title('Win Rate by Time Class')
axes[0,1].set_ylabel('Win Rate %')
axes[0,1].set_ylim(0, 100)

# 3. Win rate by color
color_wins = df.groupby('our_color')['outcome'].apply(
    lambda x: (x == 1).sum() / len(x) * 100
)
axes[1,0].bar(color_wins.index, color_wins.values, color=['#ADD8E6', '#4a4a4a'], edgecolor='grey')
axes[1,0].set_title('Win Rate by Color')
axes[1,0].set_ylabel('Win Rate %')
axes[1,0].set_ylim(0, 100)

# 4. Rating diff vs outcome
axes[1,1].scatter(
    df[df['outcome']==1]['rating_diff'],
    [1]*df['outcome'].eq(1).sum(),
    alpha=0.3, color='green', label='Win'
)
axes[1,1].scatter(
    df[df['outcome']==0]['rating_diff'],
    [0]*df['outcome'].eq(0).sum(),
    alpha=0.3, color='red', label='Loss'
)
axes[1,1].set_title('Rating Difference vs Outcome')
axes[1,1].set_xlabel('Rating Difference (You - Opponent)')
axes[1,1].set_ylabel('Outcome')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('chess_analysis.png', dpi=150)
plt.show()
print("Saved as chess_analysis.png")