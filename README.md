# ♟️ Chess Win Probability Predictor

A machine learning model trained on **3,827 of my real chess.com games** to predict win probability based on game conditions.

## 🎯 What it does
Given your rating, opponent rating, color, and time control — predicts your probability of winning based on historical patterns from real games.

## 📊 Model Performance
- **Accuracy: 79.1%** on unseen games
- Trained on 3,827 personal chess.com games (rated 500+)
- Spans 2021–2026 across all time controls
- Peak rating: 1639 Rapid | Current Daily: 1705

## 📈 Model Progression
| Version | Accuracy | Changes |
|---|---|---|
| v1 | 73.4% | Basic features (rating, color, time) |
| v2 | 76.2% | Added opening name + move count |
| v3 | 78.4% | Filtered noisy low-rated games |
| v4 | 79.1% | Hyperparameter tuning (final) |

## 🔍 Key Findings from My Games
| Time Control | Win Rate |
|---|---|
| Daily | 65% |
| Rapid | 54% |
| Blitz | 50% |
| Bullet | 50% |

- Rating difference is the strongest predictor **(38% feature importance)**
- Move count is significant **(15% feature importance)** — longer games favor me
- Opening choice matters **(9.5% feature importance)**
- Playing white gives slight advantage **(53% vs 50%)**
- Low draw rate **(2.4%)** — aggressive playing style

## 🛠️ Tech Stack
- Python
- scikit-learn (Random Forest Classifier)
- pandas
- matplotlib
- Chess.com Public API

## 📁 Project Structure
```
chess-win-predictor/
│
├── fetch_games.py          # Fetch game archive URLs from chess.com API
├── download_games.py       # Download all games and save to JSON
├── explore_data.py         # Explore raw game data structure
├── prepare_data.py         # Clean and transform data into CSV
├── visualize.py            # Generate analysis charts
├── model.py                # Train and evaluate ML model
├── predict.py              # Interactive win probability predictor
└── chess_analysis.png      # Visual analysis of my chess patterns
```

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install requests pandas scikit-learn matplotlib
```

**2. Download your games:**
```bash
python download_games.py
```

**3. Prepare data:**
```bash
python prepare_data.py
```

**4. Train model:**
```bash
python model.py
```

**5. Predict your next game:**
```bash
python predict.py
```

## 💡 Example Prediction
```
Your rating: 1152
Opponent rating: 1000
Color: white
Time class: rapid

Win Probability:  91.0%
Loss Probability: 9.0%
Verdict: You are FAVORED to win
```

## 📈 Future Improvements
- Build a web interface using Flask
- Add opponent's opening tendencies
- Include position evaluation at move 10
- Real-time prediction during live games

## 👤 Author
**Vedant Chouhan**
B.Tech CSE (AI/ML) — UPES Dehradun
[chess.com/member/vedantchouhan](https://www.chess.com/member/vedantchouhan)