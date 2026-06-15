# ♟️ Chess Win Probability Predictor

A machine learning model trained on **3,827 of my real chess.com games** to predict win probability based on game conditions. Deployed as an interactive web app.

## 🚀 Live 
**[chess-win-predictor.onrender.com](https://chess-win-predictor.onrender.com)**

Note: hosted on free tier, first load may take 30-60 seconds if the app has been inactive.

## 🎯 What it does
Given your rating, opponent rating, color, time control, and opening, predicts your probability of winning based on patterns learned from real games.

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
- Streamlit (web interface)
- scikit-learn (Random Forest Classifier)
- pandas
- matplotlib
- Chess.com Public API
- Render (deployment)

## 📁 Project Structure
```
chess-win-predictor/
│
├── streamlit_app.py        # Main web app (Streamlit UI + model)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Theme configuration
├── fetch_games.py            # Fetch game archive URLs from chess.com API
├── download_games.py         # Download all games and save to JSON
├── explore_data.py            # Explore raw game data structure
├── prepare_data.py            # Clean and transform data into CSV
├── visualize.py                # Generate analysis charts
├── model.py                     # Train and evaluate ML model
├── predict.py                    # CLI version of predictor
└── chess_analysis.png            # Visual analysis of my chess patterns
```

## 🚀 How to Run Locally

**1. Clone and install dependencies:**
```bash
git clone https://github.com/vedantchouhan/chess-win-predictor.git
cd chess-win-predictor
pip install -r requirements.txt
```

**2. Run the web app:**
```bash
streamlit run streamlit_app.py
```

**3. (Optional) Use the CLI predictor:**
```bash
python predict.py
```

## 💡 Example Prediction
```
Your rating: 1152
Opponent rating: 1000
Color: white
Time class: rapid
Opening: London

Win Probability:  84.7%
Loss Probability: 15.3%
Verdict: You are FAVORED to win
```

## 📈 Future Improvements
- Add opponent's opening tendencies
- Position evaluation features for deeper game analysis
- Real-time prediction during live games
- Game history tracking and trend visualization

## 👤 Author
**Vedant Chouhan**
B.Tech CSE (AI/ML) — UPES Dehradun
[chess.com/member/vedantchouhan](https://www.chess.com/member/vedantchouhan)
