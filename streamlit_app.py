import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Chess Win Predictor",
    page_icon="♟️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, header, footer {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #131829 50%, #0d1117 100%);
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 700px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.hero-title {
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f0d9b5 0%, #d4a373 50%, #b58863 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    text-align: center;
    color: #8b92a8;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}

.stat-tile {
    background: rgba(181, 136, 99, 0.08);
    border: 1px solid rgba(181, 136, 99, 0.2);
    border-radius: 14px;
    padding: 16px 8px;
    text-align: center;
    transition: all 0.3s ease;
}

.stat-tile:hover {
    background: rgba(181, 136, 99, 0.15);
    transform: translateY(-2px);
}

.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0d9b5;
    line-height: 1.2;
}

.stat-cap {
    font-size: 0.65rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

label, .stSelectbox label, .stNumberInput label {
    color: #b58863 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

.stNumberInput input, .stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f5f5f5 !important;
}

.stNumberInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: #b58863 !important;
    box-shadow: 0 0 0 1px #b58863 !important;
}

.stFormSubmitButton button {
    width: 100%;
    background: linear-gradient(135deg, #b58863 0%, #8b5e34 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(181, 136, 99, 0.3) !important;
}

.stFormSubmitButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(181, 136, 99, 0.5) !important;
}

.result-label {
    text-align: center;
    color: #6b7280;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 1rem;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}

.prob-label {
    width: 50px;
    font-size: 0.85rem;
    color: #8b92a8;
    font-weight: 500;
}

.prob-track {
    flex: 1;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    height: 26px;
    overflow: hidden;
    position: relative;
}

.prob-fill-win {
    height: 100%;
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.prob-fill-loss {
    height: 100%;
    background: linear-gradient(90deg, #c62828, #ef5350);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.verdict {
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-top: 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
}

.verdict-win {
    background: linear-gradient(135deg, rgba(46, 125, 50, 0.2), rgba(102, 187, 106, 0.1));
    border: 1px solid rgba(102, 187, 106, 0.4);
    color: #81c784;
}

.verdict-close {
    background: linear-gradient(135deg, rgba(245, 124, 0, 0.2), rgba(255, 213, 79, 0.1));
    border: 1px solid rgba(255, 213, 79, 0.4);
    color: #ffd54f;
}

.verdict-loss {
    background: linear-gradient(135deg, rgba(198, 40, 40, 0.2), rgba(239, 83, 80, 0.1));
    border: 1px solid rgba(239, 83, 80, 0.4);
    color: #ef9a9a;
}

.footer-text {
    text-align: center;
    color: #4b5563;
    font-size: 0.8rem;
    margin-top: 2rem;
}

.footer-text a {
    color: #b58863;
    text-decoration: none;
    font-weight: 500;
}

.chess-divider {
    display: flex;
    justify-content: center;
    gap: 8px;
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
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
        n_estimators=200, max_depth=10,
        min_samples_split=10, min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    return model, le

model, le = train_model()
openings = sorted(le.classes_.tolist())

st.markdown('<div class="chess-divider">♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Chess Win Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Machine learning model trained on 3,827 real games · 79.1% accuracy</p>', unsafe_allow_html=True)

st.markdown(f'''
<div class="stat-grid">
    <div class="stat-tile"><div class="stat-num">3,827</div><div class="stat-cap">Games</div></div>
    <div class="stat-tile"><div class="stat-num">79.1%</div><div class="stat-cap">Accuracy</div></div>
    <div class="stat-tile"><div class="stat-num">1639</div><div class="stat-cap">Peak Elo</div></div>
    <div class="stat-tile"><div class="stat-num">51.6%</div><div class="stat-cap">Win Rate</div></div>
</div>
''', unsafe_allow_html=True)

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        our_rating = st.number_input("Your rating", min_value=100, max_value=3000, value=1152, step=10)
    with c2:
        opp_rating = st.number_input("Opponent rating", min_value=100, max_value=3000, value=1000, step=10)

    c3, c4 = st.columns(2)
    with c3:
        color = st.selectbox("Your color", ["White", "Black"])
    with c4:
        time_class = st.selectbox("Time control", ["Rapid", "Blitz", "Bullet", "Daily"])

    opening = st.selectbox("Opening (type to search)", openings)

    submitted = st.form_submit_button("Predict win probability")

if submitted:
    color_encoded = 1 if color == "White" else 0
    time_encoded = {"Bullet": 0, "Blitz": 1, "Rapid": 2, "Daily": 3}[time_class]
    rating_diff = our_rating - opp_rating

    if opening not in le.classes_:
        opening = "Unknown"
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

    st.markdown('<div class="result-label">Prediction result</div>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="prob-row">
        <div class="prob-label">Win</div>
        <div class="prob-track">
            <div class="prob-fill-win" style="width: {win_prob}%;">{win_prob:.1f}%</div>
        </div>
    </div>
    <div class="prob-row">
        <div class="prob-label">Loss</div>
        <div class="prob-track">
            <div class="prob-fill-loss" style="width: {loss_prob}%;">{loss_prob:.1f}%</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if win_prob > 65:
        st.markdown('<div class="verdict verdict-win">♔ You are favored to win</div>', unsafe_allow_html=True)
    elif win_prob > 45:
        st.markdown('<div class="verdict verdict-close">♞ Close game — could go either way</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict verdict-loss">♚ Opponent is favored</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footer-text">Built by Vedant Chouhan · '
    '<a href="https://github.com/vedantchouhan/chess-win-predictor">View on GitHub</a></p>',
    unsafe_allow_html=True
)