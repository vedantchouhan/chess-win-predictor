from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Train model once on startup
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

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        our_rating = int(request.form['our_rating'])
        opp_rating = int(request.form['opp_rating'])
        color = request.form['color']
        time_class = request.form['time_class']
        opening = request.form['opening']

        color_encoded = 1 if color == 'white' else 0
        time_encoded = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}[time_class]
        rating_diff = our_rating - opp_rating

        if opening not in le.classes_:
            opening = 'Unknown'
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

        if win_prob > 65:
            verdict = "You are FAVORED to win"
        elif win_prob > 45:
            verdict = "CLOSE game, could go either way"
        else:
            verdict = "Opponent is FAVORED"

        result = {
            'win_prob': round(win_prob, 1),
            'loss_prob': round(loss_prob, 1),
            'verdict': verdict
        }

    return render_template('index.html', result=result, openings=openings)

if __name__ == '__main__':
    app.run(debug=True)