from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize API and allow frontend to talk to it
app = FastAPI(title="Chess Win Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your HTML frontend to make requests
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load pre-trained model (Instant boot!)
# Ensure 'chess_model.pkl' and 'label_encoder.pkl' are in the same folder
model = joblib.load("chess_model.pkl")
le = joblib.load("label_encoder.pkl")

# 3. Define the expected incoming data format from the frontend
class PredictionRequest(BaseModel):
    our_rating: int
    opp_rating: int
    color: str
    time_class: str
    opening: str

# 4. Create the Prediction Endpoint
@app.post("/predict")
def predict_win(req: PredictionRequest):
    # Encode inputs to match training data
    color_encoded = 1 if req.color.lower() == "white" else 0
    time_encoded = {"bullet": 0, "blitz": 1, "rapid": 2, "daily": 3}.get(req.time_class.lower(), 1)
    rating_diff = req.our_rating - req.opp_rating

    # Handle the opening
    opening = req.opening.capitalize()
    if opening not in le.classes_:
        opening = "Unknown"
    opening_encoded = le.transform([opening])[0]

    # Format into a DataFrame exactly how the Gradient Boosting model expects it
    features = pd.DataFrame(
        [[req.our_rating, req.opp_rating, rating_diff, color_encoded, time_encoded, opening_encoded]],
        columns=['our_rating', 'opp_rating', 'rating_diff', 'color_encoded', 'time_encoded', 'opening_encoded']
    )

    # Make Prediction
    win_prob = model.predict_proba(features)[0][1] * 100
    loss_prob = model.predict_proba(features)[0][0] * 100

    # Return as JSON to the frontend
    return {
        "win_prob": round(win_prob, 1),
        "loss_prob": round(loss_prob, 1)
    }

# 5. Fetch available openings for the frontend dropdown
@app.get("/openings")
def get_openings():
    return {"openings": sorted(le.classes_.tolist())}