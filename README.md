
<div align="center">

# ♟️ Chess Win Predictor

[![Live App](https://img.shields.io/badge/Live_App-Play_Now-2ea44f?style=for-the-badge&logo=github)](https://vedantchouhan.github.io/chess-win-predictor/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

*An end-to-end Machine Learning web application trained on 3,827 real games from my personal Chess.com history to predict match outcomes before the first move is played.*

</div>

---

## 🚀 Overview

This project is a full-stack machine learning pipeline that answers one question: *Based on historical data, what is the exact probability I will win this specific chess match?*

It evaluates pre-game parameters (my rating, opponent rating, time control, color, and opening) and serves live predictions through a decoupled web architecture.

**Key Metrics:**
* **Dataset:** 3,827 personal Chess.com matches (extracted via Chess.com Public API).
* **Model:** Gradient Boosting Classifier.
* **True Accuracy:** **77.4%** on unseen test data.

---

## 🧠 Machine Learning Engineering

### Model Evolution & Data Leakage Prevention
Initial prototypes using a Random Forest algorithm achieved an artificially inflated 79.1% accuracy. Through feature importance analysis, I identified a severe **data leakage** issue: the model was relying heavily on `move_count`. Because a game's total move count cannot be known *before* a game starts, passing `move_count = 0` during live inference caused blind guessing.

**The Fix:**
1. Dropped the `move_count` feature to ensure strict pre-game temporal validity.
2. Upgraded the algorithm to a **Gradient Boosting Classifier**, successfully recovering predictive power and achieving a mathematically sound **77.4% Test Accuracy** on purely pre-game variables.

---

## ⚙️ Architecture

The application uses a modern, decoupled ML architecture to eliminate the "cold-start" latency commonly found in monolithic Streamlit deployments.

    [Client Browser] <--> [GitHub Pages CDN]
           |
     (REST API JSON)
           |
    [FastAPI Backend] --> [Gradient Boosting Model .pkl]
    (Hosted on Render)

* **Frontend (GitHub Pages):** A lightweight, serverless HTML/JS interface that loads instantly (<1s) globally. Features live API-driven autocomplete for opening moves and dynamic rating difference calculators.
* **Backend (Render):** A FastAPI REST service that loads the pre-trained `.pkl` model into memory on boot. This handles inference requests seamlessly without retraining the model, reducing latency from ~50 seconds (in the v1 Streamlit monolith) to milliseconds.

---

## 🛠️ Quickstart (Run Locally)

Want to run the API and test the model on your own machine?

**1. Clone the repository & install dependencies**

    git clone https://github.com/vedantchouhan/chess-win-predictor.git
    cd chess-win-predictor
    pip install -r requirements.txt

**2. Start the FastAPI Backend**

    uvicorn main:app --reload

*The API will boot up at `http://127.0.0.1:8000`. You can view the interactive Swagger docs at `http://127.0.0.1:8000/docs`.*

**3. Launch the Frontend**
Open a new terminal tab and start a local web server:

    python3 -m http.server 3000

*Navigate to `http://127.0.0.1:3000` in your browser to interact with the UI.*

---

## 📁 Repository Structure

    chess-win-predictor/
    ├── main.py                 # FastAPI backend entry point and inference routes
    ├── train and save.py       # Data pipeline, model training, and .pkl generation
    ├── index.html              # Frontend user interface and API integration logic
    ├── requirements.txt        # Python dependencies
    ├── chess_model.pkl         # Serialized Gradient Boosting model
    ├── label_encoder.pkl       # Serialized encoder for chess openings
    ├── download_games.py       # ETL script to fetch user history from Chess.com API
    ├── prepare_data.py         # Data cleaning and feature engineering script
    └── chess_data.csv          # Cleaned dataset of 3,827 games

## 👤 Author
**Vedant Chouhan**
B.Tech CSE (AI/ML) — UPES Dehradun
[chess.com/member/vedantchouhan](https://www.chess.com/member/vedantchouhan)
