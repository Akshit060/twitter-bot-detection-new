from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.inference import BotDetector

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = BotDetector()

class PredictRequest(BaseModel):
    username: str

@app.post("/predict")
def predict(req: PredictRequest):
    prediction, confidence, (human_prob, bot_prob) = detector.predict(req.username)
    return {
        "username": req.username,
        "prediction": "BOT" if prediction == 1 else "HUMAN",
        "confidence": confidence,
        "bot_probability": bot_prob,
        "human_probability": human_prob
    }
