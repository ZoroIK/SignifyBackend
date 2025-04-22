import os
import requests
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ Define ASL Alphabet Mapping
ASL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

# 🎯 FastAPI App
app = FastAPI()

# ✅ CORS Middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update if deployed
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ✅ Model URL & File Path
MODEL_URL = "https://rfc.s3.us-east.cloud-object-storage.appdomain.cloud/asl_cnn_2D_model.h5"
MODEL_PATH = "asl_cnn_2D_model.h5"

# 🔽 Download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        if "text/html" in response.headers.get("Content-Type", ""):
            print("❌ Invalid model URL!")
            return False
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("✅ Model downloaded.")
        if os.path.getsize(MODEL_PATH) < 100000:
            print("❌ File too small. Download may have failed.")
            return False
    return True

# ⬇ Try downloading model
if not download_model():
    exit(1)

# ✅ Load the TensorFlow model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 2D Model Loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# 📦 Input schema
class LandmarkData(BaseModel):
    landmarks: list  # 42 values expected (21 points * x and y)

# 🧠 Prediction Endpoint
@app.post("/predict")
async def predict(data: LandmarkData):
    if len(data.landmarks) != 42:
        raise HTTPException(status_code=400, detail="Expected 42 landmark values (x, y for 21 points).")

    try:
        input_data = np.array([data.landmarks])
        predictions = model.predict(input_data, verbose=0)[0]
        label_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        # Apply confidence threshold
        if confidence < 0.2:
            return {"prediction": "Unknown", "confidence": confidence}

        predicted_letter = ASL_LETTERS[label_index] if label_index < len(ASL_LETTERS) else "?"

        return {
            "prediction": predicted_letter,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# ▶ Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
