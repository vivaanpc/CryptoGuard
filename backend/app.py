import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predict import predict_spam

# Static files
SCRIPT_DIR = Path(__file__).parent
FRONTEND_DIR = SCRIPT_DIR.parent / "frontend"

app = FastAPI(
    title="CryptoGuard Spam Classifier API",
    description="API for identifying crypto spam.",
    version="1.0.0",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "CryptoGuard API is operational. Index file not found."}


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "message": "Service is operational.",
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict_endpoint(
    tweet: str = Form(None, description="The tweet text you want to check"),
    audio: UploadFile = File(None, description="Optional audio file for transcription"),
):
    audio_path = None

    try:
        # Handle audio
        if audio is not None and audio.filename:
            suffix = Path(audio.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await audio.read()
                tmp.write(content)
                audio_path = tmp.name

        result = predict_spam(tweet_text=tweet, audio_path=audio_path)

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "spam_probability": result["spam_probability"],
            "combined_text": result["combined_text"],
            "audio_transcription": result["audio_transcription"],
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "Model artifacts not found.",
            "hint": "Please run 'python training/train.py' to train the model.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}",
        }
    finally:
        # Cleanup temporary audio files
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    import uvicorn

    print()
    print("Starting CryptoGuard API...")
    print("API Documentation: http://localhost:7860/docs")
    print()
    uvicorn.run(app, host="0.0.0.0", port=7860)
