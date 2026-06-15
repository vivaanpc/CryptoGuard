from pathlib import Path

import joblib
import numpy as np

from preprocess import preprocess_text

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "naive_bayes_model.joblib"

_vectorizer = None
_model = None
_whisper_model = None


def _load_classifier():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found! Please run 'python train.py' first."
            )
        
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _model = joblib.load(MODEL_PATH)
        
    return _vectorizer, _model


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("Initializing Whisper model...")
        _whisper_model = whisper.load_model("base")
        print("Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    whisper_model = _load_whisper()
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def predict_spam(tweet_text: str = None, audio_path: str = None) -> dict:
    #If audio is provided, we prioritize it and ignore the tweet_text.

    # 1. Initialize classifier
    vectorizer, model = _load_classifier()

    audio_transcription = None
    final_text = ""

    # 2. Input selection logic
    if audio_path is not None:
        # Transcribe audio input
        audio_transcription = transcribe_audio(audio_path)
        final_text = audio_transcription
    elif tweet_text:
        # Use text input
        final_text = tweet_text
    else:
        raise ValueError("No input provided. Text or audio file required.")

    # 3. Preprocess text
    cleaned_text = preprocess_text(final_text)

    # 4. Vectorize text
    text_vector = vectorizer.transform([cleaned_text])

    # 5. Model inference
    prediction_encoded = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    # 6. Map results to labels
    prediction_label = "spam" if prediction_encoded == 1 else "ham"
    spam_probability = float(probabilities[1])
    confidence = float(max(probabilities))

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_probability, 4),
        "combined_text": final_text, # In 'one-input' mode, this is just the chosen input
        "audio_transcription": audio_transcription,
    }


#Testing
if __name__ == "__main__":
    print("=" * 20)
    print("CryptoGuard - Prediction Demo")
    print("=" * 20)
    print()

    test_cases = [
        ("SCAM TOKEN 1000x GUARANTEED! bit.ly/freecrypto", "spam"),
        ("I am Vitalik Buterin. DM me for ETH rewards. 500% returns!", "spam"),
        ("Bitcoin just broke $52k. Volume looks solid.", "ham"),
        ("The Ethereum merge was a huge technical success.", "ham"),
    ]

    for text, expected in test_cases:
        result = predict_spam(text)
        status = "PASS" if result["prediction"] == expected else "FAIL"
        
        print(f"{status}: [{result['prediction'].upper()}] (Confidence: {result['confidence']:.2%})")
        print(f"   Text: {text}")
        print()

    print("=" * 20)
    print("Demo complete.")
    print("=" * 20)
