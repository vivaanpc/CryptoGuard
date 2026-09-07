# 🛡️ CryptoGuard: AI-Powered Spam Detection

CryptoGuard is a specialized machine learning pipeline and API designed to identify and filter cryptocurrency-related spam. It leverages Natural Language Processing (NLP) to analyze text and utilizes OpenAI's Whisper for transcribing audio inputs, providing a robust defense against common crypto scams, "shilling," and fraudulent promotions.

---

## 🚀 Key Features

-   **Dual-Input Analysis:** Process both raw text (tweets, DMs, forum posts) and audio files.
-   **Audio Transcription:** Integrated with **OpenAI Whisper** for high-accuracy speech-to-text conversion.
-   **ML Classification:** Utilizes a **Multinomial Naive Bayes** model trained on a specialized crypto-spam dataset.
-   **RESTful API:** Powered by **FastAPI** for high-performance, asynchronous inference.
-   **Modern UI:** A clean, minimal dark-themed frontend for manual testing and demonstration.
-   **Container Ready:** Includes a Dockerfile for streamlined deployment.

---

## 🛠️ Tech Stack

-   **Backend:** FastAPI (Python)
-   **Machine Learning:** Scikit-learn (TF-IDF Vectorization + Naive Bayes)
-   **Audio Processing:** OpenAI Whisper
-   **Frontend:** Vanilla HTML5 / CSS3 / JavaScript
-   **Inference Server:** Uvicorn
-   **Serialization:** Joblib

---

## 📦 Project Structure

```text
crypto-guard/
├── backend/
│   ├── app.py             # FastAPI entry point
│   ├── predict.py         # Inference logic & Whisper integration
│   ├── preprocess.py      # Text normalization & cleaning
│   └── models/            # Serialized ML artifacts (.joblib)
├── training/
│   ├── train.py           # Model training script
│   └── data/              # Training datasets (CSV)
├── frontend/
│   └── index.html         # Web-based demo interface
├── requirements.txt       # Project dependencies
└── Dockerfile             # Container configuration
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vivaanpc/crypto-guard.git
cd crypto-guard
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model
Ensure you have the dataset in `training/data/` then run:
```bash
python training/train.py
```
This will generate `tfidf_vectorizer.joblib` and `naive_bayes_model.joblib` in the `backend/models/` directory.

### 4. Run the Application
```bash
python backend/app.py
```
The API will be available at `http://localhost:7860`.

---

## 🖥️ Usage

### Web Interface
Navigate to `http://localhost:7860` in your browser to use the interactive dashboard. You can paste text or upload an audio file to check for spam.

### API Endpoint
**POST** `/predict`

**Form Data:**
- `tweet` (string, optional): The text content to analyze.
- `audio` (file, optional): An audio file to transcribe and analyze.

**Example Response:**
```json
{
  "success": true,
  "prediction": "spam",
  "confidence": 0.9842,
  "spam_probability": 0.9842,
  "audio_transcription": "FREE BITCOIN GIVEAWAY! CLICK LINK..."
}
```

---

## 🐳 Docker Deployment

To run CryptoGuard using Docker:

```bash
docker build -t crypto-guard .
docker run -p 7860:7860 crypto-guard
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
