import sys
from pathlib import Path

# Add the project root to the search path so we can import from backend
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from backend.preprocess import preprocess_text

DATA_PATH = SCRIPT_DIR / "data" / "crypto_spam_dataset.csv"
MODELS_DIR = SCRIPT_DIR.parent / "backend" / "models"


def load_and_prepare_data(data_path: Path) -> tuple:
    print()
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print("Preprocessing data...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    df["label_encoded"] = (df["label"] == "spam").astype(int)

    return df["cleaned_text"].values, df["label_encoded"].values


def train_model(X_text: np.ndarray, y: np.ndarray) -> tuple:
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train_text)} samples. Testing on {len(X_test_text)} samples.")
    print()

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print()

    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Training complete. Test accuracy: {accuracy:.2%}")
    print()

    return model, vectorizer, X_test, y_test, y_pred


def print_evaluation(model, vectorizer, X_test, y_test, y_pred):
    print("Report:")
    target_names = ["ham (legit)", "spam"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   - Correctly caught Spam: {cm[1][1]}")
    print(f"   - Correctly let Ham in:  {cm[0][0]}")
    print(f"   - Marked Ham as Spam: {cm[0][1]}")
    print(f"   - Let Spam slip through: {cm[1][0]}")
    print()



def save_artifacts(model, vectorizer, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    model_path = models_dir / "naive_bayes_model.joblib"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    print(f"Model artifacts saved to: {models_dir}")
    print()


def main():
    if not DATA_PATH.exists():
        print(f"Data file not found at: {DATA_PATH}")
        print("Verify that the 'data' directory contains the required CSV file.")
        sys.exit(1)


    X_text, y = load_and_prepare_data(DATA_PATH)
    model, vectorizer, X_test, y_test, y_pred = train_model(X_text, y)
    print_evaluation(model, vectorizer, X_test, y_test, y_pred)
    save_artifacts(model, vectorizer, MODELS_DIR)

if __name__ == "__main__":
    main()
