import os
import librosa
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from xgboost import XGBClassifier
import torch

# Constants
SR = 16000
N_MFCC = 13
WHISPER_MODEL = "medium"

# Load models and vectorizers
model = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(WHISPER_MODEL).to(device)

# Unlimited manual recording
def record_audio(sr=SR):
    print("🎙️ Recording... Press ENTER to stop recording.")
    duration = []
    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, callback=callback):
        input()  # Wait for ENTER key to stop recording
    audio_data = np.concatenate(recording, axis=0)
    
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio_data, sr)
    print(f"✅ Recording saved to: {temp_path}")
    return temp_path

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        result = whisper_model.transcribe(file_path, language='en', fp16=(device=="cuda"))
        transcript = result.get('text', '').strip()

        return mfcc_mean, transcript
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def predict_and_explain(file_path):
    mfcc_feat, transcript = extract_features(file_path)
    if mfcc_feat is None or transcript.strip() == "":
        print("⚠️ Failed to extract features.")
        return

    tfidf_vec = tfidf.transform([transcript])
    mfcc_vec = np.array(mfcc_feat).reshape(1, -1)
    combined = hstack([tfidf_vec, mfcc_vec]).toarray()

    prediction = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    confidence = proba[prediction]

    print(f"\n🧠 Prediction: {'FAKE' if label == 1 else 'REAL'} ({confidence:.2%} confidence)")

    # Detailed Reasoning
    duration = librosa.get_duration(filename=file_path)
    num_words = len(transcript.split())
    speaking_rate = duration / (num_words + 1e-6)

    reasons = []

    if num_words == 0:
        reasons.append("No meaningful speech content was detected.")
    else:
        reasons.append(f"Transcript has {num_words} words over {duration:.2f} seconds (Rate: {speaking_rate:.2f} sec/word).")

        if speaking_rate < 0.20:
            reasons.append("Speaking rate is unnaturally fast, possibly indicating artificial generation.")
        elif speaking_rate > 0.65:
            reasons.append("Speaking rate is very slow and inconsistent, which is sometimes used to hide synthesis artifacts.")

        if any(w in transcript.lower() for w in ['uh', 'um', 'hmm']):
            reasons.append("Natural filler words detected (e.g., 'uh', 'um'), often seen in real speech.")
        else:
            reasons.append("Speech flow is too smooth and lacks natural hesitation, a sign of synthetic audio.")

        if len(set(transcript.lower().split())) < max(3, num_words * 0.3):
            reasons.append("Low vocabulary diversity could imply script-based or synthesized speech.")

        if transcript.lower().count('.') > num_words * 0.05:
            reasons.append("Unusual punctuation density might indicate artificial segmentation.")

    print("\n🔍 Detailed Reasoning:")
    for i, r in enumerate(reasons, 1):
        print(f"{i}. {r}")

    print(f"\n🧾 Final Verdict: {'FAKE' if label == 1 else 'REAL'}")

def main():
    print("🎛️ Choose input method:")
    print("1. Record audio")
    print("2. Provide audio file path")
    choice = input("Your choice (1/2): ")

    if choice == '1':
        print("➡️ Press ENTER to start recording, then ENTER again to stop.")
        input("Press ENTER to start...")
        file_path = record_audio()
    elif choice == '2':
        file_path = input("📁 Enter full path to audio file: ").strip('"')
        if not os.path.isfile(file_path):
            print("❌ File not found. Please check the path and try again.")
            return
    else:
        print("❌ Invalid choice.")
        return

    predict_and_explain(file_path)

if __name__ == "__main__":
    main()
