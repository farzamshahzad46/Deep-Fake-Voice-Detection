📢 Deepfake Voice Detection :

This project is a voice analysis tool that uses machine learning and deep learning to detect whether an uploaded voice recording is real or a deepfake.

It’s built with a clean and interactive Streamlit web interface, allowing users to upload audio files (.wav or .mp3) and instantly get results with a confidence score.

🧠 How It Works

🎧 Extracts MFCC audio features using Librosa

🧾 Transcribes the audio using OpenAI's Whisper

🔤 Converts the transcript into text features using TF-IDF

➕ Combines audio and text features into a single input

📊 Uses a trained XGBoost model to classify the voice

✅ Displays whether it’s a REAL or FAKE voice with confidence

📂 Project Files

streamlit.py – Frontend: handles file upload & results display

prediction.py – Backend: processes audio & runs predictions

xgb_model.pkl – Pretrained XGBoost model

tfidf_vectorizer.pkl – TF-IDF model for transcript

label_encoder.pkl – For converting predicted labels

⚙️ How to Use

Clone the repository
git clone https://github.com/yourusername/deepfake-voice-detector

Install the required libraries
pip install streamlit librosa numpy joblib openai-whisper torch scikit-learn xgboost

Run the application
streamlit run streamlit.py

Upload a voice recording (.wav or .mp3)

✅ See the result — whether it’s real or a deepfake, with a confidence score.

🧪 Want to Train Your Own Model?

Use the train_xgboost.py script with your own dataset. Just extract MFCCs, transcripts, and combine them to train a custom classifier.

🔗 Dependencies

Streamlit

Librosa

Whisper

Torch

Scikit-learn

XGBoost

Joblib

NumPy

👨‍💻 Author

Farzam Shahzad
LinkedIn: https://www.linkedin.com/in/farzam-shahzad-568024283/
