import os
import re
import json
import base64
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from email import message_from_string
from email.policy import default
import nltk
from nltk.corpus import stopwords
import pytz

import binascii
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

INDIA_TZ = pytz.timezone("Asia/Kolkata")

# Initialization
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
stop_words = set(stopwords.words('english'))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Category Mapping for BDS
CATEGORY_MAP = {
    0: "Clean",
    1: "EmlIsSpam",
    2: "EmlIsPhishing",
    4: "EmlIsMalware",
    8: "EmlIsMarketing"
}

# Load models using simple direct paths
nb_model = joblib.load("model/nb_model.pkl")
lr_model = joblib.load("model/lr_model.pkl")
lgb_model = joblib.load("model/lgb_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
le = joblib.load("model/label_encoder.pkl")


SERVER_AES_KEY = b"wgrod1vkpk51gwuwgrod1vkpk51gwuwu"

def decrypt_raw_cbc_from_body(raw_body: str, iv: bytes):
    """Decrypt your corrupted ciphertext body using AES-256-CBC."""

    # Interpret your weird garbage ciphertext as latin-1 bytes
    ciphertext = raw_body.encode("latin-1", errors="ignore")

    # Auto-pad corrupted ciphertext to nearest AES block
    if len(ciphertext) % 16 != 0:
        pad_len = 16 - (len(ciphertext) % 16)
        ciphertext += b"\x00" * pad_len

    cipher = AES.new(SERVER_AES_KEY, AES.MODE_CBC, iv)
    pt = cipher.decrypt(ciphertext)

    try:
        pt = unpad(pt, AES.block_size)
    except:
        pass

    return pt.decode("latin-1", errors="ignore")

# MIME Email Extraction
def extract_subject_body_and_attachments(email_json):
    if isinstance(email_json, str):
        email_data = json.loads(email_json)
    else:
        email_data = email_json

    raw_email = email_data.get("email", "")
    if not raw_email:
        return {"subject": None, "body": None, "attachments": []}

    msg = message_from_string(raw_email, policy=default)
    subject = msg.get("Subject", "No Subject")

    body = ""
    attachments = []

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()
        filename = part.get_filename()

        if content_type == "text/plain" and content_disposition != "attachment":
            try:
                body = part.get_content().strip()
            except:
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore").strip()

        elif content_disposition == "attachment" or filename:
            payload = part.get_payload(decode=True)
            if payload:
                attachments.append({
                    "filename": filename,
                    "content": base64.b64encode(payload).decode("utf-8")
                })

    body = body.strip()
    return {
        "subject": subject.strip(),
        "body": body.strip(),
        "attachments": attachments
    }

# Text Preprocessing
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '<URL>', text)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    text = re.sub(r'[^a-z0-9\s<>]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def compute_features(content_list: list) -> np.ndarray:
    df = pd.DataFrame({'content': content_list})
    df['num_links'] = df['content'].str.count(r'http\S+|www\S+')
    df['num_exclamations'] = df['content'].str.count('!')
    df['num_uppercase_words'] = df['content'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['text_length'] = df['content'].str.len()
    df['num_special_chars'] = df['content'].str.count(r'[^a-zA-Z0-9\s]')

    spam_words = ['free', 'win', 'click', 'prize', 'buy now']
    for word in spam_words:
        df[f'has_{word.replace(" ", "_")}'] = df['content'].str.contains(word, case=False).astype(int)

    feature_cols = [
        'num_links', 'num_exclamations', 'num_uppercase_words',
        'text_length', 'num_special_chars'
    ] + [f'has_{w.replace(" ", "_")}' for w in spam_words]

    return df[feature_cols].values

def ensemble_predict(X_combined):
    nb_probs = nb_model.predict_proba(X_combined)
    lr_probs = lr_model.predict_proba(X_combined)
    lgb_probs = lgb_model.predict_proba(X_combined)
    ensemble_probs = (0.3 * nb_probs + 0.4 * lr_probs + 0.3 * lgb_probs)

    pred_label = ensemble_probs.argmax(axis=1)
    confidence = ensemble_probs.max(axis=1) * 100
    labels = le.inverse_transform(pred_label)

    return labels, confidence

# Core Prediction
def predict_single(subject: str, body: str):
    content = f"{subject} {body}".strip()
    clean_content = clean_text(content)

    X_text = tfidf.transform([clean_content])
    X_hand = compute_features([content])
    X_combined = hstack([X_text, X_hand])

    label, confidence = ensemble_predict(X_combined)
    return {"label": label[0], "confidence": round(confidence[0], 2)}

# CSV LOGGING FUNCTION (unchanged)
def log_prediction(timestamp, subject, body, predicted_label, confidence, bds_verdict, mode, encryption):
    log_path = os.path.join(OUTPUT_DIR, "predictions_log.csv")

    entry = {
        "timestamp": timestamp,
        "subject": subject,
        "body": body,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "bds_verdict": bds_verdict,
        "mode": mode,
        "encryption": encryption
    }

    df = pd.DataFrame([entry])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", index=False, header=False)
    else:
        df.to_csv(log_path, mode="w", index=False, header=True)

# FastAPI App
app = FastAPI()

class RawEmailInput(BaseModel):
    email: Optional[str] = None
    bds_verdict: Optional[int] = 0
    mode: Optional[int] = None 
    encryption: Optional[int] = 0
    error: Optional[int] = 0

@app.post("/predict")
def predict_email(input_data: RawEmailInput):
    try:
        if input_data.error == 1:
            return {"error": 1}

        if not input_data.email:
            return {"error": 1}

        # STEP 1 — Extract MIME
        extracted = extract_subject_body_and_attachments({"email": input_data.email})
        subject, body = extracted.get("subject", ""), extracted.get("body", "")

        if input_data.encryption == 1:
            # Your static 3-byte IV `{0xAA,0xBB,0xCC}` padded to 16
            iv = bytes([0xAA, 0xBB, 0xCC] + [0]*13)
            body = decrypt_raw_cbc_from_body(body, iv)

        # STEP 2 — ML prediction
        ml_result = predict_single(subject, body)

        raw_label = ml_result["label"].lower()
        ml_prediction = "Clean" if raw_label == "ham" else "Spam"
        ml_probability = f"{ml_result['confidence']}%"

        bds_label = CATEGORY_MAP.get(input_data.bds_verdict, "UnknownCategory")

        timestamp = datetime.now(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S")
        log_prediction(
            timestamp=timestamp,
            subject=subject,
            body=body,
            predicted_label=ml_prediction,
            confidence=ml_probability,
            bds_verdict=bds_label,
            mode=input_data.mode,
            encryption=input_data.encryption
        )

        return {
            "mode": input_data.mode,
            "encryption": input_data.encryption,
            "bds_verdict": bds_label,
            "ml_prediction": ml_prediction,
            "ml_probability": ml_probability,
            "decrypted_body": body
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))