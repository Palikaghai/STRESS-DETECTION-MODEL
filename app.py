import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from collections import Counter

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("stress_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# ---------------- TEXT CLEANING (SAME AS TRAIN) ----------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- SIMPLE EMOTION TAGGER ----------------
def get_emotions(text: str):
    text = text.lower()
    emotions = {
        "anxiety": ["anxious", "nervous", "worried", "panic"],
        "sadness": ["sad", "depressed", "cry", "hopeless"],
        "anger": ["angry", "furious", "rage", "irritated"],
        "fear": ["scared", "afraid", "terrified"],
        "joy": ["happy", "excited", "relaxed", "calm"],
    }

    scores = []
    for emotion, keywords in emotions.items():
        score = sum(word in text for word in keywords)
        if score > 0:
            scores.append({"label": emotion, "score": min(score / 5, 1.0)})

    if not scores:
        return [{"label": "neutral", "score": 1.0}]
    return scores


# ---------------- WORD STATS: TOP 5 UNIQUE WORDS ----------------
def get_top_words(text: str, top_k: int = 5):
    cleaned = clean_text(text)
    words = cleaned.split()
    if not words:
        return pd.DataFrame(columns=["word", "count"])
    freq = Counter(words)
    common = freq.most_common(top_k)
    df = pd.DataFrame(common, columns=["word", "count"])
    return df


# ---------------- STREAMLIT UI ----------------
st.title("STRESS DETECTION")
st.write("Enter any text; the system predicts stress level, emotions, and top words.")

user_text = st.text_area("Your text", height=150)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])

        # ---- Model probability (for info only) ----
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])  # probability of stressed
        else:
            pred = int(model.predict(X)[0])
            prob = float(pred)

        # ---- Emotion-based final decision ----
        emotions_list = get_emotions(user_text)
        negative_emotions = {"fear", "anger", "sadness"}

        is_stressed = any(e["label"] in negative_emotions for e in emotions_list)
        label = "Stressed" if is_stressed else "Not Stressed"

        st.subheader(f"Prediction: {label}")
        st.write(f"(Model stress probability: **{prob:.2f}")

        # ---- Traffic-light stress scale (uses is_stressed + prob) ----
        if not is_stressed:
            color = "#00C853"  # green
            level = "Low / No Stress"
        else:
            # refine by prob if you want
            if prob < 0.33:
                color = "#FFD600"  # yellow
                level = "Moderate Stress (negative emotions detected)"
            else:
                color = "#D50000"  # red
                level = "High Stress (negative emotions detected)"

        st.markdown(
            f"""
            <div style="border-radius:8px;padding:12px;background-color:{color};color:black;
                        font-weight:bold;text-align:center;">
                {level}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Emotions ----
        st.subheader("Detected emotions")
        for emo in emotions_list:
            st.write(f"- {emo['label']}: {emo['score']:.2f}")

        # ---- Top 5 unique words ----
        st.subheader("Top 5 repeated unique words")
        top_words_df = get_top_words(user_text, top_k=5)

        if not top_words_df.empty:
            st.table(top_words_df)
        else:
            st.info("No valid words found after cleaning the text.")

