# -*- coding: utf-8 -*-
"""
BOTH BIAS AND CREDIBILITY DETECTOR
"""
import streamlit as st
import joblib
import os

# ---------- PAGE CONFIG AND STYLES ----------
st.set_page_config(page_title="StayUnbiased", layout="centered")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Ensure this file exists


# ---------- LANGUAGE SUPPORT ----------
lang = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Malaysia"])
lang_code = "en" if lang == "English" else "ms"

translations = {
    "en": {
        "title": "ARE U SURE ABOUT THAT...?",
        "subtitle": "Check news for bias and credibility",
        "input_label": "Paste a news paragraph or headline:",
        "submit": "Analyze",
        "bias_result": " Detected Bias related to",
        "no_bias": "No bias detected.",
        "fake": "The news is predicted to be FAKE.",
        "real": "The news is predicted to be REAL.",
        "warning_empty": "Please enter some text."
    },
    "ms": {
        "title": "ADAKAH ANDA PASTI TENTANG ITU...?",
        "subtitle": "Semak berita untuk bias dan kebolehpercayaan",
        "input_label": "Tampal perenggan atau tajuk berita:",
        "submit": "Analisis",
        "bias_result": "Bias berkaitan dikesan:",
        "no_bias": "Tiada bias dikesan.",
        "fake": "Berita diramalkan PALSU.",
        "real": "Berita diramalkan BENAR.",
        "warning_empty": "Sila masukkan teks."
    }
}
t = translations[lang_code]


# ---------- LOAD MODEL AND VECTORIZER ----------
MODEL_PATH = "model_fake_news.pkl"
VECTORIZER_PATH = "vectorizer_fake_news.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("Model or vectorizer file not found. Please ensure the .pkl files are in the same directory.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ---------- HEADER ----------
st.markdown(f"""
<div style="background-color: #e6f2ff; padding: 2rem; border-radius: 10px; text-align: center;">
    <h1 style="color: #0077cc; margin-bottom: 0;">{t['title']}</h1>
    <p style="color: #1a1a1a; font-size: 1.2rem;">{t['subtitle']}</p>
</div>
""", unsafe_allow_html=True)


# ---------- TEXT INPUT ----------
news_input = st.text_area(t["input_label"], height=200, placeholder="E.g. Malaysia to switch to right-hand driving in 2026")


# ---------- BIAS DETECTION LOGIC ----------
bias_keywords = {
    "Race": ["malay", "chinese", "indian", "race", "Black","brown","arab","kaum", "etnik", "bangsa"],
    "Religion": ["muslim", "islam", "christian", "hindu", "buddhist", "agama", "allah", "church","atheist"],
    "Gender": ["woman", "women", "girl", "gender", "boy", "lelaki", "perempuan", "female", "male", "man","gay"],
    "Politics": ["government", "politics", "umno", "dap", "election", "kerajaan", "manifesto", "mps", "parliament"]
}

def detect_bias(text):
    text = text.lower()
    for category, keywords in bias_keywords.items():
        if any(word in text for word in keywords):
            return category
    return "None"


# ---------- ANALYSIS ----------
if st.button(t["submit"]):
    if news_input.strip() == "":
        st.warning(t["warning_empty"])
    else:
        # Bias Detection
        detected_bias = detect_bias(news_input)
        if detected_bias == "None":
            st.success(t["no_bias"])
        else:
            st.warning(f"{t['bias_result']} **{detected_bias}**")

        # Fake News Prediction
        input_vector = vectorizer.transform([news_input])
        prediction = model.predict(input_vector)[0]

        if prediction == "FAKE":
            st.error(t["fake"])
        elif prediction == "REAL":
            st.success(t["real"])
        else:
            print("Uncertain")


# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built using machine learning and keyword-based detection on Malaysian news data.")
