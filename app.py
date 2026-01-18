import streamlit as st
import joblib
from langdetect import detect
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Lyrics Genre Predictor",
    page_icon="🎤",
    layout="centered"
)

# -------------------- ADVANCED UI & BUBBLE ANIMATION --------------------
st.markdown("""
    <style>
    /* 1. Vibrant Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Floating Bubbles CSS */
    .bubbles {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    .bubbles li {
        position: absolute;
        list-style: none;
        display: block;
        width: 20px; height: 20px;
        background: rgba(255, 255, 255, 0.2);
        bottom: -150px;
        animation: animate 25s linear infinite;
        border-radius: 50%;
    }
    @keyframes animate {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 0; }
        100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }
    
    /* Bubble positioning */
    .bubbles li:nth-child(1){ left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .bubbles li:nth-child(2){ left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .bubbles li:nth-child(3){ left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .bubbles li:nth-child(4){ left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }

    /* 3. Button Styling */
    div.stButton > button {
        background: white !important;
        color: #e73c7e !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    </style>
    
    <ul class="bubbles">
        <li></li><li></li><li></li><li></li><li></li><li></li>
    </ul>
    """, unsafe_allow_html=True)

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("genre_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        return model, vectorizer, label_encoder
    except:
        return None, None, None

model, vectorizer, label_encoder = load_assets()

# -------------------- UI CONTENT --------------------
st.title("🌈 Lyrics Genre & Language")
st.markdown("#### Discover the soul of your song ✨")

with st.container():
    lyrics = st.text_area(
        "",
        height=200,
        placeholder="Paste your lyrics here...",
        label_visibility="collapsed"
    )

if st.button("🚀 ANALYZE VIBE", use_container_width=True):
    if not lyrics.strip():
        st.warning("The mic is quiet... please add lyrics!")
    else:
        # Progress bar for effect
        progress_text = "Tuning into the rhythm..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()

        # Language Logic
        try:
            lang_code = detect(lyrics)
            lang_map = {'en': 'English 🇺🇸', 'es': 'Spanish 🇪🇸', 'fr': 'French 🇫🇷', 'hi': 'Hindi 🇮🇳'}
            detected_lang = lang_map.get(lang_code, lang_code.upper())
            st.info(f"🌍 Detected Language: **{detected_lang}**")
        except:
            st.error("Language scan failed.")

        # Genre Logic
        X = vectorizer.transform([lyrics])
        pred_encoded = model.predict(X)[0]
        pred_genre = label_encoder.inverse_transform([pred_encoded])[0]

        # --- THE CELEBRATION ---
        st.balloons()
        st.snow() # Snow looks like falling confetti on a colored background
        
        st.success(f"🎧 Predicted Genre: **{pred_genre}**")