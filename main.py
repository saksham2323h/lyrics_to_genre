
# Importing relevant libraries
import pandas as pd
import re
from langdetect import detect, DetectorFactory
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

DetectorFactory.seed = 42
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Lowercasing, Remove punctuation characters and retain alpha-numeric part
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Multi-lingual Stemmer
SUPPORTED_SNOWBALL = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian"
}
from langdetect import detect
from nltk.stem import SnowballStemmer

def multilingual_stem(text):
    if not text or len(text) < 5:
        return text

    try:
        lang = detect(text[:300])
    except:
        return text

    if lang in SUPPORTED_SNOWBALL:
        stemmer = SnowballStemmer(SUPPORTED_SNOWBALL[lang])
        return " ".join(stemmer.stem(w) for w in text.split())

    return text

# Reading train file
df = pd.read_csv("train.csv")
X = df["Lyrics"].apply(clean_text)
X = X.apply(multilingual_stem)
Y = df["Genre"]

# Generating TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

# Encoding label & dividing input data to obtain train & test
X = vectorizer.fit_transform(X)
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=Y_encoded
)

# Using Support Vector Machine to train the model
model = LinearSVC(
    C=0.1,
    class_weight="balanced",
    max_iter=5000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(model, "genre_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
