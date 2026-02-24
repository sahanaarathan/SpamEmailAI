import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# -----------------------------
# 2️⃣ Improved Clean Text Function
# -----------------------------
def clean_text(text):
    text = text.lower()

    # Expand common short forms
    replacements = {
        "u": "you",
        "ur": "your",
        "hv": "have",
        "won": "win",
        "congrats": "congratulations"
    }

    words = text.split()
    words = [replacements[word] if word in replacements else word for word in words]

    text = " ".join(words)
    text = ''.join([char for char in text if char not in string.punctuation])

    return text


df['message'] = df['message'].apply(clean_text)


# -----------------------------
# 3️⃣ Convert Text → Numbers
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']


# -----------------------------
# 4️⃣ Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X, y)


# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam Email Checker AI", page_icon="📩")

st.title("📩 Spam Email Checker AI")
st.write("Enter a message below and check if it is Spam or Not Spam.")

user_input = st.text_area("Enter your message here:")

if st.button("Check"):

    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Spam Detected! (Spam Probability: {probability:.2f})")
        else:
            st.success(f"✅ Not Spam (Spam Probability: {probability:.2f})")