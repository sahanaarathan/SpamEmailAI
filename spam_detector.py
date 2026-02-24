import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1️⃣ Load Real Dataset
# -----------------------------

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


# -----------------------------
# 2️⃣ Convert Labels
# -----------------------------

df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# -----------------------------
# 3️⃣ Clean Text
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['message'] = df['message'].apply(clean_text)


# -----------------------------
# 4️⃣ Convert Text → Numbers
# -----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']


# -----------------------------
# 5️⃣ Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 6️⃣ Train Model
# -----------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)


# -----------------------------
# 7️⃣ Evaluate Model
# -----------------------------

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# 8️⃣ Interactive Prediction
# -----------------------------

def predict_message(message):
    message = clean_text(message)
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"


print("\n--- Spam Email Checker ---")

while True:
    user_input = input("Enter a message (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    result = predict_message(user_input)
    print("Prediction:", result)
    print()