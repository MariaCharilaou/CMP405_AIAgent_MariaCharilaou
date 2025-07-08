import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Φόρτωση dataset
df = pd.read_csv("pricerunner_aggregate.csv")
df.columns = [
    "id", "product_title_raw", "merchant_id", "cluster_id",
    "product_title_clean", "category_id", "product_category"
]

# 2. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(df["product_title_raw"]).toarray()

# 3. Label encoding για το target
le = LabelEncoder()
y = le.fit_transform(df["product_category"])

# 4. Εκπαίδευση μοντέλου
model = RandomForestClassifier()
model.fit(X, y)

# 5. AI Agent function
def ai_agent_predict(title):
    input_vec = vectorizer.transform([title]).toarray()
    prediction = model.predict(input_vec)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    tfidf_scores = input_vec[0]
    feature_names = vectorizer.get_feature_names_out()
    keyword_scores = list(zip(feature_names, tfidf_scores))
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    top_keywords = [kw for kw, score in keyword_scores if score > 0][:5]

    response = f"📦 Προβλεπόμενη κατηγορία: **{predicted_label}**\n"
    response += f"🔍 Λέξεις-κλειδιά που επηρέασαν την πρόβλεψη: {', '.join(top_keywords)}\n"
    response += "🧠 Το μοντέλο βασίστηκε στην παρουσία των παραπάνω λέξεων για να προβλέψει την κατηγορία."
    return response

# Παράδειγμα χρήσης
if __name__ == "__main__":
    example = input("Δώσε τίτλο προϊόντος: ")
    print(ai_agent_predict(example))
