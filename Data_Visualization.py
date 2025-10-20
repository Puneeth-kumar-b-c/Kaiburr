# -------------------- IMPORT LIBRARIES --------------------
import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

# -------------------- STEP 1: LOAD DATASET IN CHUNKS (HANDLE LARGE FILE) --------------------
nltk.download('stopwords')
nltk.download('punkt')

dataset_path = "D:/Kaiburr_Model/complaints.csv"   # ✅ Your dataset path

# Target sample size to avoid OOM - adjust as needed (e.g., 500k rows for ~2-4GB RAM usage)
TARGET_SAMPLE_SIZE = 500000

print(f"🔄 Loading large dataset in chunks and sampling to {TARGET_SAMPLE_SIZE:,} rows...")

try:
    # Load only relevant columns to save memory
    usecols = ['Consumer complaint narrative', 'Product']
    
    # Initialize lists to collect filtered data
    valid_rows = []
    total_rows_processed = 0
    
    chunk_size = 10000  # Process 10k rows at a time
    for chunk in pd.read_csv(dataset_path, usecols=usecols, chunksize=chunk_size, low_memory=False, encoding='utf-8'):
        # Filter non-null narratives
        chunk = chunk[chunk['Consumer complaint narrative'].notnull()]
        
        # Filter to target categories
        target_categories = {
            'Credit reporting, repair, or other': 0,
            'Debt collection': 1,
            'Consumer Loan': 2,
            'Mortgage': 3
        }
        chunk = chunk[chunk['Product'].isin(target_categories.keys())]
        chunk['Category'] = chunk['Product'].map(target_categories)
        chunk = chunk.dropna(subset=['Category'])  # Drop if category mapping fails
        
        if not chunk.empty:
            valid_rows.append(chunk)
            total_rows_processed += len(chunk)
            
            # Sample proportionally if we've exceeded target (to ensure balanced sampling)
            if total_rows_processed > TARGET_SAMPLE_SIZE:
                # Simple random sample from accumulated (but since chunked, sample per chunk)
                sample_size_per_chunk = max(1, int(len(chunk) * (TARGET_SAMPLE_SIZE / total_rows_processed)))
                chunk = chunk.sample(n=sample_size_per_chunk, random_state=42)
                valid_rows[-1] = chunk  # Replace last chunk with sample
                break  # Stop after reaching target
        
        print(f"Processed {total_rows_processed:,} valid rows so far...")
    
    if not valid_rows:
        raise ValueError("No valid data found after filtering!")
    
    # Concatenate all valid chunks
    df = pd.concat(valid_rows, ignore_index=True)
    print(f"✅ Dataset Loaded Successfully! Final shape: {df.shape}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

print(df.head())
print(df.columns)

# -------------------- STEP 2: TEXT PREPROCESSING --------------------
def preprocess_text(text):
    text = re.sub(r'\S+@\S+', ' ', text)                  # Remove emails
    text = re.sub(r'\b\d{10,}\b', ' ', text)              # Remove phone numbers
    text = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b','XXXX XXXX XXXX XXXX', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b','XXX-XX-XXXX', text)
    text = text.lower()                                   # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    return text

print("🔄 Preprocessing text...")
df['Clean_Complaint'] = df['Consumer complaint narrative'].apply(preprocess_text)

# -------------------- STEP 3: TRAIN-TEST SPLIT --------------------
X = df['Clean_Complaint']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- STEP 4: FEATURE EXTRACTION (TF-IDF) --------------------
print("🔄 Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✅ TF-IDF Train shape: {X_train_tfidf.shape}, Test shape: {X_test_tfidf.shape}")

# -------------------- STEP 5: MULTI-MODEL TRAINING --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "SVM": SVC(kernel='linear'),
    "Naive Bayes": MultinomialNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy of {name}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    results[name] = accuracy

# -------------------- STEP 6: MODEL COMPARISON --------------------
print("\n📊 Model Accuracy Comparison:")
for model_name, acc in results.items():
    print(f"{model_name} → {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model}")

# -------------------- STEP 7: CONFUSION MATRIX (BEST MODEL) --------------------
best_clf = models[best_model]
y_pred_best = best_clf.predict(X_test_tfidf)

conf_mat = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(target_categories.keys()),
            yticklabels=list(target_categories.keys()))
plt.title("Confusion Matrix - " + best_model)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------- STEP 8: SAMPLE PREDICTION --------------------
sample = ["My credit score dropped due to an error on my report!"]
sample_clean = [preprocess_text(sample[0])]
sample_tfidf = vectorizer.transform(sample_clean)
prediction = best_clf.predict(sample_tfidf)
print(f"\n🔮 Sample Prediction: {list(target_categories.keys())[list(target_categories.values()).index(prediction[0])]}")