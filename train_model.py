import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time



# Load the extracted features from GTZAN
df = pd.read_csv('features_gtzan_ray.csv')

print("✅ Dataset loaded. Shape:", df.shape)
print("🎵 Genres distribution:\n", df['genre'].value_counts())

# Prepare features and labels
X = df.drop(['file_path', 'genre'], axis=1)
y = df['genre']


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data split complete.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

print(f"Using {joblib.cpu_count()} CPU cores for RandomForest training.")
start_time = time.time()

# Train a Random Forest Classifier

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

clf.fit(X_train, y_train)
print("✅ Model training complete.")

# Evaluate the model
y_pred = clf.predict(X_test)
print("\n🎯 Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(clf, 'music_genre_classifier.pkl')
print("✅ Model saved as 'music_genre_classifier.pkl'.")
end_time = time.time()
print(f"⏱️ Model training time: {end_time - start_time:.2f} seconds")

