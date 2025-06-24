import ray
import os
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

def predict_mood(file_path):
    try:
        audio, sr = librosa.load(file_path)

        # Ray-safe features onlyv
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # Rule-based mood detection without 'tempo'
        if spectral_centroid > 3000:
            mood = 'Energetic/Happy'
        elif zero_crossings > 0.1:
            mood = 'Aggressive/Noisy'
        else:
            mood = 'Neutral/Chill'

        return mood

    except Exception as e:
        return f"Error: {e}"

# Load model and dataset features
model = joblib.load('music_genre_classifier.pkl')
dataset = pd.read_csv('features_gtzan_ray.csv')

feature_columns = [col for col in dataset.columns if col.startswith('mfcc')]
dataset_features = dataset[feature_columns].values
dataset_paths = dataset['file_path'].values
dataset_genres = dataset['genre'].values

@ray.remote
def predict_and_recommend(file_path, top_n=3):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Predict Genre
        features_df = pd.DataFrame([mfcc_mean], columns=[f'mfcc{i}' for i in range(30)])
        predicted_genre = model.predict(features_df)[0]

        # Predict Mood (Ray-safe)
        mood = predict_mood(file_path)

        # Filter dataset by predicted genre
        genre_mask = dataset['genre'] == predicted_genre
        genre_features = dataset_features[genre_mask]
        genre_paths = dataset_paths[genre_mask]

        # Compute similarity
        similarity = cosine_similarity([mfcc_mean], genre_features)
        top_indices = similarity[0].argsort()[-top_n:][::-1]

        recommendations = genre_paths[top_indices]

        return {
            'file': file_path,
            'predicted_genre': predicted_genre,
            'predicted_mood': mood,
            'recommendations': recommendations.tolist()
        }

    except Exception as e:
        return {'file': file_path, 'error': str(e)}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    TEST_FOLDER = 'sample_songs'
    test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.endswith(('.mp3', '.wav'))]

    print(f"🔍 Found {len(test_files)} files to process.")

    start_time = time.time()

    futures = [predict_and_recommend.remote(file) for file in test_files]
    results = ray.get(futures)

    for result in results:
        print(f"\n🎵 Input File: {result['file']}")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🎤 Predicted Genre: {result['predicted_genre']}")
            print(f"😊 Predicted Mood: {result['predicted_mood']}")
            print("🎶 Top 3 Recommendations:")
            for rec in result['recommendations']:
                print(f"   - {rec}")

    end_time = time.time()
    print(f"\n⏱️ Total parallel prediction + recommendation time: {end_time - start_time:.2f} seconds")
    if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    TEST_FOLDER = 'sample_songs'
    test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.endswith(('.mp3', '.wav'))]

    print(f"🔍 Found {len(test_files)} files to process.")

    # ------------ PARALLEL (RAY) ------------
    start_parallel = time.time()

    futures = [predict_and_recommend.remote(file) for file in test_files]
    results_parallel = ray.get(futures)

    end_parallel = time.time()

    # ------------ SEQUENTIAL ------------
    def predict_and_recommend_seq(file_path):
        # Same as @ray.remote function, but without Ray
        try:
            audio, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            features_df = pd.DataFrame([mfcc_mean], columns=[f'mfcc{i}' for i in range(30)])
            predicted_genre = model.predict(features_df)[0]
            mood = predict_mood(file_path)

            genre_mask = dataset['genre'] == predicted_genre
            genre_features = dataset_features[genre_mask]
            genre_paths = dataset_paths[genre_mask]

            similarity = cosine_similarity([mfcc_mean], genre_features)
            top_indices = similarity[0].argsort()[-3:][::-1]

            recommendations = genre_paths[top_indices]

            return {
                'file': file_path,
                'predicted_genre': predicted_genre,
                'predicted_mood': mood,
                'recommendations': recommendations.tolist()
            }

        except Exception as e:
            return {'file': file_path, 'error': str(e)}

    start_seq = time.time()

    results_seq = [predict_and_recommend_seq(file) for file in test_files]

    end_seq = time.time()

    if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    TEST_FOLDER = 'sample_songs'
    test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.endswith(('.mp3', '.wav'))]

    print(f"🔍 Found {len(test_files)} files to process.")

    # ------------ PARALLEL (RAY) ------------
    start_parallel = time.time()

    futures = [predict_and_recommend.remote(file) for file in test_files]
    results_parallel = ray.get(futures)

    end_parallel = time.time()

    # ------------ SEQUENTIAL ------------
    def predict_and_recommend_seq(file_path):
        # Same as @ray.remote function, but without Ray
        try:
            audio, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            features_df = pd.DataFrame([mfcc_mean], columns=[f'mfcc{i}' for i in range(30)])
            predicted_genre = model.predict(features_df)[0]
            mood = predict_mood(file_path)

            genre_mask = dataset['genre'] == predicted_genre
            genre_features = dataset_features[genre_mask]
            genre_paths = dataset_paths[genre_mask]

            similarity = cosine_similarity([mfcc_mean], genre_features)
            top_indices = similarity[0].argsort()[-3:][::-1]

            recommendations = genre_paths[top_indices]

            return {
                'file': file_path,
                'predicted_genre': predicted_genre,
                'predicted_mood': mood,
                'recommendations': recommendations.tolist()
            }

        except Exception as e:
            return {'file': file_path, 'error': str(e)}

    start_seq = time.time()

    results_seq = [predict_and_recommend_seq(file) for file in test_files]

    end_seq = time.time()

    # ------------ Print Timing ------------
    parallel_time = end_parallel - start_parallel
    sequential_time = end_seq - start_seq

    print(f"\n⏱️ Parallel (Ray) Time: {parallel_time:.2f} seconds")
    print(f"⏱️ Sequential Time: {sequential_time:.2f} seconds")

    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"🚀 Speedup: {speedup:.2f}x faster with Ray")
    else:
        print("⚠️ Sequential time is too small to compute speedup.")
