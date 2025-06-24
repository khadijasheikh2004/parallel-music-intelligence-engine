import streamlit as st
import ray
import os
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🎶 Parallel Music Intelligence Engine",
    # page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1DB954;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def predict_mood(file_path):
    """Predict mood of audio file"""
    try:
        audio, sr = librosa.load(file_path, duration=30)

        # Safe features (no beat_track, no Numba crash)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # Rule-based simple mood detection (no tempo)
        if spectral_centroid > 3000:
            mood = 'Energetic/Happy'
        elif zero_crossings > 0.1:
            mood = 'Aggressive/Noisy'
        else:
            mood = 'Neutral/Chill'

        return mood, spectral_centroid, zero_crossings

    except Exception as e:
        return f"Error: {e}", 0, 0

@st.cache_resource
def load_model_and_data():
    """Load the model and dataset (cached for performance)"""
    try:
        model = joblib.load('music_genre_classifier.pkl')
        dataset = pd.read_csv('features_gtzan_ray.csv')
        
        feature_columns = [col for col in dataset.columns if col.startswith('mfcc')]
        dataset_features = dataset[feature_columns].values
        dataset_paths = dataset['file_path'].values
        dataset_genres = dataset['genre'].values
        
        return model, dataset, dataset_features, dataset_paths, dataset_genres, feature_columns
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None, None

@ray.remote
def predict_and_recommend(file_path, model, dataset, dataset_features, dataset_paths, dataset_genres, top_n=3):
    """Ray remote function for parallel processing"""
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Predict Genre
        features_df = pd.DataFrame([mfcc_mean], columns=[f'mfcc{i}' for i in range(30)])
        predicted_genre = model.predict(features_df)[0]

        # Predict Mood
        mood, spectral_centroid, zero_crossings = predict_mood(file_path)

        # Filter dataset by predicted genre
        genre_mask = dataset['genre'] == predicted_genre
        genre_features = dataset_features[genre_mask]
        genre_paths = dataset_paths[genre_mask]

        # Compute similarity
        similarity = cosine_similarity([mfcc_mean], genre_features)
        top_indices = similarity[0].argsort()[-top_n:][::-1]

        recommendations = genre_paths[top_indices]
        similarity_scores = similarity[0][top_indices]

        return {
            'file': file_path,
            'predicted_genre': predicted_genre,
            'predicted_mood': mood,
            'spectral_centroid': spectral_centroid,
            'zero_crossings': zero_crossings,
            'recommendations': recommendations.tolist(),
            'similarity_scores': similarity_scores.tolist()
        }

    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def create_audio_features_chart(results):
    """Create a chart showing audio features"""
    if not results or any('error' in result for result in results):
        return None
    
    # Extract features for visualization
    files = [os.path.basename(result['file']) for result in results if 'error' not in result]
    spectral_centroids = [result['spectral_centroid'] for result in results if 'error' not in result]
    zero_crossings = [result['zero_crossings'] for result in results if 'error' not in result]
    
    # Create subplot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Spectral Centroid',
        x=files,
        y=spectral_centroids,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Audio Features Analysis',
        xaxis_title='Audio Files',
        yaxis_title='Spectral Centroid (Hz)',
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎶 Parallel Music Intelligence Engine</h1>', unsafe_allow_html=True)
    
    # Initialize Ray
    if 'ray_initialized' not in st.session_state:
        ray.init(ignore_reinit_error=True)
        st.session_state.ray_initialized = True
    
    # Load model and data
    model, dataset, dataset_features, dataset_paths, dataset_genres, feature_columns = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model and data. Please check if the required files exist.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # File selection method
    selection_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Files", "Use Sample Folder"]
    )
    
    
    files_to_process = []
    
    if selection_method == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload audio files:",
            type=['mp3', 'wav'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Save uploaded files temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_to_process.append(temp_path)
    
    else:  # Use Sample Folder
        sample_folder = st.sidebar.text_input(
            "Sample folder path:",
            value="data/sample_songs"
        )
        
        if os.path.exists(sample_folder):
            files_to_process = [
                os.path.join(sample_folder, f) 
                for f in os.listdir(sample_folder) 
                if f.endswith(('.mp3', '.wav'))
            ]
            st.sidebar.success(f"Found {len(files_to_process)} files in folder")
        else:
            st.sidebar.warning("Sample folder not found")
    
    # Main content
    if files_to_process:
        if st.button("🚀 Analyze Music Files", type="primary"):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Processing {len(files_to_process)} files...")
            
            start_time = time.time()
            
            # Parallel processing with Ray
            futures = [
                predict_and_recommend.remote(
                    file, model, dataset, dataset_features, 
                    dataset_paths, dataset_genres
                ) 
                for file in files_to_process
            ]
            
            results = ray.get(futures)
            end_time = time.time()
            
            progress_bar.progress(100)
            status_text.text(f"✅ Processing completed in {end_time - start_time:.2f} seconds")
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            successful_results = [r for r in results if 'error' not in r]
            error_results = [r for r in results if 'error' in r]
            
            col1.metric("Total Files", len(files_to_process))
            col2.metric("Successful", len(successful_results))
            col3.metric("Errors", len(error_results))
            col4.metric("Processing Time", f"{end_time - start_time:.2f}s")
            
            # Audio features chart
            if successful_results:
                st.subheader("📈 Audio Features Analysis")
                chart = create_audio_features_chart(successful_results)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Detailed results
            st.subheader("🎯 Detailed Predictions")
            
            for i, result in enumerate(results):
                with st.expander(f"🎵 {os.path.basename(result['file'])}", expanded=True):
                    if 'error' in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>🎤 Predicted Genre</h4>
                                <h3 style="color: #1DB954;">{result['predicted_genre']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>Predicted Mood</h4>
                                <h3 style="color: #FF6B6B;">{result['predicted_mood']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**🎛️ Audio Features:**")
                            st.write(f"Spectral Centroid: {result['spectral_centroid']:.2f} Hz")
                            st.write(f"Zero Crossings: {result['zero_crossings']:.4f}")
                        
                        # Recommendations
                        st.markdown("**🎶 Top Recommendations:**")
                        for j, (rec, score) in enumerate(zip(result['recommendations'], result['similarity_scores'])):
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <strong>#{j+1}</strong> {os.path.basename(rec)}<br>
                                <small>Similarity Score: {score:.4f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            # 🎧 Play the recommended audio file
                            if os.path.exists(rec):
                                audio_bytes = open(rec, 'rb').read()
                                st.audio(audio_bytes, format='audio/mp3')
                            else:
                                st.warning(f"File {rec} not found!")
            
            # Genre distribution
            if successful_results:
                st.subheader("🎨 Genre Distribution")
                genres = [result['predicted_genre'] for result in successful_results]
                genre_counts = pd.Series(genres).value_counts()
                
                fig_pie = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title="Predicted Genres Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Mood distribution
                st.subheader("😊 Mood Distribution")
                moods = [result['predicted_mood'] for result in successful_results]
                mood_counts = pd.Series(moods).value_counts()
                
                fig_mood = px.bar(
                    x=mood_counts.index,
                    y=mood_counts.values,
                    title="Predicted Moods Distribution",
                    labels={'x': 'Mood', 'y': 'Count'}
                )
                st.plotly_chart(fig_mood, use_container_width=True)
    
    else:
        st.info("👆 Please upload audio files or specify a sample folder to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, Ray, and Librosa by Tehreem, Khadija, Huzaira 😎")

if __name__ == "__main__":
    main()