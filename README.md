# Parallel Music Intelligence Engine

A fast, parallelized system for music genre and mood prediction, plus song recommendations, built with machine learning and audio processing.

## Project Goal
This project aims to solve the problem of **slow music analysis and recommendation** by leveraging **parallel computing** to speed up audio feature processing and genre prediction.

## Features
- Genre classification of audio files using a trained Random Forest model.
- Mood detection based on extracted features.
- Fast, parallel processing of input files using **Ray**.
- Song recommendations based on **cosine similarity** of extracted features.
- Interactive user interface powered by **Streamlit**.

## Tech Stack
- **Language:** Python  
- **Libraries/Tools:**  
  - [Ray](https://docs.ray.io/en/latest/) – for parallel execution  
  - [Librosa](https://librosa.org/) – for audio feature extraction (MFCCs, spectral centroid, etc.)  
  - [Scikit-learn](https://scikit-learn.org/) – for machine learning  
  - [Joblib](https://joblib.readthedocs.io/) – for saving/loading models  
  - [Streamlit](https://streamlit.io/) – for GUI

## Parallel Computing Design
- **Decomposition Type:** Task Decomposition  
- **Synchronization:** Managed by Ray’s task scheduler  
- **Communication Pattern:** Implicit through Ray’s object store  
- **Load Balancing:** Dynamic, handled automatically by Ray

Training data is included along with 5 testing sample files 
