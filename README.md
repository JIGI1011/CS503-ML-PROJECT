# Music Genre Classification using Machine Learning and Deep Learning

## Overview

This project explores multiple approaches for automatic music genre classification using the GTZAN dataset. The work compares traditional machine learning models, dimensionality reduction techniques, deep learning architectures, and pretrained audio representation models to understand the effectiveness of different feature representations for music understanding.

The project was completed as part of the CS503 course.

### Team Members

* Hardik Rangi
* Jagrati Gupta
* Nishit Soni

---

## Objectives

* Analyze and visualize audio features extracted from music tracks.
* Compare handcrafted feature-based machine learning models.
* Investigate dimensionality reduction using PCA.
* Train deep learning models on spectrogram representations.
* Evaluate pretrained audio foundation models such as Wav2Vec2 and MERT.
* Build a music recommendation system using learned audio embeddings.

---

## Dataset

The project uses the **GTZAN Music Genre Dataset**, containing 1,000 audio tracks across 10 genres:

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

The dataset provides:

* 30-second audio clips
* Spectrogram representations
* Handcrafted audio features
* Segment-level (3-second) feature data

---

## Project Pipeline

### 1. Handcrafted Feature-Based Classification

Feature datasets containing spectral, temporal, and statistical audio descriptors were analyzed.

#### Models Evaluated

* Logistic Regression
* Random Forest
* Weighted Hybrid Model (Random Forest + Logistic Regression)

#### Experiments

* Classification using 30-second song-level features
* Classification using 3-second segment-level features
* Aggregated segment predictions for song-level classification

#### Evaluation Metrics

* Accuracy
* Macro F1 Score

---

### 2. Dimensionality Reduction with PCA

Principal Component Analysis (PCA) was applied to reduce feature dimensionality before classification.

Objectives:

* Reduce redundancy among correlated features
* Improve model efficiency
* Visualize genre separability in lower-dimensional space

The impact of PCA on classification performance was evaluated using the best-performing machine learning model.

---

### 3. Spectrogram-Based Deep Learning

Audio signals were converted into spectrogram representations and used as inputs for deep neural networks.

#### Models Explored

##### CNN

A 2D Convolutional Neural Network trained directly on spectrogram images.

##### Transfer Learning with ResNet18

A pretrained ResNet18 model was fine-tuned for music genre classification.

##### CNN + LSTM Ensemble

A hybrid architecture combining:

* CNN for spatial feature extraction
* LSTM for temporal dependency modeling

---

### 4. Pretrained Audio Representations

Modern audio foundation models were leveraged to extract high-level audio embeddings.

#### Wav2Vec2

* Pretrained self-supervised speech representation model
* Feature extraction followed by genre classification

#### MERT (Music Understanding Representation Transformer)

* Music-specific pretrained transformer model
* Embedding extraction and downstream classification

---

### 5. Music Recommendation System

A content-based music recommender was developed using embeddings extracted from the MERT model.

#### Methodology

* Generate song embeddings
* Normalize embedding vectors
* Compute cosine similarity
* Retrieve most similar tracks

This demonstrates how learned audio representations can be applied beyond classification tasks.

---

## Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* PCA
* Logistic Regression
* Random Forest

### Deep Learning

* PyTorch
* Torchvision

### Audio Processing

* Librosa
* Torchaudio

### Foundation Models

* Wav2Vec2
* MERT

### Visualization

* Matplotlib
* Seaborn

### Data Processing

* NumPy
* Pandas

---

## Results

The project compares:

* Traditional machine learning models
* PCA-enhanced classifiers
* CNN architectures
* Transfer learning approaches
* Audio foundation models

Performance is evaluated using Accuracy and Macro F1 Score to provide a comprehensive comparison of feature representations and model architectures for music genre classification.

---

## Key Learnings

* Handcrafted audio features remain effective for genre classification.
* PCA helps reduce feature redundancy while preserving important information.
* Spectrogram-based deep learning models capture richer musical patterns.
* Transfer learning significantly improves performance with limited data.
* Foundation models such as Wav2Vec2 and MERT provide powerful audio embeddings that can be used for both classification and recommendation tasks.

---

## Future Work

* Experiment with larger music datasets.
* Fine-tune foundation models end-to-end.
* Explore transformer-based architectures for genre classification.
* Develop a real-time music recommendation system.
* Deploy the model as a web application or API.

---

## Repository Structure

```text
├── ML_PROJECT_CODE.ipynb
├── features_30_sec.csv
├── features_3_sec.csv
├── spectrograms/
├── final_results.csv
└── README.md
```

---

## References

* GTZAN Music Genre Dataset
* Wav2Vec2
* MERT (Music Understanding Representation Transformer)
* Scikit-learn Documentation
* PyTorch Documentation
