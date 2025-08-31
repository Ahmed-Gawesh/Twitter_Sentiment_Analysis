# Twitter Sentiment Analysis

📖 Project Overview
This project develops a deep learning-based sentiment analysis model to classify Twitter (X) texts into three categories: Positive, Negative, or Neutral. Built using TensorFlow/Keras and deployed with a Gradio web interface, it provides an end-to-end pipeline for analyzing tweet sentiments. The model is trained on a public Kaggle dataset and is designed for scalability and ease of use.
The project aims to enable users to understand public sentiment on social media, useful for brand monitoring, opinion analysis, or market research.

✨ Features

Robust Preprocessing: Cleans tweets by removing HTML tags, mentions, URLs, special characters, and extra spaces.
Deep Learning Model: Utilizes an LSTM-based architecture for accurate sentiment classification.
Interactive Web App: A user-friendly Gradio interface for real-time sentiment predictions.
Deployment-Ready: Easily run locally or deploy on platforms like Hugging Face Spaces.
Extensible Pipeline: Modular code for preprocessing, training, and inference.


📊 Dataset
The model is trained on the Twitter Entity Sentiment Analysis dataset from Kaggle, containing labeled tweets.

Training Data: ~74,000 samples (after removing 'Irrelevant' labels).
Validation Data: ~1,000 samples.
Label Distribution (training set):

Negative: 22,358
Positive: 20,655
Neutral: 18,108



The 'Irrelevant' label was excluded to focus on core sentiment classes.

🏗️ Project Structure
texttwitter-sentiment-analysis/
├── twitter-sentiment-analysis.ipynb   # Data loading, preprocessing, and model training
├── Deplyment_gradio.ipynb            # Model deployment with Gradio
├── best_model.keras                  # Trained Keras model (not in repo; generated during training)
├── tokenizer.pkl                     # Tokenizer for text processing (generated during training)
├── README.md                         # Project documentation
└── LICENSE                           # MIT License

🔧 Design and Architecture
1. Data Preprocessing

Text Cleaning:

Converts text to lowercase.
Removes HTML tags, mentions (@user), URLs, special characters, digits, and extra spaces using regex.


Tokenization: Uses Keras Tokenizer to convert text to sequences (vocabulary size: 5,000 words).
Padding: Sequences are padded to a fixed length of 25 tokens for consistent model input.
Label Encoding: Sentiments are one-hot encoded (Negative: 0, Positive: 1, Neutral: 2).

2. Model Architecture

Embedding Layer: Maps tokens to 128-dimensional dense vectors.
Bidirectional LSTM: 64 units to capture contextual dependencies in text sequences.
Global Max Pooling: Reduces dimensionality while preserving key features.
Dense Layers: Fully connected layers with 50% dropout for regularization, followed by a softmax output for 3-class classification.
Training Configuration:

Optimizer: Adam
Loss: Categorical cross-entropy
Batch Size: 32
Epochs: 10 (with early stopping)



The architecture balances performance and efficiency, leveraging LSTM's strength in sequential data processing for short texts like tweets.
3. Deployment

Gradio Interface: A clean UI with a text input for tweets and an output displaying the predicted sentiment (Negative, Positive, Neutral).
Prediction Workflow:

Input tweet text.
Clean and preprocess (tokenize, pad).
Run through the trained model.
Display sentiment label.



Example:

Input: "This product is amazing!"
Output: Positive


🚀 Installation

Install Dependencies:
bashpip install tensorflow gradio nltk pandas numpy matplotlib scikit-learn


Download Dataset:

Obtain the Twitter Entity Sentiment Analysis dataset from Kaggle.
Place twitter_training.csv and twitter_validation.csv in the project directory.




🛠️ Usage
1. Training the Model

Open twitter-sentiment-analysis.ipynb in Jupyter Notebook.
Execute cells to:

Load and preprocess the dataset.
Train the LSTM model.
Save the trained model (best_model.keras) and tokenizer (tokenizer.pkl).



2. Running the Gradio App

Open Deplyment_gradio.ipynb.
Execute cells to load the saved model and tokenizer.
Launch the Gradio interface:
pythondemo.launch()

Access the app at http://127.0.0.1:7865 or use share=True for a public link.

3. Making Predictions

Enter a tweet in the Gradio text box.
Receive the predicted sentiment (e.g., "Positive", "Negative", "Neutral").


📋 Dependencies

Python 3.10+
TensorFlow/Keras 2.10+
Gradio 3.0+
NLTK
Pandas
NumPy
Matplotlib
Scikit-learn


⚠️ Limitations

Trained on English tweets only; may not generalize to other languages.
Optimized for short texts; performance may degrade on longer or noisy inputs.
Does not account for sarcasm, irony, or complex emotions.
