import pandas as pd
import numpy as np
import logging
import configparser
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration settings from an INI file
config = configparser.ConfigParser()
config.read('servier/config.ini')  
DATASET_SINGLE_PATH = config['DEFAULT']['DATASET_SINGLE_PATH']
MODEL_2_PATH = config['DEFAULT']['MODEL_2_PATH']

# Function to load dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        X = data['smiles']
        Y = data['P1']
        logging.info(f"Dataset loaded from {file_path}")
        return X, Y
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise
    
# Load the dataset from a CSV file
X, Y = load_dataset(DATASET_SINGLE_PATH)

# Function to tokenize and pad sequences
def tokenize_and_pad(X):
    """
    Tokenize and pad the SMILES sequences.

    Parameters:
        X: Pandas Series containing SMILES strings.

    Returns:
        tokenizer: Fitted tokenizer.
        padded_sequences: Numpy array of padded sequences.
    """
    
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded_sequences = pad_sequences(sequences, padding='post')
    return tokenizer, padded_sequences


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Function to create the model
def create_model_2(vocab_size, max_length):
    """
    Create a Sequential model with Embedding and LSTM layers.

    Parameters:
        vocab_size: The total number of unique tokens.
        max_length: The maximum length of the sequences.

    Returns:
        A compiled Keras model.
    """
    
    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_length),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model_2(epochs=100, batch_size=64):
    """
    Train the model on the training dataset and save it to a file.

    Parameters:
        epochs (int): The number of epochs for training the model.
        batch_size (int): The batch size for training the model.
    """
    
    # Tokenize and pad the training set
    tokenizer, padded_sequences = tokenize_and_pad(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = padded_sequences.shape[1]
    model = create_model_2(vocab_size, max_length)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save(MODEL_2_PATH)
    logging.info("Model 2 saved.")
    

# Function to evaluate the model
def evaluate_model_2(epochs=100, batch_size=64, n_splits=5):
    """
    Evaluate the model performance using cross-validation and compute additional metrics like precision, recall, and F1-score.

    Parameters:
        epochs (int): The number of epochs for training the model.
        batch_size (int): The batch size for training the model.
        n_splits (int): The number of folds in StratifiedKFold.

    Returns:
        dict: A dictionary containing cross-validation results, mean accuracy, 
              standard deviation, precision, recall, and F1-score.
    """
    
    # Tokenize and pad the SMILES sequences
    tokenizer, padded_sequences = tokenize_and_pad(X)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = padded_sequences.shape[1]
    
    estimator = KerasClassifier(build_fn=lambda: create_model_2(vocab_size, max_length), epochs=epochs, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    y_pred = cross_val_predict(estimator, padded_sequences, Y.values.reshape(-1, 1), cv=kfold)
    
    accuracy = accuracy_score(Y, y_pred)
    precision = precision_score(Y, y_pred)
    recall = recall_score(Y, y_pred)
    f1 = f1_score(Y, y_pred)
    conf_matrix = confusion_matrix(Y, y_pred)
    
    logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": conf_matrix}

# Function for prediction
def predict_model_2(smiles):
    """
    Predict the binary property of a given SMILES string.

    Parameters:
        smiles: A SMILES string.

    Returns:
        Binary prediction (1 or 0).
    """
    
    model = load_model(MODEL_2_PATH)
    tokenizer, padded_sequences = tokenize_and_pad([smiles])
    sequence = tokenizer.texts_to_sequences([smiles])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=padded_sequences.shape[1])
    prediction = model.predict(padded_sequence)
    
    binary_prediction = 1 if prediction >= 0.5 else 0
    logging.info(f"Prediction for {smiles}: {binary_prediction} (Probability: {prediction[0][0]})")
    
    return binary_prediction
