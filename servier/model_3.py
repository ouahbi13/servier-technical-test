import pandas as pd
import numpy as np
import logging
import configparser
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
import pandas as pd
import numpy as np
import logging
import configparser
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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
DATASET_MULTI_PATH = config['DEFAULT']['DATASET_MULTI_PATH']
MODEL_3_PATH = config['DEFAULT']['MODEL_3_PATH']

# Function to load multi-label dataset
def load_dataset(file_path):
    """
    Tokenize and pad the SMILES sequences.

    Parameters:
        X: Pandas Series containing SMILES strings.

    Returns:
        tokenizer: Fitted tokenizer.
        padded_sequences: Numpy array of padded sequences.
    """
    
    try:
        data = pd.read_csv(file_path)
        X = data['smiles']  # Considering only SMILES as features for simplicity
        Y = data.drop(['smiles', 'mol_id'], axis=1)  # Taking all properties as labels
        logging.info(f"Multi-label dataset loaded from {file_path}")
        return X, Y
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Load the multi-label dataset from a CSV file
X, Y = load_dataset(DATASET_MULTI_PATH)

# Function to tokenize and pad sequences
def tokenize_and_pad(X):
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded_sequences = pad_sequences(sequences, padding='post')
    return tokenizer, padded_sequences

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Modify the model creation function to support multi-label classification
def create_model_3(vocab_size, max_length):
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
        Dense(9, activation='sigmoid')  # Changed to 9 neurons for P1 to P9
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Modify training function
def train_model_3(epochs=100, batch_size=64):
    """
    Train the model on the training dataset and save it to a file.

    Parameters:
        epochs (int): The number of epochs for training the model.
        batch_size (int): The batch size for training the model.
    """
    
    tokenizer, padded_sequences = tokenize_and_pad(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = padded_sequences.shape[1]
    model = create_model_3(vocab_size, max_length)
    model.fit(padded_sequences, y_train.values, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save(MODEL_3_PATH)
    logging.info("Model 3 saved.")
    

def evaluate_model_3(epochs=100, batch_size=64, n_splits=5):
    """
    Evaluate the model on a test set and print various performance metrics.

    Parameters:
        X_test: Test features.
        Y_test: True labels for the test set.

    Returns:
        Various classification metrics.
    """
    
    # Tokenize and pad the SMILES sequences
    tokenizer, padded_sequences = tokenize_and_pad(X)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = padded_sequences.shape[1]
    
    estimator = KerasClassifier(build_fn=lambda: create_model_3(vocab_size, max_length), epochs=epochs, batch_size=batch_size)
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    y_pred = cross_val_predict(estimator, padded_sequences, Y.values, cv=kfold)

    hamming = hamming_loss(Y, y_pred)
    accuracy = accuracy_score(Y, y_pred)
    class_report = classification_report(Y, y_pred, target_names=Y.columns)

    logging.info(f"Hamming Loss: {hamming}")
    logging.info(f"Subset Accuracy: {accuracy}")
    logging.info("Classification Report:")
    logging.info(class_report)

    return {"hamming_loss": hamming, "subset_accuracy": accuracy, "classification_report": class_report}


# Modify prediction function to return probabilities for each property
def predict_model_3(smiles):
    """
    Predict the binary properties P1 to P9 of a given SMILES string.

    Parameters:
        smiles: A SMILES string.

    Returns:
        A dictionary containing binary predictions for each property.
    """
    
    model = load_model(MODEL_3_PATH)
    tokenizer, _ = tokenize_and_pad([smiles])
    sequence = tokenizer.texts_to_sequences([smiles])
    padded_sequence = pad_sequences(sequence, padding='post')
    predictions = model.predict(padded_sequence)[0]  # Get the first (and only) prediction in the batch
    # Converting probabilities to binary labels based on a threshold
    binary_predictions = (predictions >= 0.5).astype(int)
    result = {f"P{i+1}": binary_predictions[i] for i in range(len(binary_predictions))}
    
    logging.info(f"Predictions for {smiles}: {result}")
    
    return predictions
