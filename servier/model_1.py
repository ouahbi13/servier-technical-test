import pandas as pd
import numpy as np
import configparser
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from servier.feature_extractor import fingerprint_features  # Assuming this is a custom module
import matplotlib.pyplot as plt
import seaborn as sns


# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read config file
config = configparser.ConfigParser()
try:
    config.read('servier/config.ini')
    logging.info("Config file loaded successfully.")
except Exception as e:
    logging.error(f"Error reading the config file: {e}")
    raise

# Constants from config file
DATASET_SINGLE_PATH = config['DEFAULT']['DATASET_SINGLE_PATH']
MODEL_1_PATH = config['DEFAULT']['MODEL_1_PATH']


# Function to load dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Optimized feature extraction
def extract_features(smiles):
    try:
        features = np.frombuffer(fingerprint_features(smiles).ToBitString().encode(), 'u1') - ord('0')
        return features
    except Exception as e:
        logging.error(f"Error extracting features for SMILES {smiles}: {e}")
        return None

# Load the dataset
df = load_dataset(DATASET_SINGLE_PATH)

# Extract features and prepare the dataset
try:
    fingerprints = df['smiles'].apply(extract_features)
    X = pd.DataFrame(fingerprints.tolist(), columns=[f'Bit_{i}' for i in range(fingerprints.iloc[0].shape[0])])
    Y = df['P1']
    INPUT_SIZE = X.shape[1]  # Update INPUT_SIZE based on the actual input features
    logging.info("Features extracted and dataset prepared")
except Exception as e:
    logging.error(f"Error in feature extraction or dataset preparation: {e}")
    raise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)


def create_model_1():
    """
    Create and return a Sequential model with specified layers and activation functions.

    Returns:
        model: A Sequential model with input, hidden, and output layers.
    """
    # Create a Sequential model
    model = Sequential([
        Dense(1024, input_dim=INPUT_SIZE, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_model_1(epochs=100, batch_size=64):
    """
    Train the model on the training dataset and save it to a file.

    Parameters:
        epochs (int): The number of epochs for training the model.
        batch_size (int): The batch size for training the model.
    """
    model_1 = create_model_1()
    model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model_1.save(MODEL_1_PATH)
    logging.info("Model 1 saved.")


def evaluate_model_1(epochs=100, batch_size=64, n_splits=5):
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
    try:
        estimator = KerasClassifier(build_fn=create_model_1, epochs=epochs, batch_size=batch_size)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        # Getting cross-validated predictions
        y_pred = cross_val_predict(estimator, X.values, Y.values, cv=kfold)
        
        # Computing metrics
        accuracy = accuracy_score(Y.values, y_pred)
        precision = precision_score(Y.values, y_pred)
        recall = recall_score(Y.values, y_pred)
        f1 = f1_score(Y.values, y_pred)
        conf_matrix = confusion_matrix(Y.values, y_pred)
        
        # Logging and returning results
        logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        # Plotting confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="g", cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        return {"precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": conf_matrix}

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise
    
def load_trained_model():
    try:
        loaded_model = load_model(MODEL_1_PATH)
        logging.info("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise



def predict_model_1(smiles):
    """
    Predict the binary property of a given SMILES string using the trained model.

    Parameters:
        smiles (str): The SMILES string of the chemical compound.

    Returns:
        tuple: Binary prediction (1 or 0).
    """
    loaded_model = load_trained_model()

    try:
        features = np.frombuffer(fingerprint_features(smiles).ToBitString().encode(), 'u1') - ord('0')
        prediction = loaded_model.predict(np.array([features]))
        
        binary_prediction = 1 if prediction >= 0.5 else 0
        logging.info(f"Prediction: {binary_prediction}, Probability: {prediction[0][0]}")
        
        return binary_prediction
    
    except Exception as e:
        logging.error(f"Error during prediction for SMILES {smiles}: {e}")
        raise
    