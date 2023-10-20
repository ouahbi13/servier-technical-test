import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Sequential
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model


DATASET_SINGLE_PATH = 'servier/data/dataset_single.csv'
MODEL_2_PATH = 'servier/models/model2/model_2.h5'

df = pd.read_csv(DATASET_SINGLE_PATH)
X = df.drop(['P1', 'mol_id'], axis=1)
Y = df['P1']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(X['smiles'])

def create_model_2(vocab_size, embedding_dim, padded_sequences):
    """
        Create the Model:
            - Embedding Layer
            - LSTM Layer with 128 units
            - LSTM Layer with 64 units
            - Dense Layer
            - Activation Function: Sigmoid
    """
    # Create model
    model_2 = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=padded_sequences.shape[1]),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model_2
    
def train_model_2(epochs, batch_size):
    """
        Train the Model on the training set
    """
    sequences = tokenizer.texts_to_sequences(X_train['smiles'])
    padded_sequences = pad_sequences(sequences, padding='post')
    embedding_dim = 64
    vocab_size = len(tokenizer.word_index) + 1
    
    model_2 = create_model_2(vocab_size, embedding_dim, padded_sequences)
    model_2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    model_2.save(MODEL_2_PATH)
    

def evaluate_model_2(epochs, batch_size, n_splits):
    """
        Evaluate the model using Cross-Validation
    """
    sequences = tokenizer.texts_to_sequences(X['smiles'])
    padded_sequences = pad_sequences(sequences, padding='post')

    estimator = KerasClassifier(model=create_model_2, epochs=epochs, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    results = cross_val_score(estimator, padded_sequences.values, Y.values.reshape(-1, 1), cv=kfold)
    
    return results



def predict_model_2(smiles):
    """
        Predict the binary property of a SMILES
    """
    model_2 = load_model(MODEL_2_PATH)
    sequence = tokenizer.texts_to_sequences([smiles])
    padded_sequence = pad_sequences(sequence, padding='post')
    prediction = model_2.predict(padded_sequence)
    
    return 1 if prediction >= 0.5 else 0