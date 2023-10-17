import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .feature_extractor import fingerprint_features
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model


DATASET_SINGLE_PATH = './servier/data/dataset_single.csv'
df = pd.read_csv(DATASET_SINGLE_PATH)

# Create dataframe with the 2048 features
fingerprints = df['smiles'].apply(lambda x: (np.frombuffer(fingerprint_features(x).ToBitString().encode(), 'u1') - ord('0')))
X = pd.DataFrame(fingerprints.to_list(), columns=[f'Bit_{i}' for i in range(fingerprints[0].shape[0])])
Y = df['P1']

INPUT_SIZE = X.shape[1] # input size = 2048
EPOCHS = 100
BATCH_SIZE = 64
MODEL_PATH = './servier/models/model1/model_1.h5'

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

def create_model_1():
    """
        Create the Model:
            - Input Layer: 2048 neurons
            - Hidden Layer 1: 1024 neurons
            - Activation Function: relu
            - Hidden Layer 2: 512 neurons
            - Activation Function: relu
            - Hidden Layer 3: 256 neurons
            - Activation Function: relu
            - Output Layer: 1 neuron
            - Activation Function: sigmoid
    """
    
    # Create model
    model = Sequential()
    model.add(Dense(1024, input_shape=(INPUT_SIZE,), activation='relu'))
    model.add(Dense(512, input_shape=(INPUT_SIZE,), activation='relu'))
    model.add(Dense(256, input_shape=(INPUT_SIZE,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
    
def train_model_1(epochs, batch_size):
    """
        Train the Model on the training set
    """
    model_1 = create_model_1()

    # Train model
    model_1.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    # Save model
    model_1.save(MODEL_PATH)
        

def evaluate_model_1(epochs, batch_size, n_splits):
    """
        Evaluate the model using Cross-Validation
    """
    estimator = KerasClassifier(model=create_model_1, epochs=epochs, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    results = cross_val_score(estimator, X.values, Y.values.reshape(-1, 1), cv=kfold)
    
    return results

def predict_model_1(smiles):
    """
        Predict the binary property of a SMILES
    """
    model_1 = load_model(MODEL_PATH)
    feature = fingerprint_features(smiles)
    prediction = model_1.predict(np.array([feature]))
    
    return 1 if prediction >= 0.5 else 0

    
