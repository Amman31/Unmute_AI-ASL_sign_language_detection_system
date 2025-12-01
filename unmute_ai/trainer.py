"""
Model training module.
Trains a RandomForest classifier on hand landmark data.
"""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .config import DATA_PICKLE_FILE, MODEL_FILE, TEST_SIZE, RANDOM_STATE


def train_model() -> None:
    """
    Train a RandomForest classifier on the processed dataset.
    Saves the trained model to a pickle file.
    """
    if not DATA_PICKLE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PICKLE_FILE}\n"
            f"Please run create_dataset.py first to generate the dataset."
        )
    
    print(f"Loading dataset from {DATA_PICKLE_FILE}...")
    with open(DATA_PICKLE_FILE, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    
    print(f"Dataset loaded: {len(data)} samples, {len(set(labels))} classes")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data, 
        labels, 
        test_size=TEST_SIZE, 
        shuffle=True, 
        stratify=labels,
        random_state=RANDOM_STATE
    )
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Train model
    print("Training RandomForest classifier...")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    # Evaluate
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    
    print(f"Model accuracy: {score * 100:.2f}%")
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f"Model saved to {MODEL_FILE}")

