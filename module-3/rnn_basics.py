import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional

import matplotlib.pyplot as plt

# ==================== 1. SimpleRNN Basics ====================
def simple_rnn_example():
    """Basic SimpleRNN for sequence modeling"""
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(10, 32), return_sequences=True),
        SimpleRNN(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==================== 2. LSTM Basics ====================
def lstm_model():
    """LSTM model for sequence prediction"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(10, 32), return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==================== 3. Sequence Generation ====================
def bidirectional_lstm():
    """Bidirectional LSTM for better context understanding"""
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(10, 32)),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])
    return model

# ==================== 4. Generate Synthetic Data ====================
def create_synthetic_sequence(n_samples=100, seq_length=10, feature_dim=32):
    """Create synthetic sequential data"""
    X = np.random.randn(n_samples, seq_length, feature_dim)
    # Simple rule: sum of sequence elements predicts y
    y = X.sum(axis=(1, 2)).reshape(-1, 1)
    return X, y

# ==================== 5. Train & Evaluate ====================
def train_model():
    """Complete training pipeline"""
    # Generate data
    X, y = create_synthetic_sequence()
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train model
    model = lstm_model()
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss[0]:.4f}")
    
    return model, history

# ==================== 6. Text Sequence Example ====================
def text_sequence_model(vocab_size=1000, max_length=50):
    """LSTM for text/NLP tasks"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32, dropout=0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==================== Main ====================
if __name__ == "__main__":
    print("1. Training LSTM Model...")
    model, history = train_model()
    
    print("\n2. Model Summary:")
    model.summary()
    
    print("\n3. Making Predictions...")
    X_test, _ = create_synthetic_sequence(n_samples=5)
    predictions = model.predict(X_test)
    print(f"Sample predictions: {predictions.flatten()}")