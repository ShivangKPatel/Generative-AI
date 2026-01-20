import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        """Forward propagation"""
        self.cache = [(None, None, X)]
        A = X
        
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:  # Output layer
                A = self.sigmoid(Z)
            else:  # Hidden layers
                A = self.relu(Z)
            
            self.cache.append((Z, A, A))
        
        return A
    
    def backward(self, y_true):
        """Backpropagation"""
        m = y_true.shape[0]
        
        # Output layer error
        dA = self.cache[-1][1] - y_true  # Binary cross-entropy derivative
        
        for i in reversed(range(len(self.weights))):
            Z, A, _ = self.cache[i + 1]
            A_prev = self.cache[i][2]
            
            # Gradient for weights and biases
            dZ = dA * (self.sigmoid_derivative(A) if i == len(self.weights) - 1 else self.relu_derivative(Z))
            dW = np.dot(A_prev.T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB
            
            # Propagate error backward
            dA = np.dot(dZ, self.weights[i].T)
    
    def train(self, X, y, epochs=100, batch_size=32):
        """Train the MLP"""
        losses = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Calculate loss
                loss = -np.mean(y_batch * np.log(output + 1e-8) + 
                               (1 - y_batch) * np.log(1 - output + 1e-8))
                epoch_loss += loss
                
                # Backward pass
                self.backward(y_batch)
            
            losses.append(epoch_loss / (X.shape[0] // batch_size))
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) > 0.5).astype(int)


# Example usage
if __name__ == "__main__":
    # Generate dataset
    X, y = make_classification(n_samples=400, n_features=4, n_classes=2, 
                               n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)
    
    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train MLP
    mlp = MLP([4, 8, 4, 1], learning_rate=0.1)
    losses = mlp.train(X_train, y_train, epochs=100, batch_size=16)
    
    # Evaluate
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()