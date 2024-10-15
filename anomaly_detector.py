from sklearn.ensemble import IsolationForest
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# LSTM Model: A deep learning model effective for sequence prediction and anomaly detection
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.relu(lstm_out[:, -1, :]))
        return out

# LSTM Detector: Uses LSTM to detect anomalies in time series data
class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(0)  # Remove batch dimension if input was unbatched

# Early Stopping: Prevents overfitting by stopping training when validation loss doesn't improve
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Adaptive LSTM Detector: An LSTM model that can adapt to changing patterns in the data
class AdaptiveLSTMDetector(LSTMDetector):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, patience=7):
        super(AdaptiveLSTMDetector, self).__init__(input_size, hidden_size, num_layers, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=patience)

    def update(self, x, y, val_x, val_y):
        # Training step
        self.train()
        self.optimizer.zero_grad()
        
        # Ensure x and y have batch dimension and non-zero length
        if x.dim() == 2 and x.size(0) > 0:
            x = x.unsqueeze(0)
        if y.dim() == 1 and y.size(0) > 0:
            y = y.unsqueeze(0)
        
        if x.size(0) > 0 and y.size(0) > 0:
            output = self(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

        # Validation step
        self.eval()
        with torch.no_grad():
            # Ensure val_x and val_y have batch dimension and non-zero length
            if val_x.dim() == 2 and val_x.size(0) > 0:
                val_x = val_x.unsqueeze(0)
            if val_y.dim() == 1 and val_y.size(0) > 0:
                val_y = val_y.unsqueeze(0)
            
            if val_x.size(0) > 0 and val_y.size(0) > 0:
                val_output = self(val_x)
                val_loss = self.loss_fn(val_output, val_y)
                self.early_stopping(val_loss.item())

# Isolation Forest Detector: Effective for detecting anomalies in high-dimensional datasets
class IsolationForestDetector:
    def __init__(self, contamination):
        self.model = IsolationForest(contamination=contamination)

    def fit(self, X):
        self.model.fit(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

# Main Anomaly Detector: Combines multiple models for robust anomaly detection
class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.models = {}
        
        # Initialize different models based on configuration
        if 'lstm_detector' in config:
            self.models['lstm'] = LSTMDetector(**config['lstm_detector'])
        
        if 'adaptive_lstm_detector' in config:
            self.models['adaptive_lstm'] = AdaptiveLSTMDetector(**config['adaptive_lstm_detector'])
        
        if 'isolation_forest' in config:
            self.models['isolation_forest'] = IsolationForestDetector(**config['isolation_forest'])
        
        self.window_size = config.get('window_size', 100)  # Default to 100 if not specified
        self.data_buffer = []

    def detect(self, data_point):
        try:
            # Maintain a sliding window of data
            self.data_buffer.append(data_point)
            if len(self.data_buffer) > self.window_size:
                self.data_buffer.pop(0)

            if len(self.data_buffer) == self.window_size:
                X = torch.FloatTensor(self.data_buffer).unsqueeze(0)
                
                # Get scores from each model
                scores = {}
                if 'lstm' in self.models:
                    scores['lstm'] = self.models['lstm'](X).item()
                if 'adaptive_lstm' in self.models:
                    scores['adaptive_lstm'] = self.models['adaptive_lstm'](X).item()
                if 'isolation_forest' in self.models:
                    scores['isolation_forest'] = self.models['isolation_forest'].decision_function([self.data_buffer])[0]

                # Combine scores and determine if it's an anomaly
                if scores:
                    combined_score = sum(scores.values()) / len(scores)
                    is_anomaly = combined_score > self.config['anomaly_threshold']
                    return is_anomaly, combined_score
                
            return False, 0
        except Exception as e:
            logger.exception("Error in anomaly detection")
            return False, 0
