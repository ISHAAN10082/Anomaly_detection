import numpy as np
from typing import Generator, Tuple, List

class DataStreamGenerator:
    def __init__(self, mean=0, std_dev=1, anomaly_freq=0.05, anomaly_probability=None, anomaly_scale=3, anomaly_std_dev=None, seasonal_period=24, trend_strength=0.1, dimensions=1):
        self.mean = mean
        self.std_dev = std_dev
        self.anomaly_freq = anomaly_probability if anomaly_probability is not None else anomaly_freq
        self.anomaly_scale = anomaly_std_dev if anomaly_std_dev is not None else anomaly_scale
        self.seasonal_period = seasonal_period
        self.trend_strength = trend_strength
        self.dimensions = dimensions
        self.t = 0

    def generate_stream(self) -> Generator[Tuple[List[float], bool], None, None]:
        while True:
            self.t += 1
            trend = self.trend_strength * self.t
            seasonality = np.sin(2 * np.pi * self.t / self.seasonal_period) * 10
            base_value = self.mean + trend + seasonality
            
            is_anomaly = np.random.random() < self.anomaly_freq
            
            if is_anomaly:
                values = [np.random.normal(base_value, self.std_dev * self.anomaly_scale) for _ in range(self.dimensions)]
            else:
                values = [np.random.normal(base_value, self.std_dev) for _ in range(self.dimensions)]
            
            # Add sudden shifts
            if self.t % 500 == 0:
                self.mean += np.random.normal(0, 20)
            
            # Add cyclical patterns
            cyclical_pattern = np.sin(2 * np.pi * self.t / 100) * 5
            values = [v + cyclical_pattern for v in values]
            
            yield values, is_anomaly
