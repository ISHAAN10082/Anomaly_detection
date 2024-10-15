import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.line1, = self.ax1.plot([], [], lw=2)
        self.scatter1 = self.ax1.scatter([], [], c='red', marker='x')
        self.line2_if, = self.ax2.plot([], [], lw=2, label='Isolation Forest')
        self.line2_lstm, = self.ax2.plot([], [], lw=2, label='LSTM')
        self.line2_adaptive, = self.ax2.plot([], [], lw=2, label='Adaptive LSTM')
        self.ax2.legend()
        
    def init(self):
        self.ax1.set_xlim(0, self.config['window_size'])
        self.ax1.set_ylim(-5, 5)
        self.ax1.set_title('Data Stream with Anomalies')
        self.ax2.set_xlim(0, self.config['window_size'])
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Detection Accuracy over Time')
        return self.line1, self.scatter1, self.line2_if, self.line2_lstm, self.line2_adaptive

    def update(self, frame):
        data, anomalies, accuracies = frame
        self.line1.set_data(range(len(data)), data)
        self.scatter1.set_offsets(np.c_[np.where(anomalies)[0], np.array(data)[anomalies]])
        self.line2_if.set_data(range(len(accuracies['isolation_forest'])), accuracies['isolation_forest'])
        self.line2_lstm.set_data(range(len(accuracies['lstm'])), accuracies['lstm'])
        self.line2_adaptive.set_data(range(len(accuracies['adaptive_lstm'])), accuracies['adaptive_lstm'])
        return self.line1, self.scatter1, self.line2_if, self.line2_lstm, self.line2_adaptive

    def animate(self, data_generator):
        ani = FuncAnimation(self.fig, self.update, frames=data_generator,
                            init_func=self.init, blit=True, interval=200)
        plt.show()
