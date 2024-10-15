import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from multiprocessing import Queue
import numpy as np
from typing import List, Tuple
import seaborn as sns

class StreamAnimator:
    def __init__(self, data_queue: Queue, max_points: int = 1000, interval: int = 50):
        self.max_points = max_points
        self.interval = interval
        self.data_queue = data_queue
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = GridSpec(3, 3, figure=self.fig)
        self.fig.suptitle("Real-time Data Stream Anomaly Detection", fontsize=16, color='white')
        
        self.line_plot = self.fig.add_subplot(self.gs[0, :])
        self.heatmap = self.fig.add_subplot(self.gs[1, :])
        self.histogram = self.fig.add_subplot(self.gs[2, 0:2])
        self.stats_display = self.fig.add_subplot(self.gs[2, 2])
        
        self.setup_line_plot()
        self.setup_heatmap()
        self.setup_histogram()
        self.setup_stats_display()
        
        self.data = []
        
    def setup_line_plot(self):
        self.line_plot.set_xlim(0, self.max_points)
        self.line_plot.set_ylim(0, 200)
        self.line_plot.set_title("Time Series", color='white')
        self.line_plot.set_xlabel("Time", color='white')
        self.line_plot.set_ylabel("Value", color='white')
        self.line_plot.grid(True, linestyle='--', alpha=0.3)
        self.line_plot.tick_params(colors='white')
        self.line, = self.line_plot.plot([], [], 'cyan', linewidth=1.5, alpha=0.8)
        self.normal_scatter = self.line_plot.scatter([], [], c='green', s=30, zorder=4, label='Normal')
        self.anomaly_scatter = self.line_plot.scatter([], [], c='red', s=50, zorder=5, label='Anomaly')
        self.line_plot.legend()

    def setup_heatmap(self):
        self.heatmap_data = np.zeros((1, self.max_points))
        self.heatmap_img = self.heatmap.imshow(self.heatmap_data, aspect='auto', cmap='viridis')
        self.heatmap.set_title("Anomaly Heatmap", color='white')
        self.heatmap.set_xlabel("Time", color='white')
        self.heatmap.set_ylabel("Anomaly", color='white')
        self.heatmap.tick_params(colors='white')

    def setup_histogram(self):
        self.histogram.set_title("Value Distribution", color='white')
        self.histogram.set_xlabel("Value", color='white')
        self.histogram.set_ylabel("Frequency", color='white')
        self.histogram.tick_params(colors='white')

    def setup_stats_display(self):
        self.stats_display.axis('off')
        self.stats_display.set_title("Statistics", color='white')
        self.stats_text = self.stats_display.text(0.1, 0.9, "", transform=self.stats_display.transAxes, color='white')

    def init_animation(self):
        return self.line, self.normal_scatter, self.anomaly_scatter, self.heatmap_img, self.stats_text

    def animate(self, frame):
        while not self.data_queue.empty():
            data = self.data_queue.get()
            if data is None:  # Stop signal
                plt.close(self.fig)
                return self.line, self.normal_scatter, self.anomaly_scatter, self.heatmap_img, self.stats_text
            self.data.append(data)

        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]

        t, values, anomalies = zip(*self.data)
        
        # Update line plot
        self.line.set_data(t, [v[0] for v in values])
        
        # Update scatter plots
        normal_t = [t[i] for i in range(len(t)) if not anomalies[i]]
        normal_values = [values[i][0] for i in range(len(values)) if not anomalies[i]]
        anomaly_t = [t[i] for i in range(len(t)) if anomalies[i]]
        anomaly_values = [values[i][0] for i in range(len(values)) if anomalies[i]]
        
        self.normal_scatter.set_offsets(np.column_stack((normal_t, normal_values)))
        self.anomaly_scatter.set_offsets(np.column_stack((anomaly_t, anomaly_values)))
        
        # Update heatmap
        self.heatmap_data = np.roll(self.heatmap_data, -1, axis=1)
        self.heatmap_data[0, -1] = 1 if anomalies[-1] else 0
        self.heatmap_img.set_array(self.heatmap_data)
        
        # Update histogram
        self.histogram.clear()
        sns.histplot([v[0] for v in values], ax=self.histogram, kde=True, color='cyan')
        self.histogram.set_title("Value Distribution", color='white')
        self.histogram.set_xlabel("Value", color='white')
        self.histogram.set_ylabel("Frequency", color='white')
        self.histogram.tick_params(colors='white')
        
        # Update statistics
        anomaly_rate = sum(anomalies) / len(anomalies)
        mean_value = np.mean([v[0] for v in values])
        std_value = np.std([v[0] for v in values])
        stats_text = f"Anomaly Rate: {anomaly_rate:.2f}\nMean Value: {mean_value:.2f}\nStd Dev: {std_value:.2f}"
        self.stats_text.set_text(stats_text)
        
        return self.line, self.normal_scatter, self.anomaly_scatter, self.heatmap_img, self.stats_text

    def run_animation(self):
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
                                       interval=self.interval, blit=True)
        plt.tight_layout()
        plt.show()

def run_animation_process(data_queue: Queue):
    animator = StreamAnimator(data_queue)
    animator.run_animation()

if __name__ == "__main__":
    # This is just for testing the animation separately
    from multiprocessing import Queue
    import time
    import random

    test_queue = Queue()
    
    def generate_test_data():
        for i in range(1000):
            value = random.random() * 100
            is_anomaly = random.random() < 0.05
            test_queue.put((i, value, is_anomaly))
            time.sleep(0.01)
        test_queue.put(None)  # Signal to stop

    import threading
    data_thread = threading.Thread(target=generate_test_data)
    data_thread.start()
    
    run_animation_process(test_queue)
