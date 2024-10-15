
## Comprehensive Documentation

### Overview of the Codebase

This project implements a real-time anomaly detection system with visualization capabilities. The system is designed to process a continuous stream of data, detect anomalies using various algorithms, and visualize the results in real-time.

### Main Components

1. **Data Generation**: `data_stream_generator.py`
2. **Anomaly Detection**: `anomaly_detector.py`
3. **Visualization**: `stream_animator.py`
4. **Main Orchestration**: `main.py`
5. **Utility Functions**: `utils.py`

### Data Flow and Pipeline

1. **Data Generation**:
   - The `DataStreamGenerator` class in `data_stream_generator.py` creates a synthetic data stream.
   - It simulates normal data points and introduces anomalies based on configured parameters.

2. **Anomaly Detection**:
   - The `AnomalyDetector` class in `anomaly_detector.py` processes the data stream.
   - It implements multiple detection algorithms: Isolation Forest, LSTM, and Adaptive LSTM.
   - Each data point is classified as normal or anomalous.

3. **Visualization**:
   - The `StreamAnimator` class in `stream_animator.py` handles real-time visualization.
   - It creates an animated plot showing the data stream, detected anomalies, and statistics.

4. **Main Orchestration**:
   - `main.py` is the entry point of the application.
   - It sets up logging, loads configuration, initializes components, and manages the overall process flow.

### Detailed Process Flow

1. `main.py` starts by loading the configuration from `config.json`.
2. It initializes the `DataStreamGenerator`, `AnomalyDetector`, and sets up the visualization process.
3. The main loop in `data_generator_with_anomalies` function:
   - Generates data points using `DataStreamGenerator`.
   - Passes each point through the `AnomalyDetector`.
   - Sends the results to a queue for visualization.
4. A separate process runs `run_animation_process`, which continuously reads from the queue and updates the visualization.

### Configuration

The `config.json` file allows customization of various parameters:
- Data generation settings (e.g., anomaly rate, data dimensions)
- Anomaly detection algorithm parameters
- Visualization settings

### Running the System

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Adjust `config.json` as needed.
3. Run `python main.py`

### Output

- Real-time visualization of the data stream and detected anomalies.
- Log file (`anomaly_detection.log`) with runtime information and statistics.

### Extending the System

- To add new anomaly detection algorithms, extend the `AnomalyDetector` class in `anomaly_detector.py`.
- For different data generation methods, modify `DataStreamGenerator` in `data_stream_generator.py`.
- Enhance visualization by updating `StreamAnimator` in `stream_animator.py`.

This system provides a flexible framework for real-time anomaly detection with easy configuration and extensibility.



