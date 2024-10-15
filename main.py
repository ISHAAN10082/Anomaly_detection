import sys
import logging
import argparse
import json
import time
from multiprocessing import Queue, Process
from data_stream_generator import DataStreamGenerator
from anomaly_detector import AnomalyDetector
from stream_animator import  run_animation_process
from utils import prepare_data_for_lstm, setup_logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='anomaly_detection.log'
)
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log uncaught exceptions.
    
    Args:
    exc_type: Type of the exception
    exc_value: Exception instance
    exc_traceback: Traceback object
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
    config_path (str): Path to the configuration file
    
    Returns:
    dict: Loaded configuration
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.exception(f"Error loading configuration from {config_path}")
        sys.exit(1)

def data_generator_with_anomalies(config, data_generator, anomaly_detector, data_queue):
    """
    Generator function that yields data points with anomaly detection results.
    
    Args:
    config (dict): Configuration dictionary
    data_generator (DataStreamGenerator): Instance of DataStreamGenerator
    anomaly_detector (AnomalyDetector): Instance of AnomalyDetector
    data_queue (Queue): Queue to store data points
    
    Yields:
    tuple: (data_point, is_anomaly, anomaly_score, accuracies)
    """
    window = []
    accuracies = {'combined': []}
    for t, (data_point, true_anomaly) in enumerate(data_generator.generate_stream()):
        window.append(data_point[0])
        if len(window) > config['anomaly_detector']['window_size']:
            window.pop(0)

        if len(window) == config['anomaly_detector']['window_size']:
            is_anomaly, anomaly_score = anomaly_detector.detect(window)
            
            X, y = prepare_data_for_lstm(window, config['anomaly_detector']['sequence_length'])
            val_X, val_y = prepare_data_for_lstm(window[-config['anomaly_detector']['sequence_length']:], config['anomaly_detector']['sequence_length'])
            anomaly_detector.models['adaptive_lstm'].update(X, y, val_X, val_y)

            accuracies['combined'].append(int(is_anomaly == true_anomaly))

            data_queue.put((t, data_point, is_anomaly))

            if t % config.get('log_interval', 100) == 0:
                avg_accuracy = sum(accuracies['combined']) / len(accuracies['combined'])
                logging.info(f"Average accuracy: {avg_accuracy:.2f}")

            time.sleep(config.get('update_interval', 0.1))  # Default to 0.1 if not specified

    data_queue.put(None)  # Signal to stop the animation

def main(config):
    """
    Main function to run the anomaly detection system.
    
    Args:
    config (dict): Configuration dictionary
    """
    try:
        data_generator = DataStreamGenerator(**config.get('data_generator', {}))
        anomaly_detector = AnomalyDetector(config.get('anomaly_detector', {}))
        
        data_queue = Queue()
        animation_process = Process(target=run_animation_process, args=(data_queue,))
        animation_process.start()

        data_generator_with_anomalies(config, data_generator, anomaly_detector, data_queue)

        animation_process.join()

    except KeyError as e:
        logging.error(f"Missing configuration: {str(e)}")
    except Exception as e:
        logging.exception("An error occurred in the main execution")
    finally:
        logging.info("Program terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    if 'logging' in config:
        setup_logging(config['logging'])
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(config)
