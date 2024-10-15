import time
from typing import Generator, Any
from functools import wraps
import logging
import torch

def process_stream(stream: Generator[Any, None, None], max_iterations: int = None) -> Generator[Any, None, None]:
    """
    Process the data stream, optionally limiting the number of iterations.
    """
    if max_iterations is None:
        yield from stream
    else:
        for i, value in enumerate(stream):
            if i >= max_iterations:
                break
            yield value
            time.sleep(0.01)

def timeit(func):
    """
    A decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def moving_average(data, window_size):
    """
    Calculate the moving average of a list of numbers.
    """
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]

def z_score(data):
    """
    Calculate the z-score for each point in a list of numbers.
    """
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std for x in data]

def setup_logging(config):
    """
    Set up logging configuration.

    Args:
        config (dict): Logging configuration dictionary.
    """
    logging.basicConfig(
        level=config['level'],
        format=config['format'],
        filename=config['filename'],
        filemode='w'
    )

    # Add console handler if enabled
    if config['console_output']:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(config['format'])
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def prepare_data_for_lstm(data, sequence_length):
    try:
        sequences = []
        targets = []

        if len(data) > sequence_length:
            for i in range(len(data) - sequence_length):
                seq = data[i:i+sequence_length]
                target = data[i+sequence_length]
                sequences.append(seq)
                targets.append(target)

            sequences = torch.FloatTensor(sequences).unsqueeze(-1)  # Add feature dimension
            targets = torch.FloatTensor(targets).unsqueeze(-1)  # Add feature dimension
        else:
            # Handle case where data is shorter than sequence_length
            sequences = torch.FloatTensor([]).unsqueeze(-1)
            targets = torch.FloatTensor([]).unsqueeze(-1)

        return sequences, targets
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Error in preparing data for LSTM")
        return torch.FloatTensor(), torch.FloatTensor()
