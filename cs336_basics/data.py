from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Calculate the maximum valid starting index
    # We need context_length tokens for input and 1 more token for the label
    max_start_idx = len(dataset) - context_length - 1
    
    # Randomly sample starting indices
    starting_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Extract input sequences (x) and target sequences (y)
    # x: tokens from start_idx to start_idx + context_length
    # y: tokens from start_idx + 1 to start_idx + context_length + 1 (shifted by 1)
    x = np.array([dataset[i:i + context_length] for i in starting_indices])
    y = np.array([dataset[i + 1:i + context_length + 1] for i in starting_indices])
    
    # Convert to PyTorch tensors and move to the specified device
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    
    return x_tensor, y_tensor
