import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class SP500(Dataset):
    """A class for loading and transforming data for the lstm model"""

    def __init__(
        self,
        filename="data/sp500.csv",
        cols=["Close", "Volume"],
        seq_len=51,
        normalise=False,
    ):
        dataframe = pd.read_csv(filename)
        self.raw_data = dataframe.get(cols).values[:]
        self.data = self._split_seq_data(seq_len, normalise)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        window = self.data[index]
        x = window[:-1]
        y = window[
            -1,
        ][0]
        return torch.as_tensor(np.array(x).astype("float32")), torch.as_tensor(
            np.array(y).astype("float32")
        )

    def _split_seq_data(self, seq_len, normalise):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        split_data = []
        for i in tqdm(range(len(self.raw_data) - seq_len), desc="Loading dataset"):
            window = self.raw_data[i : i + seq_len]
            window = (
                self._normalise_windows(window, single_window=True)[0]
                if normalise
                else window
            )
            split_data.append(window)
        return np.array(split_data)

    def _normalise_windows(self, window_data, single_window=False):
        """Normalise window with a base value of zero"""
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [
                    ((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]
                ]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window
            ).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
