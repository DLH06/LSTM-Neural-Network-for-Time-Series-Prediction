import torch
import torch.nn as nn

from core.utils import Timer


class Net(nn.Module):
    def __init__(self, dropout_rate=0.2, input_dim=2):
        super().__init__()
        self.first_lstm = nn.LSTM(input_dim, 100, batch_first=True, dropout=dropout_rate)
        self.second_lstm = nn.LSTM(100, 100)
        self.third_lstm = nn.LSTM(100, 100, dropout=dropout_rate)
        self.linear = nn.Linear(100, 1)

    def forward(self, x):
        o_1, (h_s_1, c_s_1) = self.first_lstm(x)
        o_2, (h_s_2, c_s_2) = self.second_lstm(o_1)
        o_3, (h_s_3, c_s_3) = self.second_lstm(o_2)
        # output = self.linear(h_s_3)
        return o_3, (h_s_3, c_s_3)


if __name__ == '__main__':
    model = Net()
    input_sample = torch.rand(3, 50, 2)
    output, (hidden_state, cell_state) = model(input_sample)
    print(hidden_state.shape)