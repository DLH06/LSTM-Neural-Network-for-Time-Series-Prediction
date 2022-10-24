import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dropout_prob=0.2, input_dim=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.first_lstm = nn.LSTM(input_dim, 100, batch_first=True)
        self.second_lstm = nn.LSTM(100, 100)
        self.third_lstm = nn.LSTM(100, 100)
        self.linear = nn.Linear(100, 1)

    def forward(self, x):
        o_1, (h_s_1, c_s_1) = self.first_lstm(x)
        o_1 = self.dropout(o_1)
        o_2, (h_s_2, c_s_2) = self.second_lstm(o_1)
        o_3, (h_s_3, c_s_3) = self.third_lstm(o_2)
        output = self.linear(h_s_3)

        # return o_3, (h_s_3, c_s_3)
        return output


if __name__ == '__main__':
    model = Net()
    input_sample = torch.rand(3, 50, 2)
    output = model(input_sample)
    print(output.shape)
    # output = output.squeeze(0)
    output = output.view(-1, 50)
    print(output.shape)
    