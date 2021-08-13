
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRULayer, self).__init__()
        self._num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: F.Tensor, hxs: F.Tensor):
        # (T=1): x=[N, input_size], hxs=[N, L, hidden_size]
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), hxs.transpose(0, 1).contiguous())
            x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]
        # (Chunk): x=[T * N, input_size], hxs=[N, L, hidden_size]
        else:
            N = hxs.size(0)
            T = x.size(0) // N
            x, hxs = self.gru(x.view(T, N, -1), hxs.transpose(0, 1).contiguous())
            x = x.view(T * N, -1)       # [T, N, input_size] => [T * N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]
        x = self.norm(x)
        return x, hxs


if __name__ == "__main__":
    input_size, hidden_size, num_layers = 5, 10, 1
    print("\n---------test GRULayer---------\n")
    grulayer = GRULayer(input_size, hidden_size, num_layers)

    print("ONE")
    x = torch.rand(1, input_size)
    hxs = torch.rand(1, num_layers, hidden_size)
    print(" input size: x=", x.shape, "hxs=", hxs.shape)
    x_, hxs_ = grulayer(x, hxs)
    print(" output size: x=", x_.shape, "hxs=", hxs_.shape)

    print("BATCH")
    x = torch.rand(4, input_size)
    hxs = torch.rand(4, num_layers, hidden_size)
    print(" input size: x=", x.shape, "hxs=", hxs.shape)
    x_, hxs_ = grulayer(x, hxs)
    print(" output size: x=", x_.shape, "hxs=", hxs_.shape)

    print("CHUNK")
    x = torch.rand(5*4, input_size)
    hxs = torch.rand(4, num_layers, hidden_size)
    print(" input size: x=", x.shape, "hxs=", hxs.shape)
    x_, hxs_ = grulayer(x, hxs)
    print(" output size: x=", x_.shape, "hxs=", hxs_.shape)
