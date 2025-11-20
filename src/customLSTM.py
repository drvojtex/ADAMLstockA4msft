import torch
import torch.nn as nn
from torch import Tensor


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_ih: weights for input -> gates (4 * hidden_size x input_size)
        # W_hh: weights for hidden -> gates (4 * hidden_size x hidden_size)
        # b_ih, b_hh: biases for input and hidden part
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x_t: Tensor, h_t: Tensor, c_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        x_t: (batch, input_size)
        h_t: (batch, hidden_size)
        c_t: (batch, hidden_size)
        """
        # Compute all gates at once: (batch, 4 * hidden_size)
        gates = (x_t @ self.W_ih.T + self.b_ih + h_t @ self.W_hh.T + self.b_hh)

        # Split gates into input, forget, cell, output
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class CustomLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.batch_first = batch_first

        # First layer takes input_size, subsequent layers take hidden_size
        layers = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            layers.append(CustomLSTMCell(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        # dropout module
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        x:  (batch, seq_len, input_size) if batch_first=True
            or (seq_len, batch, input_size) otherwise

        hx: (h_0, c_0), each (num_layers, batch, hidden_size)
            If None, they are initialized as zeros.

        returns:
            output: (batch, seq_len, hidden_size) if batch_first=True
                    (seq_len, batch, hidden_size) otherwise
            (h_n, c_n): each (num_layers, batch, hidden_size)
        """
        if not self.batch_first:
            # Convert to (batch, seq_len, input_size)
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if hx is None:
            h_t = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
            c_t = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hx
            # Split into per-layer tensors
            h_t = [h_0[layer] for layer in range(self.num_layers)]
            c_t = [c_0[layer] for layer in range(self.num_layers)]

        outputs = []

        # Iterate over time steps
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size) for first layer / (batch, hidden_size) for others

            # Pass through all layers
            for layer_idx, cell in enumerate(self.layers):
                h_t[layer_idx], c_t[layer_idx] = cell(x_t, h_t[layer_idx], c_t[layer_idx])
                x_t = h_t[layer_idx]  # output of this layer is input to the next

                # Apply dropout between layers (like PyTorch)
                if self.dropout_p > 0.0 and layer_idx < self.num_layers - 1 and self.training:
                    x_t = self.dropout(x_t)

            # x_t is now output of the last layer at time t
            outputs.append(x_t.unsqueeze(1))

        # Concatenate outputs over time
        output = torch.cat(outputs, dim=1)

        # Stack final hidden and cell states: (num_layers, batch, hidden_size)
        h_n = torch.stack(h_t, dim=0)
        c_n = torch.stack(c_t, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)  # back to (seq_len, batch, hidden_size)

        return output, (h_n, c_n)
