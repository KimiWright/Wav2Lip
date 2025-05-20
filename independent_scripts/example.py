import torch
import torch.nn as nn

# New input size after concatenation: 184 + 1 + 1 + 1 = 187
input_size = 187
hidden_size = 32
num_layers = 2
bidirectional = False

batch_size = 1

# Dummy data: (seq_len=5, batch_size=1, input_size=187)
# If you only have one sequence, wrap it as batch_size=1
x = torch.randn(5, batch_size, input_size)  # shape = (seq_len, batch, input_size)

# Define the GRU
gru = nn.GRU(input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             bidirectional=bidirectional)

# Forward pass
output, h_n = gru(x)

# Output shapes
print("Output shape:", output.shape)  # (5, 1, 32)
print("Last hidden state shape:", h_n.shape)  # (2, 1, 32) for 2 layers

size = torch.Size([64, 5, 512])
seq_len = size[1]
print(seq_len)
x = torch.randn(64, 512, 1, 1)  # Shape: [64, 512]
x = x.squeeze().unsqueeze(1).expand(-1, seq_len, -1)#x.view(64, 512)            # Shape: [64, 512]

# # Add time dimension and expand across it
# x = x.unsqueeze(1)             # Shape: [64, 1, 512]
# x = x.expand(-1, 5, -1) 
print(x.shape)  # Shape: [64, 5, 512]
