import torch as t
from torch import nn

# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
# 假定输入的图像形状为[64,64,3]
input = t.randn(1,5,512)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入

print(input.shape)
output = W_Q(input) # 调用全连接层
print(output.shape)
print(output.view(1, -1, n_heads, d_k).transpose(1, 2).shape)