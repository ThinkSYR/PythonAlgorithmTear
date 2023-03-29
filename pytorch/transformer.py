"""Refer to Attention ia all you need
Transformer:
Encoder:
    (input + PE) -> (self-MHA - add&norm - FFN - add&norm) * N
Decoder:
    (output-shifted + PE) -> (self-MMHA - add&norm - MHA - add&norm - FFN - add&norm) * N
"""
import torch 
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """PE
    PE(pos, 2i) = sin(pos / ( 1e4 ^ (2i / d_model) ) )
    PE(pos, 2i + 1) = cos(pos / ( 1e4 ^ (2i / d_model) ) )
    """
    def __init__(self, max_seq_len=512, hidden_size=768) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = hidden_size

    def _test_pe(self):
        import matplotlib.pyplot as plt
        pe = PositionalEncoding(100, 20)
        pe = pe()
        plt.plot(np.arange(100), pe[:, 4:8])
        plt.show()

    def forward(self):
        pos = torch.arange(self.max_seq_len).unsqueeze(1)
        # exp( - 2i / d_m * log1e4)
        state = torch.arange(self.d_model)
        state = torch.exp( - state / self.d_model * np.log(1e4) )[0::2]
        encodings = torch.zeros(self.max_seq_len, self.d_model)
        encodings[:, 0::2] = torch.sin(pos * state)
        encodings[:, 1::2] = torch.cos(pos * state)
        return encodings

class FeedForwardNet(nn.Module):
    """FFN
    max(x, xW1 + b1)W2 + b2
    """
    def __init__(self, hidden_size=768) -> None:
        super().__init__()
        self.linear1    = nn.Linear(hidden_size, hidden_size, True)
        self.relu       = nn.ReLU()
        self.linear2    = nn.Linear(hidden_size, hidden_size, True)
        self.dropout    = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    """Attention
    Attn(Q, K, V) = softmax( QK^T / gd_k) * V
    """
    def __init__(self, hidden_size=768, h=12):
        super().__init__()
        self.q_W = nn.Linear(hidden_size, hidden_size, False)
        self.k_W = nn.Linear(hidden_size, hidden_size, False)
        self.v_W = nn.Linear(hidden_size, hidden_size, False)
        self.h   = h
        self.d_k = hidden_size // h
        self.hs  = hidden_size
        self.softmax = nn.Softmax(dim=3) # to key seq
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None):
        # (bs, 512, 768)
        bs, seq_len, hs = q.size()
        q = self.q_W(q).view(bs, -1, self.h, self.d_k).permute(0,2,1,3)
        k = self.k_W(k).view(bs, -1, self.h, self.d_k).permute(0,2,3,1)
        v = self.v_W(v).view(bs, -1, self.h, self.d_k).permute(0,2,1,3)
        # bs, h, seq_len, d_k
        score = torch.matmul(q, k) / np.sqrt(self.d_k)
        # mask
        if mask is not None:
            score.masked_fill(mask == 0, float('-inf')) # e^-inf = 0
        score = self.softmax(score) 
        # bs, h, seq_len_q, seq_len_k
        score = self.dropout(score)
        # bs, h, seq_len, d_k
        v = torch.matmul(score, v)
        # bs, seq_len, h, d_k
        bs, seq_len = v.size(0), v.size(2)
        v = v.permute(0, 2, 1, 3).view(bs, seq_len, -1)
        return v


class Transformer(nn.Module):
    def __init__(self, num_layer):
        super().__init__()
        self.num_layer = num_layer
    
    def forward(self):
        pass 






