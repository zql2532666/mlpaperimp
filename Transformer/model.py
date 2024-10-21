import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):

        '''
        d_model: This is the dimensionality of the model's input embeddings 
        (e.g., if each token is represented by a vector of size 512, then d_model = 512).

        max_len: This is the maximum length of the input sequence that the model is expected to handle. This is set to a large value (5000 by default), 
        which means the positional encodings will be precomputed for sequences of up to 5000 tokens.
        '''

        super(PositionalEncoding, self).__init__()

        # Create a matrix of size (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)
        
        # Position indices (0, 1, 2, ..., max_len-1)
        # a column vector representing the positions of the tokens in the sequence, ranging from 0 to max_len - 1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Dividing term for the sinusoidal function
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices in the positional encoding matrix
        # start from 0 to the end, in step of 2
        pe[:, 0::2] = torch.sin(position * div_term)    
        
        # Apply cos to odd indices in the positional encoding matrix
        # start from 1 to the end, in step of 2
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra batch dimension (unsqueeze(0)) to match input shape
        pe = pe.unsqueeze(0)
        
        # Register buffer so that it's part of the model's state but not trainable
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        # Add positional encoding to input tensor x
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        # transpose over the last two dimensions
        # since K[i] indexes into each batch
        # K[batch][j] indexes into each head within that batch
        # the dot product is done for a single head
        attention_pattern_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_pattern_weights.masked_fill(mask==0, -1e9)
        
        attention = torch.softmax(attention_pattern_weights, dim=-1)
        output = torch.matmul(attention, V)

        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads
        self.self_attention = ScaledDotProductAttention(self.d_k)

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)


    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Reshape Q, K, V for multi-head attention (splitting into multiple heads)
        # Shape: (batch_size, seq_len, num_heads, d_k), before transpose
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q = self.linear_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        O, attention = self.self_attention(Q, K, V, mask)

        # Concatenate heads back together
        O = O.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        out = self.linear_o(O)
        return out, attention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncodingLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_hidden=2048, dropout=0.1):
        super(EncodingLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_hidden, dropout)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        attention_output, attention = self.self_attention(x, x, x, mask=mask)
        x = self.LN1(x + attention_output)
        ff_output = self.feed_forward(x)
        x = self.LN2(x + ff_output) 
        return x


class DecodingLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_hidden=2048, dropout=0.1):
        super(DecodingLayer, self).__init__()
        self.self_atttention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_hidden=d_hidden, dropout=dropout)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.LN3 = nn.LayerNorm(d_model)
    

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        attention_output, _ = self.self_atttention(x, x, x, tgt_mask)
        x = self.LN1(x + attention_output)

        attention_output, _ = self.cross_attention(encoder_output, encoder_output, x, src_mask)
        x = self.LN2(x + attention_output)

        ff_output = self.feed_forward(x)
        x = self.LN3(x + ff_output)

        return x
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_encoding_layers=6, n_decoding_layers=6, d_model=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncodingLayer(d_model) for _ in range(n_encoding_layers)])
        self.decoder = nn.ModuleList([DecodingLayer(d_model) for _ in range(n_decoding_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_embedding(src)
        src = self.pos_encoding(src)
        src = self.dropout(src)

        for encoding_layer in self.encoder:
            src = encoding_layer(src, src_mask)

        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(src)

        for decoding_layer in self.decoder:
            tgt = decoding_layer(tgt, src, src_mask, tgt_mask)
        
        output = self.linear(tgt)
        return output
