import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        N, C, L = q.shape

        q = self.query(q).view(N, C // self.num_heads, self.num_heads, L).transpose(1, 2)
        k = self.key(k).view(N, C // self.num_heads, self.num_heads, L).transpose(1, 2)
        v = self.value(v).view(N, C // self.num_heads, self.num_heads, L).transpose(1, 2)

        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, C, L)

        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_hidden_dim, vocab_size, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(embed_dim, vocab_size)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_len, tgt_len, batch_size = src.shape[1], tgt.shape[1], src.shape[0]

        # Add positional encoding to the source and target sequences
        src = self.embedding(src) + self.positional_encoding[:src_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt_len, :].unsqueeze(0).repeat(batch_size, 1, 1)

        # Apply dropout
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask)

        # Output
        output = self.out(tgt)
        return output


# Example usage
if __name__ == "__main__":
    batch_size = 2
    src_len = 10
    tgt_len = 12
    vocab_size = 1000
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ffn_hidden_dim = 2048
    dropout = 0.1

    model = Transformer(num_layers, embed_dim, num_heads, ffn_hidden_dim, vocab_size, max_len=max(src_len, tgt_len),
                        dropout=dropout)

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(src, tgt)
    print(output.shape)  # Should be (batch_size, tgt_len, vocab_size)