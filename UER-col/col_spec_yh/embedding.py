# -*- encoding:utf-8 -*-
# author : Guo Yuhe
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class TabEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(TabEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(seg // 10000)
        emb = word_emb + pos_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb