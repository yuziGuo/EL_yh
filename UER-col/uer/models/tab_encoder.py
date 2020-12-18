# -*- encoding:utf-8 -*-
# author : Guo Yuhe

import torch
import torch.nn as nn

from uer.layers.embeddings import BertEmbedding, WordEmbedding
from uer.encoders.bert_encoder import BertEncoder

class TabEncoder(nn.Module):
    def __init__(self, args):
        super(TabEncoder, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        self.encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
        self.pooling = args.pooling
        # self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, src, mask):
        import ipdb; ipdb.set_trace()
        emb = self.embedding(src, mask)  # [8, 64, 768]
        output = self.encoder(emb, mask)  # # [8, 64, 768]

        # Target.
        if self.pooling == "mean":
            from torch_scatter import scatter_mean, scatter_max
            ex_mask = (mask // 100).unsqueeze(-1).expand(-1, -1, output.size(-1))
            _ = scatter_mean(src=output, index=ex_mask, dim=-2)  # [batch_size, col_num+1, 768]
            output = _[:,1:,:]  # columns
        elif self.pooling == "max":
            from torch_scatter import scatter_mean, scatter_max
            ex_mask = (mask // 100).unsqueeze(-1).expand(-1, -1, output.size(-1))
            _ = scatter_max(src=output, index=ex_mask, dim=-2)[0]
            output = _[:,1:,:]
        elif self.pooling == "last":
            output = output[:, -1, :]
        elif self.pooling == 'bert':
            output = output[:, 0, :]
        elif self.pooling == 'crosswise-bert':
            mask = (mask % 100 == 1).float() * mask
            _t = torch.FloatTensor([-10000]).repeat(mask.size()[0]).unsqueeze(-1).to(mask.device)
            _for_calc = torch.cat((_t, mask[:, :-1]), 1)
            _idxs = torch.nonzero(((mask - _for_calc)>0).float()).T
            try:
                output = output[_idxs[0], _idxs[1], :].reshape(mask.shape[0], -1, emb.shape[-1])  # [bz,col_num,768]
            except Exception as e:
                print(e)
        print(output.shape)
        # output = torch.tanh(self.output_layer_1(output))  # output: [batch_size, emb_size]  #
        return output