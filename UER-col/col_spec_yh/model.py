# -*- encoding:utf-8 -*-
# author : Guo Yuhe

import torch
import torch.nn as nn

from uer.encoders.bert_encoder import BertEncoder
from torch_scatter import scatter_mean

from col_spec_yh.embedding import TabEmbedding
from col_spec_yh.encode_utils import get_sep_idxs

class TabEncoder(nn.Module):
    def __init__(self, args):
        super(TabEncoder, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        self.encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
        self.pooling = args.pooling
        # self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)


    def forward(self, src, mask):
        emb = self.embedding(src, mask)  # [8, 64, 768]
        output = self.encoder(emb, mask)  # # [8, 64, 768]
        return output


    def encode(self, src, seg, option, pooling=None): # option: ['table', 'cols', 'rows', 'cells']
        if pooling:
            self.pooling = pooling
        output = self.forward(src, seg)
        bz, seq_len, emb_size = output.shape
        if option == 'table':
            # Expected out tensor shape: [bz, emb_size]
            if self.pooling == 'avg':
                index=torch.nonzero(seg).T
                embs_table = scatter_mean(src=output[index[0], index[1], :], index=index[0], dim=-2)
                # todo: check: should avg pooling including [cls] and (many added) [sep] ?
            if self.pooling == 'cls':
                index = get_sep_idxs(seg, 'tab')
                embs_table = output[index[0], index[1], :]
            if self.pooling == 'avg-col-seg':
                index = get_sep_idxs(seg, 'col').T
                embs_table = scatter_mean(src=output[index[0], index[1], :], index=index[0], dim=-2)

            if self.pooling == 'avg-cell-seg':
                index = get_sep_idxs(seg, 'cell').T
                embs_table = scatter_mean(src=output[index[0], index[1], :], index=index[0], dim=-2)
                
            assert embs_table.shape == (bz, embs_size)
            return embs_table

        if option == 'columns':
            col_num = torch.max(seg%10000//100).item()
            # Expected out tensor shape: [bz, col_num, emb_size]
            if self.pooling == 'seg':
                index = get_sep_idxs(seg, 'col').T
                emb_cols = output[index[0], index[1], :]
                emb_cols.b.view(bz, -1,emb_size)

            if self.pooling == 'avg-cell-seg':
                index = get_sep_idxs(seg, 'cell').T
                col_ptr = ((seg%10000)//100)[index[0], index[1]]  # [1,2,3,4,5,1,2,3,4,5,1,1,2,3,4,5,...]
                tb_ptr = index[0]  # start from 0 [0 0 0  1 1 1 1]
                # output[index[0], index[1], :].view(bz, _, col_num, emb_size)  # bz, row_num, col_num, shape
                emb_cols = scatter_mean(src=output[index[0], index[1], :], index=tb_ptr*col_num+col_ptr-1, dim=-2)
                emb_cols = emb_cols.view(bz, col_num, emb_size)

            if self.pooling == 'first-cell':
                index = torch.nonzero( ((seg//10000)==1).float() * ((seg%100)==1).float() ).T  # bz*col_num, 2
                emb_cols = output[index[0], index[1], :]

            assert embs_cols.shape == (bz, col_num, embs_size)
            return emb_cols

        # if option == 'cells':
        #     if self.pooling == 'sep':
        #         pass
        #     if self.pooling == 'avg':
        #         pass

