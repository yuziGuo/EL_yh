import torch
from col_spec_yh.constants import *
import random

def generate_seg(args, cols, noise_num=0, row_wise_fill=False):
    '''
    :param cols -> List[List[str]]
            ps :
            len(cols)<=98, corresponding to col_id 1..98
            because col_id 99 is left to denote [CLS]
    :param row_wise_fill -> Bool
    :return: tokens -> List[int]; seg -> List[int]
    '''
    tokens = [CLS_ID]
    seg = [9999]
    if not row_wise_fill:
        pass
        # for idx_c, col in enumerate(cols, 1):  # start from 1
        #     seg.append(100*idx_c)
        #     tokens.append(SEP_ID)
        #     for idx_r, dataframe in enumerate(col, 1):  # start from 1
        #         temp = [SEP_ID] + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
        #         tokens.extend(temp)
        #         seg.extend([idx_c*100+idx_r]*len(temp))
        # tokens = tokens[:args.seq_len]
        # seg = seg[:args.seq_len]
    elif row_wise_fill:
        dataframe_max_len = args.seq_len // len(cols)
        for idx_c in range(1, len(cols) + 1):
            idx_r = 98 # fake
            seg.append(100*idx_c+idx_r)
            tokens.append(CLS_ID)
        for idx_r in range(1, len(cols[0])+1):
            for idx_c in range(1, len(cols) + 1):
                # import ipdb; ipdb.set_trace()
                try:
                    dataframe = cols[idx_c-1][idx_r-1]
                except:
                    IndexError
                    import ipdb; ipdb.set_trace()

                temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[:dataframe_max_len-2]
                if len(temp) == 0:
                    continue
                else:
                    temp = [CLS_ID] + temp
                    tokens.extend(temp)
                    for idx_tok in range(1, len(temp)+1):
                        seg.append(idx_tok*10000 + idx_c*100 +idx_r)
        tokens = tokens[:args.seq_len]
        seg = seg[:args.seq_len]
    while len(tokens) < args.seq_len:
        tokens.append(PAD_ID)
        seg.append(0)
    for _ in range(noise_num): # two noise
        _i = random.randint(0, len(tokens)-1)
        tokens[_i] = MASK_ID
    return tokens, seg


def generate_mask(seg, mask_mode='crosswise'):
    '''
    :param seg -> torch.LongTensor          shape: [bz, seq_len]
    :return: mask -> torch.FloatTensor      shape: [bz, 1, seq_len, seq_len]
    '''
    bz, seq_len = seg.shape
    seg = seg % 10000

    # cls_see_all_mask = (seg > 0).float()  # [bz, seq_len]
    # cls_see_all_mask = cls_see_all_mask.view(bz, 1, 1, seq_len)

    seg = seg.view(bz, seq_len, 1)
    seg_2 = seg.view(bz, 1, seq_len)
    row_wise_see = (seg % 100 == seg_2 % 100).unsqueeze(1).float()  # mask: [batch_size x 1 x seq_length x seq_length]
    if mask_mode == 'row-wise':
        return row_wise_see
    col_wise_see = (seg // 100 == seg_2 // 100).unsqueeze(1).float()*2
    if mask_mode == 'col-wise':
        return col_wise_see
    if mask_mode == 'cross-wise':
        mask = row_wise_see + col_wise_see
        return mask
    hier_tab_col_see = ((seg % 100 > 90) * (seg_2 % 100 > 90)).unsqueeze(1).float() * 4
    if mask_mode == 'cross-and-hier-wise':
        mask = row_wise_see + col_wise_see + hier_tab_col_see
        return mask
    # mask = torch.cat((cls_see_all_mask, mask[:, :, 1:, :]), dim=-2)
    # then we can use 1,2,3.. (or more possible cnt numbers) to distinguish different relations -> relation-aware attention



def get_sep_idxs(seg, option):
    '''
    :param
        seg -> torch.LongTensor; shape: [bz, seq_len]
        option: in ['tab', 'col', 'cell']
    :return:
    '''
    bz, seq_len = seg.shape
    if option == 'tab':
        return torch.stack([torch.arange(0, bz), torch.zeros(bz).long()], dim=0)  # shape: [2, bz]
    if option == 'col':
        return torch.nonzero(((seg % 100) == 98).float())
    if option == 'cell':
        return torch.nonzero(((seg // 10000) == 1).float())
