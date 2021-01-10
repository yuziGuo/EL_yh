from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise

from demos.samples.sample_mini_tables import table_a, table_b, table_with_empty_values_1, table_with_empty_values_2
from demos.utils import get_args

import torch


def check_segs_general(tokenizer, iter, seg_idx, tokens_idx):
    for _ in iter:
        seg = _[seg_idx]
        tokens = _[tokens_idx]
        # seg = seg % 10000
        i = 0; s = 0; now = seg[s] % 10000
        import ipdb; ipdb.set_trace()
        while s < len(seg):
            while i < len(seg) and seg[i]%10000 == now: i += 1
            print(seg[s:i])
            print(tokenizer.convert_ids_to_tokens(tokens[s:i]))
            s = i
            if s < len(seg):
                now = seg[s]%10000


def check_segs(iter):
    args = get_args()
    for (seg, tokens) in iter:
        # seg = seg % 10000
        i = 0; s = 0; now = seg[s] % 10000
        while s < len(seg):
            while i < len(seg) and seg[i]%10000 == now: i += 1
            print(seg[s:i])
            print(args.tokenizer.convert_ids_to_tokens(tokens[s:i]))
            s = i
            if s < len(seg):
                now = seg[s]%10000


def test_1():
    args = get_args()
    args.seq_len = 128
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=True)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    mask = generate_mask_crosswise(seg)
    import ipdb; ipdb.set_trace()


def test_3():
    args = get_args()
    args.seq_len = 16
    tokens_0, seg_0 = generate_seg(args, table_with_empty_values_1, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg(args, table_with_empty_values_2, row_wise_fill=True)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    mask = generate_mask_crosswise(seg)
    import ipdb; ipdb.set_trace()

def test_2():
    from col_spec_yh.store_utils import test_decode_spider_file
    tab_file = 'demos/samples/sample_file_type0-1.tb'
    tab_cols_list = test_decode_spider_file(tab_file)

    args = get_args()
    seg_list = []
    for tab_col in tab_cols_list:
        _, seg = generate_seg(args, tab_col, row_wise_fill=True)
        seg_list.append(seg)
    seg = torch.LongTensor(seg_list)
    mask = generate_mask_crosswise(seg)  # mask.shape: torch.Size([10, 1, 64, 64])
    import ipdb; ipdb.set_trace()

if __name__=='__main__':
    test_3()
    test_1()
    test_2()



