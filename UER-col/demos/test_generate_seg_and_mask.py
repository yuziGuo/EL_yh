from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise

from demos.samples.sample_mini_tables import table_a, table_b
from demos.utils import get_args

import torch

def check_segs(iter):
    for (seg, tokens) in iter:
        i = 0; s = 0; now = seg[s]
        while s < len(seg):
            while i < len(seg) and seg[i] == now: i += 1
            print(seg[s:i])
            print(args.tokenizer.convert_ids_to_tokens(tokens[s:i]))
            s = i
            if s < len(seg):
                now = seg[s]


def test_1():
    args = get_args()
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=True)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    mask = generate_mask_crosswise(seg)
    # import ipdb; ipdb.set_trace()


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
    test_2()



