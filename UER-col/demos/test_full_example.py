from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise
from col_spec_yh.model import TabEncoder

from demos.samples.sample_mini_tables import table_a, table_b
from demos.utils import get_args
from demos.utils import load_or_initialize_parameters

import torch
from col_spec_yh.store_utils import test_decode_spider_file


args = get_args()

# data
tab_file = 'demos/samples/sample_file_type0-1.tb'
tab_cols_list = test_decode_spider_file(tab_file)
args = get_args()
seg_list = []
src_list = []
for tab_col in tab_cols_list:
    tokens, seg = generate_seg(args, tab_col, row_wise_fill=True)
    seg_list.append(seg)
    src_list.append(tokens)
seg = torch.LongTensor(seg_list)
src = torch.LongTensor(src_list)
mask = generate_mask_crosswise(seg)  # mask.shape: torch.Size([10, 1, 64, 64])

# model
# args.pooling = 'avg'
# args.pooling = 'cls'
# args.pooling = 'avg-col-seg'
args.pooling = 'avg-cell-seg'
# args.pooling = 'first-cell'
ta_encoder = TabEncoder(args)
load_or_initialize_parameters(args, ta_encoder)

# encode
_ = ta_encoder.encode(src, seg, option='columns')
_ = ta_encoder.encode(src, seg, option='tables', pooling='avg-cell-seg')
