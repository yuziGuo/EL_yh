from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.store_utils import test_decode_spider_file

tab_file = 'demos/samples/sample_file_type0-1.tb'
tab_cols_list = test_decode_spider_file(tab_file)
print(tab_cols_list)