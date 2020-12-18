import sys
sys.path.append('./col_spec_yh')

from constants import *

def test_decode_spider_file(data_path):
    tab_cols_list = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n') #
            _cols = line.split(ROW_SEP)
            _cols = [_.split(COL_SEP) for _ in _cols]
            # ipdb.set_trace()
            try:
                assert [len(_) for _ in _cols] == [len(_cols[0])] * len(_cols)
                print(_cols)
                tab_cols_list.append(_cols)
            except:
                AssertionError
                ipdb.set_trace()
                print(_cols)
                ipdb.set_trace()
    print('Checked! ' + data_path)
    return tab_cols_list