import sys
sys.path.append('./col_spec_yh')

from constants import *

def decode_and_verify_spider_file(data_path):
    tab_cols_list = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n') #
            _cols = line.split(ROW_SEP)
            _cols = [_.split(COL_SEP) for _ in _cols]
            # ipdb.set_trace()
            try:
                assert [len(_) for _ in _cols] == [len(_cols[0])] * len(_cols)
                # print(_cols)
                tab_cols_list.append(_cols)
            except:
                AssertionError
                ipdb.set_trace()
                print(_cols)
                ipdb.set_trace()
    print('Checked! ' + data_path)
    return tab_cols_list


def decode_and_verify_aida_file(data_path):
    tab_cols_list = []
    label_list = []
    raw_tab_id_list = []
    with open(data_path) as f:
        for idx, csv_line in enumerate(f):
            csv_line = csv_line.rstrip('\n') #
            [raw_tab_info, cls_name, _main_col, _sur_cols] = csv_line.split(CSV_SEP)
            raw_tab_id_list.append(':'.join(raw_tab_info.split(':')[:-1]))
            label_list.append(cls_name)
            _main_col = _main_col.split(COL_SEP)
            _sur_cols = _sur_cols.split(ROW_SEP)
            _sur_cols = [_.split(COL_SEP) for _ in _sur_cols]
            _cols = [_main_col] + _sur_cols
            # ipdb.set_trace()
            try:
                assert [len(_) for _ in _cols] == [len(_cols[0])] * len(_cols)
                # print(_cols)
                tab_cols_list.append(_cols)
            except:
                AssertionError
                ipdb.set_trace()
                print(_cols)
                ipdb.set_trace()
    print('Checked! DataPath: {}. Lines (#micro tables): {}'.format(data_path, len(label_list)))
    return raw_tab_id_list, label_list, tab_cols_list


def get_labels_map_from_aida_file(data_path):
    cls_names = []
    with open(data_path) as f:
        for idx, csv_line in enumerate(f):
            csv_line = csv_line.rstrip('\n')
            cls_name = csv_line.split(CSV_SEP)[1]
            cls_names.append(cls_name)
    labels_map = {k: i for i, k in enumerate(list(set(cls_names)))}
    return labels_map