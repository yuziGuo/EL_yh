import os
import chardet
import json
import sqlite3 as lite
from test_util_kb import look_up_mention, get_cand_info_by_mention
from test_sqlite_basic_ops import test_select_by_cursor
from util_tb import get_table_content, is_measure_col
from util_cache import create_lite_tb_for_cache
from constants import base_dir
test_tb_id = '3389822_6_374624044314151266'

# def select_rows_by_cursor(con, tb_name='cars'):
#     try:
#         cur = con.cursor()
#         cur.execute('SELECT * FROM {}'.format(tb_name))
#         rows = cur.fetchall()
#         for row in rows:
#             print(row)
#     except lite.Error as e:
#         print('Errors: {}'.format(e.args[0]))
#         sys.exit(1)


def cache_one_table(tb_id):
    print(tb_id)
    con = create_lite_tb_for_cache(tb_id)
    col_names = 'row_id, col_id, cell_value, lookup_order, label, ' \
                 + 'entity_uri, clses, RefCount'
    table_cols = get_table_content(tb_id)
    for col_id, col in enumerate(table_cols):  # to +1
        # if is_digit_col(col):
        #     print('Digit! {}'.format(col[:5]))
        #     continue
        if is_measure_col(col):
            print(tb_id, col_id)
            print('Quantity measurement! {}'.format(col[:5]))
            continue
        for row_id, cell_item in enumerate(col[1:]):  # to +1
            # print(col_id, row_id, cell_item)
            cand_set = get_cand_info_by_mention(row_id+1, col_id, cell_item)
            # import ipdb; ipdb.set_trace()
            if cand_set is None:
                continue
            try:
                cur = con.cursor()
                cur.executemany('INSERT INTO "tb_{}" ({}) VALUES(?,?,?,?, ?,?,?,?)'.format(tb_id, col_names), cand_set)
                con.commit()
            except KeyboardInterrupt:
                con.rollback()  # 会撤销当前这个 cell mention 的 cand set
                sys.exit()
            except lite.Error as e:
                print('Error! {}'.format(e.args[0]))
                con.rollback()
    con.close()

def TODO_evaluate_oracle_for_one_table(tb_id):
    import pandas as pd
    tb_cache_name = 'tb_' + tb_id
    gold_fn  = os.path.join(base_dir, 'instance', tb_id+'.csv')
    gold_df = pd.read_csv(gold_fn, header=None,
                            names=['ResourceURI', 'KeyValue', 'RowId']).sort_values(by='RowId')

    import ipdb; ipdb.set_trace()
    pass

def test_cache_one_table(tb_id):
    tb_cache_name = 'tb_'+tb_id
    test_select_by_cursor(tb_name=tb_cache_name)
    return


def test_vital():
    import ipdb; ipdb.set_trace()
    cache_one_table('1438042989790_89_20150728002309-00310-ip-10-236-191-2_664422904_7')

if __name__=='__main__':
    # evaluate_oracle_for_one_table(test_tb_id)
    # cache_one_table(test_tb_id)
    # test_cache_one_table(test_tb_id)
    test_vital()
