import sqlite3 as lite
import os
import chardet
import json
import sqlite3 as lite
from test_util_kb import look_up_mention, get_cand_info_by_mention
from test_sqlite_basic_ops import test_select_by_cursor
from util_tb import get_table_content, is_measure_col
from constants import base_dir, cache_dir, lite_db_name

test_tb_id = '3389822_6_374624044314151266'
db_name = os.path.join(cache_dir, lite_db_name)

def create_lite_tb_for_cache(tb_id, logger):
    try:
        # import ipdb; ipdb.set_trace()
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute('DROP TABLE IF EXISTS "tb_{}"'.format(tb_id))
        schema_str = 'row_id INT, col_id INT, cell_value STRING, lookup_order INT, label STRING, ' \
            + 'entity_uri STRING, clses STRING, RefCount INT, abstract STRING, comment STRING'
        cur.execute('CREATE TABLE "tb_{}" ({})'.format(tb_id, schema_str))
    except lite.Error as e:
        # import ipdb; ipdb.set_trace()
        logger.error('Error! {} Fail to create lite tb'.format(e.args[0]))
    # finally:
    #     if con:
    #         con.close()
    return con
    # need to be closed!


def check_tb_exists(tb_name):
    try:
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND tbl_name = '{}';"
                    .format(tb_name))
        rows = cur.fetchall()
    except:
        pass
    return len(rows)==1


def cache_one_table(tb_id, logger, support_breakpoint=False):
    # if already cached, skip
    if support_breakpoint and check_tb_exists('tb_' + tb_id):
        logger.info('Skip! Table {} already cached to {} ! '.format(tb_id, db_name))
        return

    logger.info('Caching table {} to {}'.format(tb_id, db_name))
    con = create_lite_tb_for_cache(tb_id, logger)
    col_names = 'row_id, col_id, cell_value, lookup_order, label, ' \
                 + 'entity_uri, clses, RefCount'
    table_cols = get_table_content(tb_id)
    for col_id, col in enumerate(table_cols):  # to +1
        logger.info('Caching column. col_id:{}, row_num:{}'.format(col_id, len(col)))
        # if is_digit_col(col):
        #     print('Digit! {}'.format(col[:5]))
        #     continue
        if is_measure_col(col):
            logger.info('Judge as measure col! {}'.format(col[:5]))
            continue
        for row_id, cell_item in enumerate(col[1:]):  # to +1
            # print(col_id, row_id, cell_item)
            # if col_id==2 and row_id==34:
            #     import ipdb; ipdb.set_trace()
            cell_item = cell_item.strip()
            if cell_item=='':
                continue
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
                logger.error('lite.Error {}'.format(e.args[0]))
                con.rollback()
    con.close()


def test_check_tb_exists():
    res1 = check_tb_exists('tb_1438042986423_95_20150728002306-00125-ip-10-236-191-2_88435628_5')
    res2 = check_tb_exists('tb_1438042986423_95_20150728002306-00125-ip-10-236-191-2_88435628_000')
    print(res1, res2)



def test_cache_one_table(tb_id):
    tb_cache_name = 'tb_'+tb_id
    test_select_by_cursor(tb_name=tb_cache_name)
    return


def test_vital(logger):
    # import ipdb; ipdb.set_trace()
    # cache_one_table('1438042989790_89_20150728002309-00310-ip-10-236-191-2_664422904_7')
    # cache_one_table('1438042989891_18_20150728002309-00079-ip-10-236-191-2_64375179_16',logger)
    cache_one_table('62564020_0_3836030043284699244', logger)

if __name__=='__main__':
    # test_check_tb_exists()
    # # evaluate_oracle_for_one_table(test_tb_id)
    from util_other import getYHLogger
    logger = getYHLogger(prefix='test')
    # # cache_one_table(test_tb_id, logger)
    # # test_cache_one_table(test_tb_id)
    test_vital(logger)