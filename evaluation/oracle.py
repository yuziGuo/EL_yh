import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from util_kb import *
from util_other import getYHLogger
from constants import base_dir, cache_dir, lite_db_name

import os
db_name = os.path.join(cache_dir, lite_db_name)

from tqdm import tqdm
import sqlite3 as lite
import pandas as pd
test_tb = '62564020_0_3836030043284699244'


def _get_lookup_res_df(tb_id):
    con = None
    try:
        con = lite.connect(db_name)
        lookup_df = pd.read_sql_query("SELECT * from 'tb_{}'".format(tb_id), con)
        # import ipdb; ipdb.set_trace()
    except lite.Error as e:
        print(e.args[0])
        return -1
    except KeyboardInterrupt:
        if con:
            con.close()
        return -1
    finally:
        if con:
            con.close()
        return lookup_df


def _get_gold_df(tb_id):
    try:
        fn = os.path.join(base_dir, 'instance', tb_id+'.csv')
        df = pd.read_csv(fn, quotechar='"', header=None, names=['entity_uri', 'key_mention_value', 'row_id'])
    except:
        print('Error!', tb_id)
    return df


def _eval_one_table(tb_id):
    lookup_df = _get_lookup_res_df(tb_id)[['row_id', 'entity_uri']].drop_duplicates()
    gold_df = _get_gold_df(tb_id)
    true_pos = len(pd.merge(gold_df, lookup_df))
    false_pos = len(lookup_df['row_id'].drop_duplicates()) - true_pos
    false_neg = len(gold_df) - true_pos
    return true_pos, false_neg, false_pos


def evaluate():
    logger = getYHLogger(prefix='eval_oracle')

    total_true_pos, total_false_neg, total_false_pos = 0., 0., 0.
    d = os.path.join(base_dir, 'instance')
    test_table_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(d)))

    for _ in tqdm(test_table_list):
        true_pos = _eval_one_table(_)[0]
        false_neg = _eval_one_table(_)[1]
        false_pos = _eval_one_table(_)[2]
        logger.info('tb:{}, TP:{}, FN:{}, FP:{}'.format(_, true_pos, false_neg, false_pos))
        total_true_pos += true_pos
        total_false_neg += false_neg
        total_false_pos += false_pos
    logger.info('Total: TP:{}, FN:{}, FP:{}'.format(total_true_pos, total_false_neg, total_false_pos))
    precision = total_true_pos / (total_true_pos + total_false_pos)
    recall = total_true_pos / (total_true_pos + total_false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    logger.info('Oracel Evaluation Result: P:{}, R:{}, F1:{}'.format(
        precision, recall, f1))
    # res: 71, 75, 73 (make sense!)
    # turl result on other wikiGS, using wikidata lookup: 88, 64, 74


if __name__=='__main__':
    evaluate()
