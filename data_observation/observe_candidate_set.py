import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from constants import cache_dir, lite_db_name, base_dir

import os
db_name = os.path.join(cache_dir, lite_db_name)

import sqlite3 as lite

def cand_count_all():
    tb_list = os.listdir(os.path.join(base_dir, 'instance'))
    tb_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), tb_list))

    con = lite.connect(db_name)
    cur = con.cursor()
    total_cand_num = 0
    for tb_name in tb_list:
        cur.execute("SELECT COUNT(*) FROM 'tb_{}'".format(tb_name))
        total_cand_num += cur.fetchone()[0]
    print("Total candidate num: {}".format(total_cand_num)) # 1110441

    total_cand_num_no_dup = 0
    for tb_name in tb_list:
        cur.execute("SELECT COUNT(*) FROM (SELECT DISTINCT entity_uri FROM 'tb_{}')".format(tb_name))
        total_cand_num_no_dup += cur.fetchone()[0]
    print("Total candidate num (no duplicate): {}".format(total_cand_num_no_dup))  # 629678

    entity_set = set()
    for tb_name in tb_list:
        cur.execute("SELECT DISTINCT entity_uri FROM 'tb_{}'".format(tb_name))
        for u in cur.fetchall():
            u = u[0] if type(u) == tuple else u
            entity_set.add(u)
    print("Total candidate num (no duplicate overall): {}".format(len(entity_set)))  # 189077
    con.close()


if __name__=='__main__':
    cand_count_all()
    # check_cand_valid()

