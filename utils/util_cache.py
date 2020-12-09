import sqlite3 as lite
db_name = 'test.db'

def create_lite_tb_for_cache(tb_id):
    try:
        # import ipdb; ipdb.set_trace()
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute('DROP TABLE IF EXISTS "tb_{}"'.format(tb_id))
        schema_str = 'row_id INT, col_id INT, cell_value STRING, lookup_order INT, label STRING, ' \
            + 'entity_uri STRING, clses STRING, RefCount INT, abstract STRING, comment STRING'
        cur.execute('CREATE TABLE "tb_{}" ({})'.format(tb_id, schema_str))
    except lite.Error as e:
        import ipdb; ipdb.set_trace()
        print('Error! {}'.format(e.args[0]))
    # finally:
    #     if con:
    #         con.close()
    return con
    # need to be closed!

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