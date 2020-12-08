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