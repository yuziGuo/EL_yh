import os
import json
import chardet
from tqdm import tqdm
import logging
import pandas as pd
import langdetect

import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from util_tb import get_table_desc, get_table_content, webtable2dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关  # 脚本中没有配置logger.setLevel会使用handler.setLevel
logfile = 'log'
fh = logging.FileHandler(logfile, mode='w')
# fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
'''
<base_dir>/data/t2d
    - ./tables/             # 779
    - ./instance/           # 503
    - ./property/           # 502
    - ./classes_GS.csv      # 237 lines
'''
base_dir = '/home/gyh/pack_for_debug_2/EL_yh/data/t2d'

'''
"1438042986423_95_20150728002306-00125-ip-10-236-191-2_88435628_5.tar.gz","Political Party","http://dbpedia_org/ontology/PoliticalParty"
"1438042986423_95_20150728002306-00329-ip-10-236-191-2_805336391_10.tar.gz","swimmer","http://dbpedia_org/ontology/Swimmer"
"1438042989018_40_20150728002309-00067-ip-10-236-191-2_57714692_2.tar.gz","cricketer","http://dbpedia_org/ontology/Cricketer"
"1438042989043_35_20150728002309-00287-ip-10-236-191-2_875026214_2.tar.gz","mountain","http://dbpedia_org/ontology/Mountain"
"68779923_2_1000046510804975562.tar.gz","Country","http://dbpedia.org/ontology/Country"
'''

def cnt_v_in_list(v, l):
    return len(list(filter(lambda x: x==v, l)))

def get_table_list(table_dir='tables'):
    # import ipdb; ipdb.set_trace()
    table_dir = os.path.join(base_dir, table_dir)
    # table_list = list(map(lambda x: x.split('.')[0], os.listdir(table_dir))) # It's wrong!
    table_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(table_dir))) # It's wrong!\
    return table_list

def check_classes_GS_in_table_list(fn='classes_GS.csv'):
    fn = os.path.join(base_dir, fn)
    tb_fn_idxs = []
    with open(fn, 'r') as f:
        tb_fn_idxs.extend(map(lambda x: x.split(',')[0][1:-1].split('.')[0], f.readlines()))
    tab_lst = get_table_list()
    not_in_tab_list = list(filter(lambda x: x not in tab_lst, tb_fn_idxs))
    assert len(not_in_tab_list) == 0

def check_property_gs_in_table_list(d='property'):
    d = os.path.join(base_dir, d)
    tb_fn_idxs = list(map(lambda x: x.split('.')[0], os.listdir(d)))
    tab_lst = get_table_list()
    not_in_tab_list = list(filter(lambda x: x not in tab_lst, tb_fn_idxs))
    assert len(not_in_tab_list) == 1

    res = os.popen('wc -l  {}/*'.format(d)).readlines()
    res = [tuple(_.strip().split()) for _ in res]
    print("{} out of {} property gold annotation files are emtpy.".format(
        len(list(filter(lambda x: x[0]=='0', res))), len(res)))

def check_instance_gs_in_table_list(d='instance'):
    d = os.path.join(base_dir, d)
    tb_fn_idxs = list(map(lambda x: x.split('.')[0], os.listdir(d)))
    tab_lst = get_table_list()
    not_in_tab_list = list(filter(lambda x: x not in tab_lst, tb_fn_idxs))
    # import ipdb; ipdb.set_trace()
    assert len(not_in_tab_list) == 2
    '''
    I deleted these two annotation files, which are not in tables:
        1438042986423_95_20150728002306-00014-ip-10-236-191-2_388949232_4.json
        1438042986423_95_20150728002306-00006-ip-10-236-191-2_877245825_2.json
    '''

    res = os.popen('wc -l  {}/*'.format(d)).readlines()
    res = [tuple(_.strip().split()) for _ in res]
    print("{} out of {} instance gold annotation files are emtpy.".format(
        len(list(filter(lambda x: x[0]=='0', res))), len(res)))


'''
这一步检查，目的是发现 instance level 的标注里，有哪些来自非horizontal陈列的表格.
最终发现有四个：
    # /home/gyh/pack_for_debug_2/EL_yh/data/t2d/instance/41648740_0_6959523681065295632.csv     mixed
    # /home/gyh/pack_for_debug_2/EL_yh/data/t2d/instance/28494901_6_7026744149694237309.csv     vertical
    # /home/gyh/pack_for_debug_2/EL_yh/data/t2d/instance/44206774_0_3810538885942465703.csv     mixed
    # /home/gyh/pack_for_debug_2/EL_yh/data/t2d/instance/79966524_0_1484492112455633784.csv     horizonal（打错字）

'''
def check_instance_annotation_on_non_horizontal_table(d='instance'):
    non_hori_tb_list = []
    with open('log_orientation', 'r') as f:
        non_hori_tb_list.extend([os.path.basename(_.strip().split()[-3])[:-5] for _ in f.readlines()])

    d = os.path.join(base_dir, d)
    tb_fn_idxs = list(map(lambda x: x.split('.')[0], os.listdir(d)))
    gold_in_non_hori = list(filter(lambda x:x in non_hori_tb_list, tb_fn_idxs))

    for _ in gold_in_non_hori:
        res = os.popen('wc -l  {}'.format(os.path.join(d, _+'.csv'))).readlines()[0]
        print(res)

    import ipdb; ipdb.set_trace()



def decode_one_table(fn='3389822_6_374624044314151266.json', check_orientation=False):
    if not fn.startswith('/home'):
        fn = os.path.join(base_dir, 'tables', fn)
    line = ''

    with open(fn, 'rb') as f:
        # 1. charset can be resolved
        line_b = f.readlines()[0]
        chardet_info = chardet.detect(line_b)
        try:
            assert chardet_info['confidence'] > 0.65
            line += line_b.decode(chardet_info['encoding'])
        except:
            AssertionError
            print('Unrecognized encoding schema', fn)

        # 2. json can be loaded
        try:
            t = json.loads(line)
            # print(chardet_info)
        except:
            JSONDecodeError
            print('sss')
            logger.error('Json error when decoding table {}'.format(fn))

    if check_orientation:
        # 3
        try:
            assert t['tableOrientation'] == 'HORIZONTAL'
        except:
            AssertionError
            logger.warning('Orientation of table {} is {}'.format(fn, t['tableOrientation']))
    # import ipdb; ipdb.set_trace()
    return t['relation']
    # assert t['tableOrientation'] == 'HORIZONTAL'
    # import ipdb; ipdb.set_trace()

def decode_all_tables():
    table_dir = os.path.join(base_dir, 'tables')
    table_list = os.listdir(table_dir)
    for tb in tqdm(table_list):
        decode_one_table(tb)
        # print('tb')

def check_text_property_for_one_webtable(fn='3389822_6_374624044314151266.json'):
    desc = get_table_desc(fn)
    rec = {k: len(v.strip())!=0 for k,v in desc.items()}
    return rec

def check_text_property_for_all_webtables():
    table_dir = os.path.join(base_dir, 'tables')
    table_list = os.listdir(table_dir)
    prop_rec_list = []
    for tb in tqdm(table_list):
        prop_rec_list.append(check_property_for_one_webtable(tb))
    df = pd.DataFrame(prop_rec_list)
    import ipdb; ipdb.set_trace()
    assert len(prop_rec_list) == 779
    assert cnt_v_in_list(False, df['pageTitle'].tolist()) == 14
    assert cnt_v_in_list(False, df['textAfterTable'].tolist()) == 35
    assert cnt_v_in_list(False, df['textBeforeTable'].tolist()) == 14
    assert cnt_v_in_list(False, df['title'].tolist()) == 748
    df_tem = df[(df['textBeforeTable'] == False) & (df['pageTitle'] == False) & (df['textAfterTable'] == False)]
    assert len(df_tem) == 0
    df_tem = df[(df['textBeforeTable'] == False) & (df['textAfterTable'] == False)]
    assert len(df_tem) == 3
    # priority: textAfterText or textBeforeText; title; pageTitle


def check_raw_language_for_all_table():
    def get_lang_for_data_row(info_dict):
        priority = ['textAfterTable', 'textBeforeTable', 'title', 'pageTitle']
        for k in priority:
            s = info_dict.get(k).strip()
            if s != '':
                try:
                    res = langdetect.detect_langs(s)[0]
                    return res.lang, res.prob
                except langdetect.lang_detect_exception.LangDetectException as e:
                    print('LangDetectException {}'.format(e.args[0]))
                    print(s)

    table_dir = os.path.join(base_dir, 'tables')
    table_list = os.listdir(table_dir)
    desc_list = []
    for tb in tqdm(table_list):
        desc_list.append(get_table_desc(tb))
    df = pd.DataFrame(desc_list)
    langs = [get_lang_for_data_row(dict(df.iloc[row_id])) for row_id in range(len(df))]
    # import ipdb; ipdb.set_trace()
    from collections import Counter
    assert Counter([_[0] for _ in langs]) == Counter({'en': 708, 'de': 32, 'fr': 8, 'pl': 5, 'ca': 5, 'es': 2, 'sv': 2, 'nl': 2, 'tr': 2, 'af': 2, 'pt': 2, 'th': 1, 'no': 1, 'ru': 1, 'cy': 1, 'ko': 1, 'zh-cn': 1, 'ro': 1, 'ar': 1, 'id': 1})
    # pass

def check_hasKeyColumn_for_all_tables():
    table_dir = os.path.join(base_dir, 'tables')
    table_list = os.listdir(table_dir)
    info_list = []
    for tb in tqdm(table_list):
        info_list.append(webtable2dict(tb)['hasKeyColumn'])
    from collections import Counter
    assert Counter(info_list) == Counter({True: 586, False: 193})

    d = os.path.join(base_dir, 'instance')
    table_list_2 = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(d)))
    info_list_2 = []
    for tb in tqdm(table_list_2):
        info_list_2.append(webtable2dict(tb)['hasKeyColumn'])
    assert Counter(info_list_2) == Counter({True: 418, False: 83})


'''
    确保都可以用 pandas.read_csv() 读出
'''
def decode_annotation_for_entity():
    d = os.path.join(base_dir, 'instance')
    for fn in os.listdir(d):
        res = os.popen('wc -l  {}'.format(os.path.join(d, fn))).readlines()[0]
        if res.strip().split()[0] == '0':
            continue
        # annotation_for_tb = pd.read_csv(os.path.join(d, fn), quotechar='"')
        try:
            annotation_for_tb = pd.read_csv(os.path.join(d, fn), quotechar='"')
        except:
            # ParserError
            import ipdb; ipdb.set_trace()
            print('Error!', fn)


'''
    检查发现都已经整理成t['relation'][idx]代表一列的形式
    不像原始的webtables那般混乱
'''
def compare_layout():
    horizontal = '/home/gyh/pack_for_debug_2/EL_yh/data/t2d/tables/79966524_0_1484492112455633784.json'
    vertical = '/home/gyh/pack_for_debug_2/EL_yh/data/t2d/tables/28494901_6_7026744149694237309.json'
    mixed = '/home/gyh/pack_for_debug_2/EL_yh/data/t2d/tables/44206774_0_3810538885942465703.json'
    h_table = decode_one_table(os.path.basename(horizontal))
    v_table = decode_one_table(os.path.basename(vertical))
    m_table = decode_one_table(os.path.basename(mixed))
    import ipdb; ipdb.set_trace()


'''
                               DBpedia resource URI            Key value  Row index
    0              http://dbpedia.org/resource/Crab                 crab         43
    1  http://dbpedia.org/resource/Green_sea_turtle  green    sea turtle         88
    2        http://dbpedia.org/resource/Polar_bear        polar    bear        168
    3          http://dbpedia.org/resource/Seahorse             seahorse        198
    4              http://dbpedia.org/resource/Gull          sea    gull        193
'''
def check_entity_correspondences_for_tables():

    def f(stri):
        import string
        punctuation_string = string.punctuation
        for i in punctuation_string:
            stri = stri.replace(i, '')
        return stri

    fn = '79966524_0_1484492112455633784'
    entity_fn = os.path.join(base_dir, 'instance', fn+'.csv')
    table_fn = os.path.join(base_dir, 'tables', fn+'.json')
    table = decode_one_table(table_fn)
    entity_df = pd.read_csv(entity_fn, header=None,
                            names=['ResourceURI', 'KeyValue', 'RowId'])
    for idx in tqdm(range(len(entity_df))):
        try:
            annotated = entity_df.iloc[idx]['KeyValue']
            annotated = f(annotated)
            table_row = [table[_][entity_df.iloc[idx]['RowId']].lower() for _ in range(len(table))]
            # [...,...,...]
            table_row = list(map(f, table_row))
            assert annotated.split() in [_.split() for _ in table_row]
            print([_.split() for _ in table_row].index(annotated.split()))
        except:
            AssertionError
            import ipdb; ipdb.set_trace()
            # print('hard to match')
    return


'''
    检查文件包含情况
'''
def test_1():
    # check_classes_GS_in_table_list()
    # check_property_gs_in_table_list()
    check_instance_gs_in_table_list()
    # check_instance_annotation_on_non_horizontal_table()

'''
    检查 table 能否被正常识别
'''
def test_2():
    # check_raw_language_for_all_table()
    # decode_all_tables()
    # compare_layout() # differente layout of tables
    # check_text_property_for_all_webtables()
    check_hasKeyColumn_for_all_tables()

'''
    检查 entity correspondences for the tables：
        DBpedia resource URI	Key value	Row index
'''
def test_3():
    # decode_annotation_for_entity()
    check_entity_correspondences_for_tables()


if __name__=='__main__':
    test_2()
