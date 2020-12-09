# from util_kb import lookup_entities
import requests
import sparql
import xml.etree.ElementTree as ET


'''
1. 基本用法
2. 长问句 http://localhost:9274/lookup-application/api/search?query=fly%20me%20to%20the%20moon&MaxHits=10
'''
def get_cand_info_by_mention(row_id, col_id, cell_item='fly me to the moon', top_k=50):
    if len(cell_item) > 50:
        return None
    res_by_dict = look_up_mention(cell_item, top_k)
    if res_by_dict == []:
        return None
    tem = [(row_id, col_id,)+tuple(_.values()) for _ in res_by_dict]
    # ['row_id', 'col_id']+['mention', 'lookup_order', 'label', 'uri', 'clses', 'refCnt']
    return tuple(tem)


def look_up_mention(cell_item='fly me to the moon', top_k=50):
    lookup_url = 'http://localhost:9274/lookup-application/api/search?query=%s&MaxHits=%d' \
                 % (cell_item, top_k)
    lookup_res = requests.get(lookup_url)
    if lookup_res.status_code>400:  # 400, 500
        return []
    root = ET.fromstring(lookup_res.content)
    cand_set = []
    for candidate_idx, child in enumerate(root):
        res = {}
        res['mention'] = cell_item
        res['lookup_order'] = candidate_idx
        res['label'] = child[0].text
        res['uri'] = child[1].text
        res['clses'] = []
        for cls in child[2]:
            res['clses'].append(cls[0].text)
        res['clses'] = str(res['clses'])
        res['refCnt'] = child[3].text
        res['refCnt'] = int(res['refCnt']) if res['refCnt'] else None
        cand_set.append(res)
    return cand_set




def test_api_connection(cell_item='fly me to the moon'):
    cand_set = look_up_mention(cell_item)
    print(cand_set)


def test_1():
    test_api_connection()
    pass


if __name__=='__main__':
    test_1()
    pass