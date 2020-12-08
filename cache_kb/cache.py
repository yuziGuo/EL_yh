import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from constants import base_dir
from test_one_table_lookup_and_write_to_table import cache_one_table
import os
from tqdm import tqdm

d = os.path.join(base_dir, 'instance')
test_table_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(d)))
print(test_table_list)

for _ in tqdm(test_table_list):
    cache_one_table(_)