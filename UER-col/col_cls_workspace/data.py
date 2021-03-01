import random

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from col_spec_yh.store_utils import decode_aida_ds_in_rows
from col_spec_yh.encode_utils import generate_seg
from demos.utils import set_args_2


def fn_wrapper(args):
    def fn(batch):  # batch: [(<tab-col-id>, <cls-name>, <micro-table-in-cols>), ()]
        labels = list(map(lambda k: args.labels_map[k], [_[1] for _ in batch]))
        raw_tab_ids = [_[0] for _ in batch]
        tab_cols = [_[2] for _ in batch]
        tokens, segs = list(zip(*[generate_seg(args, _) for _ in tab_cols]))
        src = torch.LongTensor(tokens)
        tgt = torch.LongTensor(labels)
        seg = torch.LongTensor(segs)
        return src, tgt, seg, raw_tab_ids
    return fn


class microTableDataset(IterableDataset):
    def __init__(self, data_path=None, train=True, shuffle_rows=True):
        super(microTableDataset).__init__()
        self.tb_to_cls_name, self.tb_to_rows = \
            decode_aida_ds_in_rows(data_path)
        self.train=train
        self.shuffle_rows = shuffle_rows
        self.samples = []

    def __len__(self):
        return sum([(len(_)+4)//5 for _ in self.tb_to_rows.values()])

    def _shuffle_rows(self):
        # print('--------')
        self.tb_to_rows = {k: random.shuffle(v) or v for k,v in self.tb_to_rows.items()}

    def __iter__(self):
        if self.shuffle_rows == True:
            self._shuffle_rows()
        self.samples = []
        for tid, rows in self.tb_to_rows.items():
            micro_tables = [list(zip(*rows[i:i + 5])) for i in range(0, len(rows), 5)]
            self.samples.extend([(tid, self.tb_to_cls_name[tid], micro_table) for micro_table in micro_tables])
        if self.train:
            random.shuffle(self.samples)
        return iter(self.samples)


if __name__=='__main__':
    args = set_args_2()
    ds = microTableDataset(data_path='././data/aida/IO/train_samples/', train=True)
    dl = DataLoader(ds, batch_size=1, collate_fn=fn_wrapper(args))
    for _ in range(2):
        for _ in dl:
            for _ in _:
                print(_)
            break