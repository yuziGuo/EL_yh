import os
import sys
from sklearn.utils import Bunch

import logging  # 引入logging模块
import time
import torch

from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer


def get_logger(option="detail", dir_name='logs_default', file_name='log_rec_all'):
    logger = logging.getLogger(option)
    logger.setLevel(logging.DEBUG)  # Log等级总开关

    if option == "detail":
        if not os.path.exists(dir_name) or os.path.isfile(dir_name):
            os.makedirs(dir_name)
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = os.path.join(dir_name, rq) + '.log'
        # add a file handler
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # add a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # DEBUG < INFO （more strict）
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if option == "results":
        logfile = file_name
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        formatter = logging.Formatter("%(asctime)s - %(filename)s [#l:%(lineno)d] - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


    return logger




def get_args():
    args = Bunch()
    args.mask_mode = 'crosswise'
    args.seq_len = 64

    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.emb_size = 768
    args.embedding = 'tab'  # before: bert
    args.encoder = 'bert'
    args.subword_type = 'none'
    args.pooling = 'avg'  # before: crosswise-bert
    args.tokenizer = 'bert'
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    args.batch_size = 4
    args.dropout = 0.1

    # args.target = 'bert'
    return args


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("[YH INFO] : Loading pretrained parameters from {}.".format(args.pretrained_model_path))
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch