import os
import sys
from sklearn.utils import Bunch

import logging  # 引入logging模块
import os.path
import time
import torch

from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Log等级总开关
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # log_path = os.path.dirname(os.getcwd()) + '/Logs/'
    # log_name = log_path + rq + '.log'
    log_name = 'Logs_2/' + rq + '.log'
    # os.mknod(log_name)
    logfile = log_name
    # logfile = 'yh_logging'
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # add a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
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