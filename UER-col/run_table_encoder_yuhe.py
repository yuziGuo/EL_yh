# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
import os
import random
from col_spec_yh.constants import CSV_SEP
from col_spec_yh.test import generate_seg
# from torch_scatter import scatter_mean, scatter_max

import logging
# logging.basicConfig(level=logging.INFO)
# logging.info('hello') #     terminal: 'INFO:root:hello’

import logging  # 引入logging模块
import os.path
import time

class BertTabEncoder(nn.Module):
    def __init__(self, args, model):
        super(BertTabEncoder, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, src, mask):
        emb = self.embedding(src, mask)
        output = self.encoder(emb, mask)
        # Target.
        # import ipdb; ipdb.set_trace()
        if self.pooling == "mean":
            ex_mask = (mask // 100).unsqueeze(-1).expand(-1, -1, output.size(-1))
            _ = scatter_mean(src=output, index=ex_mask, dim=-2)  # [batch_size, col_num+1, 768]
            output = _[:,1:,:]  # columns
        elif self.pooling == "max":
            ex_mask = (mask // 100).unsqueeze(-1).expand(-1, -1, output.size(-1))
            _ = scatter_max(src=output, index=ex_mask, dim=-2)[0]
            output = _[:,1:,:]
        elif self.pooling == "last":
            output = output[:, -1, :]
        elif self.pooling == 'bert':
            output = output[:, 0, :]
        elif self.pooling == 'crosswise-bert':
            # import ipdb; ipdb.set_trace()
            mask = (mask % 100==1).float() * mask
            _t = torch.FloatTensor([-10000]).repeat(mask.size()[0]).unsqueeze(-1).to(mask.device)
            _for_calc = torch.cat((_t, mask[:, :-1]), 1)
            _idxs = torch.nonzero(((mask - _for_calc)>0).float()).T
            # import ipdb; ipdb.set_trace()
            output = output[_idxs[0], _idxs[1], :].reshape(mask.shape[0], -1, emb.shape[-1])  # [bz,col_num,768]
        output = torch.tanh(self.output_layer_1(output))  # output: [batch_size, emb_size]  #
        return output


def get_args(parser):
    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--not_train", type=str, required=False, default=False,
                        help="For only test option.")
    parser.add_argument("--epochs_num_namely", type=str, required=False, default=False,
                        help="For only test option.")
    parser.add_argument("--dump_logits", type=str, required=False, default=False,
                        help="For only test option.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--taskname", default=None, type=str,
                        help="define the taskname.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--mask_mode", type=str, default='origin')
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    args.target = "bert"
    # args.target = 'nothing'
    return args

def main():
    ## Build model.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = get_args(parser)

    model = build_model(args)
    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    model = BertTabEncoder(args, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            yield input_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, mask_ids_batch

    # Read dataset.
    def read_dataset(taskname, path, is_train=True):
        if is_train: noise_num=2
        else: noise_num=0
        dataset = []
        logging.debug(path)
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.rstrip('\n')
                try:
                    tab_col, cls, tokens, seg = generate_seg(line, args.seq_length, noise_num)
                    dataset.append((tokens, seg, tab_col))  # yuhe note
                    print(len(dataset))
                except:
                    pass
        return dataset

    # Evaluation function.
    import ipdb; ipdb.set_trace()
    dataset = read_dataset(args.taskname, args.test_path, False)
    input_ids = torch.LongTensor([sample[0] for sample in dataset])
    mask_ids = torch.LongTensor([sample[1] for sample in dataset])
    for i, (input_ids_batch, mask_ids_batch) in enumerate(
            batch_loader(args.batch_size, input_ids, mask_ids)):
        input_ids_batch = input_ids_batch.to(device)
        mask_ids_batch = mask_ids_batch.to(device)
        with torch.no_grad():
            model(input_ids_batch, mask_ids_batch)


if __name__ == "__main__":
    main()
