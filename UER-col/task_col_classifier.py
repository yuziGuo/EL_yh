from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise
from col_spec_yh.model import TabEncoder

from sklearn.utils import Bunch
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer
from uer.utils.optimizers import WarmupLinearSchedule, AdamW


from demos.samples.sample_mini_tables import table_a, table_b
# from demos.utils import get_args
from demos.utils import load_or_initialize_parameters
from demos.utils import batch_loader
from demos.utils import get_logger

import random
import os
from collections import defaultdict

import torch
from torch import nn
from col_spec_yh.store_utils import decode_and_verify_aida_file, get_labels_map_from_aida_file


def set_args():
    # options for model
    args = Bunch()
    args.mask_mode = 'crosswise'
    args.pooling = 'avg-token'
    args.seq_len = 90

    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.emb_size = 768
    args.embedding = 'tab'  # before: bert
    args.encoder = 'bert'
    args.subword_type = 'none'
    # args.pooling = 'avg-token'  # before: crosswise-bert
    args.pooling = 'avg-cell-seg'
    args.tokenizer = 'bert'
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    args.warmup = 0.1
    args.batch_size = 32
    args.dropout = 0.1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # options task-specific
    args.epochs_num = 8
    # args.train_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_train_samples'
    # args.t2d_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_t2d'
    # args.limaye_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_limaye'
    # args.wiki_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_wikipedia'
    args.train_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_train_samples'
    args.t2d_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_t2d'
    args.limaye_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_limaye'
    args.wiki_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_wikipedia'
    decode_and_verify_aida_file(args.train_path)
    decode_and_verify_aida_file(args.t2d_path)
    decode_and_verify_aida_file(args.limaye_path)
    decode_and_verify_aida_file(args.wiki_path)


    args.labels_map = get_labels_map_from_aida_file(args.train_path)
    args.labels_num = len(args.labels_map)

    # other options
    args.logger = get_logger(option='detail', dir_name='logs_col')
    args.logger_2 = get_logger(option='results', file_name='rec_all_1')
    args.report_steps = 100
    # args.logger.info('hello logged world')
    # args.logger.debug('hello logged world')
    return args


def read_dataset(args, data_path):
    dataset = []
    raw_tab_id_list, label_names, tab_cols_list = decode_and_verify_aida_file(data_path)
    for raw_tab_id, label_name, tab_col in zip(raw_tab_id_list, label_names, tab_cols_list):
        tokens, seg = generate_seg(args, tab_col, row_wise_fill=True)
        label = args.labels_map.get(label_name)
        dataset.append((tokens, label, seg, raw_tab_id))
    return dataset


class col_classifier(nn.Module):
    def __init__(self, args):
        super(col_classifier, self).__init__()
        self.labels_num = args.labels_num
        self.encoder = TabEncoder(args)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.criterion = nn.NLLLoss()

    def forward(self, src, seg, tgt=None):
        output = self.encoder.encode(src, seg, option='first-column')
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = self.criterion(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.train_steps*args.warmup, t_total=args.train_steps)
    return optimizer, scheduler


def train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    loss, _ = model(src_batch, seg_batch, tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


def eval_batch(args, model, src_batch, seg_batch):
    # model.zero_grad()
    src_batch = src_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    _, logits = model(src_batch, seg_batch)
    return logits


def _loader(args, ds, shuffle=True):
    # ds: list < tokens, label, seg, raw_tab_id>
    # raw_tab_id is not loaded here
    src = torch.LongTensor([example[0] for example in ds])
    tgt = torch.LongTensor([example[1] for example in ds])
    seg = torch.LongTensor([example[2] for example in ds])
    if shuffle:
        random.shuffle(ds)  # todo: set seed
    return batch_loader(args.batch_size, src, tgt, seg)



def evaluate(args, model, epoch_id=None):
    model.eval()

    def reduce_to_tab_level(logits_all, ds):
        raw_tab_to_sample_id = defaultdict(list)
        raw_tab_to_label = {}
        raw_tab_ids = []
        for sample_id, raw_tab_id in enumerate([_[-1] for _ in ds]):
            raw_tab_to_sample_id[raw_tab_id].append(sample_id)  # rawtab_to_sample_id : {'<raw_tab_id>': [0,1,2,...]}
        raw_tab_to_label = {raw_tab: ds[sample_id_list[0]][1]
                            for raw_tab, sample_id_list in raw_tab_to_sample_id.items()}
        raw_tab_ids = raw_tab_to_sample_id.keys()
        tab_level_ground_truth = torch.LongTensor(list(map(lambda i: raw_tab_to_label[i], raw_tab_ids))).to(args.device)
        tab_level_logits = torch.stack(
            [
                torch.mean(logits_all[raw_tab_to_sample_id[raw_tab_id]], dim=0)
                for raw_tab_id in raw_tab_ids
            ]
        )
        # import ipdb; ipdb.set_trace()
        tab_level_preds = torch.argmax(tab_level_logits, dim=1)
        return tab_level_ground_truth, tab_level_preds

    for ds_path in [args.t2d_path, args.wiki_path, args.limaye_path]:
        ds = read_dataset(args, ds_path)
        # ds: list < tokens, label, seg, raw_tab_id >
        with torch.no_grad():
            try:
                logits_all = torch.cat(
                    [
                        eval_batch(args, model, src_batch, seg_batch).view(-1)
                        for src_batch, _, seg_batch in _loader(args, ds, shuffle=False)
                    ]
                ).view(-1, args.labels_num)
            except:
                import ipdb; ipdb.set_trace()
                for src_batch, _, seg_batch in _loader(args, ds, shuffle=False):
                    _b = eval_batch(args, model, src_batch, seg_batch)
                    print(_b.shape)
        tab_level_ground_truth, tab_level_preds = reduce_to_tab_level(logits_all, ds)
        acc_score = accuracy_score(tab_level_ground_truth.data.cpu().numpy(),
                                   tab_level_preds.data.cpu().numpy())

        args.logger.info("Epoch_id: {}\t DataSet: {}\tAcc: {}".format(
            epoch_id, os.path.basename(ds_path), acc_score
        ))
        args.logger_2.info("Epoch_id: {}\t DataSet: {}\tAcc: {}".format(
            epoch_id, os.path.basename(ds_path).split('_')[-1], acc_score
        ))


def train(args, model):
    ds = read_dataset(args, args.train_path)
    args.train_steps = int(len(ds) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    total_loss = 0.
    for epoch in range(1, args.epochs_num+1):
        model.train()

        for i, (src_batch, tgt_batch, seg_batch) in enumerate(_loader(args, ds, shuffle=True)):
            loss = train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.
        # args.logger.INFO()
        # if epoch % 4 == 0:
        evaluate(args, model, epoch_id=epoch)



if __name__ == '__main__':
    for _ in range(10): # repeat 10 times
        args = set_args()
        args.logger.info('args: {}'.format(args))
        args.logger_2.info('args: {}'.format(args))
        model = col_classifier(args)
        load_or_initialize_parameters(args, model.encoder)
        model = model.to(args.device)
        args.logger.info('Model sent to device: {}/{}'.format(model.state_dict()['output_layer_2.bias'].device, args.device))
        train(args, model)

from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise
from col_spec_yh.model import TabEncoder

from sklearn.utils import Bunch
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
from uer.utils.vocab import Vocab
from uer.utils.tokenizer import BertTokenizer
from uer.utils.optimizers import WarmupLinearSchedule, AdamW


from demos.samples.sample_mini_tables import table_a, table_b
# from demos.utils import get_args
from demos.utils import load_or_initialize_parameters
from demos.utils import batch_loader
from demos.utils import get_logger

import random
import os
from collections import defaultdict

import torch
from torch import nn
from col_spec_yh.store_utils import decode_and_verify_aida_file, get_labels_map_from_aida_file


def set_args():
    # options for model
    args = Bunch()
    args.mask_mode = 'crosswise'
    args.pooling = 'avg-token'
    args.seq_len = 90

    args.pretrained_model_path = "./models/bert_model.bin-000"
    args.vocab_path = 'models/google_uncased_en_vocab.txt'
    args.vocab = Vocab()
    args.vocab.load(args.vocab_path)
    args.emb_size = 768
    args.embedding = 'tab'  # before: bert
    args.encoder = 'bert'
    args.subword_type = 'none'
    # args.pooling = 'avg-token'  # before: crosswise-bert
    args.pooling = 'avg-cell-seg'
    args.tokenizer = 'bert'
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    args.feedforward_size = 3072
    args.hidden_size = 768
    args.heads_num = 12
    args.layers_num = 12
    args.learning_rate = 2e-5
    args.warmup = 0.1
    args.batch_size = 32
    args.dropout = 0.1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # options task-specific
    args.epochs_num = 8
    # args.train_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_train_samples'
    # args.t2d_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_t2d'
    # args.limaye_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_limaye'
    # args.wiki_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_no_dup_test_samples_wikipedia'
    args.train_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_train_samples'
    args.t2d_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_t2d'
    args.limaye_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_limaye'
    args.wiki_path = '/home/gyh/pack_for_debug_2/UER-py-master/ff_test_samples_wikipedia'
    decode_and_verify_aida_file(args.train_path)
    decode_and_verify_aida_file(args.t2d_path)
    decode_and_verify_aida_file(args.limaye_path)
    decode_and_verify_aida_file(args.wiki_path)


    args.labels_map = get_labels_map_from_aida_file(args.train_path)
    args.labels_num = len(args.labels_map)

    # other options
    args.logger = get_logger(option='detail', dir_name='logs_col')
    args.logger_2 = get_logger(option='results', file_name='rec_all_1')
    args.report_steps = 100
    # args.logger.info('hello logged world')
    # args.logger.debug('hello logged world')
    return args


def read_dataset(args, data_path):
    dataset = []
    raw_tab_id_list, label_names, tab_cols_list = decode_and_verify_aida_file(data_path)
    for raw_tab_id, label_name, tab_col in zip(raw_tab_id_list, label_names, tab_cols_list):
        tokens, seg = generate_seg(args, tab_col, row_wise_fill=True)
        label = args.labels_map.get(label_name)
        dataset.append((tokens, label, seg, raw_tab_id))
    return dataset


class col_classifier(nn.Module):
    def __init__(self, args):
        super(col_classifier, self).__init__()
        self.labels_num = args.labels_num
        self.encoder = TabEncoder(args)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.criterion = nn.NLLLoss()

    def forward(self, src, seg, tgt=None):
        output = self.encoder.encode(src, seg, option='first-column')
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = self.criterion(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.train_steps*args.warmup, t_total=args.train_steps)
    return optimizer, scheduler


def train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    loss, _ = model(src_batch, seg_batch, tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


def eval_batch(args, model, src_batch, seg_batch):
    # model.zero_grad()
    src_batch = src_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    _, logits = model(src_batch, seg_batch)
    return logits


def _loader(args, ds, shuffle=True):
    # ds: list < tokens, label, seg, raw_tab_id>
    # raw_tab_id is not loaded here
    src = torch.LongTensor([example[0] for example in ds])
    tgt = torch.LongTensor([example[1] for example in ds])
    seg = torch.LongTensor([example[2] for example in ds])
    if shuffle:
        random.shuffle(ds)  # todo: set seed
    return batch_loader(args.batch_size, src, tgt, seg)



def evaluate(args, model, epoch_id=None):
    model.eval()

    def reduce_to_tab_level(logits_all, ds):
        raw_tab_to_sample_id = defaultdict(list)
        raw_tab_to_label = {}
        raw_tab_ids = []
        for sample_id, raw_tab_id in enumerate([_[-1] for _ in ds]):
            raw_tab_to_sample_id[raw_tab_id].append(sample_id)  # rawtab_to_sample_id : {'<raw_tab_id>': [0,1,2,...]}
        raw_tab_to_label = {raw_tab: ds[sample_id_list[0]][1]
                            for raw_tab, sample_id_list in raw_tab_to_sample_id.items()}
        raw_tab_ids = raw_tab_to_sample_id.keys()
        tab_level_ground_truth = torch.LongTensor(list(map(lambda i: raw_tab_to_label[i], raw_tab_ids))).to(args.device)
        tab_level_logits = torch.stack(
            [
                torch.mean(logits_all[raw_tab_to_sample_id[raw_tab_id]], dim=0)
                for raw_tab_id in raw_tab_ids
            ]
        )
        # import ipdb; ipdb.set_trace()
        tab_level_preds = torch.argmax(tab_level_logits, dim=1)
        return tab_level_ground_truth, tab_level_preds

    for ds_path in [args.t2d_path, args.wiki_path, args.limaye_path]:
        ds = read_dataset(args, ds_path)
        # ds: list < tokens, label, seg, raw_tab_id >
        with torch.no_grad():
            try:
                logits_all = torch.cat(
                    [
                        eval_batch(args, model, src_batch, seg_batch).view(-1)
                        for src_batch, _, seg_batch in _loader(args, ds, shuffle=False)
                    ]
                ).view(-1, args.labels_num)
            except:
                import ipdb; ipdb.set_trace()
                for src_batch, _, seg_batch in _loader(args, ds, shuffle=False):
                    _b = eval_batch(args, model, src_batch, seg_batch)
                    print(_b.shape)
        tab_level_ground_truth, tab_level_preds = reduce_to_tab_level(logits_all, ds)
        acc_score = accuracy_score(tab_level_ground_truth.data.cpu().numpy(),
                                   tab_level_preds.data.cpu().numpy())

        args.logger.info("Epoch_id: {}\t DataSet: {}\tAcc: {}".format(
            epoch_id, os.path.basename(ds_path), acc_score
        ))
        args.logger_2.info("Epoch_id: {}\t DataSet: {}\tAcc: {}".format(
            epoch_id, os.path.basename(ds_path).split('_')[-1], acc_score
        ))


def train(args, model):
    ds = read_dataset(args, args.train_path)
    args.train_steps = int(len(ds) * args.epochs_num / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    total_loss = 0.
    for epoch in range(1, args.epochs_num+1):
        model.train()

        for i, (src_batch, tgt_batch, seg_batch) in enumerate(_loader(args, ds, shuffle=True)):
            loss = train_batch(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.
        # args.logger.INFO()
        # if epoch % 4 == 0:
        evaluate(args, model, epoch_id=epoch)



if __name__ == '__main__':
    for _ in range(10): # repeat 10 times
        args = set_args()
        args.logger.info('args: {}'.format(args))
        args.logger_2.info('args: {}'.format(args))
        model = col_classifier(args)
        load_or_initialize_parameters(args, model.encoder)
        model = model.to(args.device)
        args.logger.info('Model sent to device: {}/{}'.format(model.state_dict()['output_layer_2.bias'].device, args.device))
        train(args, model)

