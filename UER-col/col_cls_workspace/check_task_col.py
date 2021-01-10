from col_cls_workspace.task_col_classifier import *
from demos.test_generate_seg_and_mask import check_segs_general

def check(predefined_dict_groups=None):
    args = set_args(predefined_dict_groups=predefined_dict_groups)
    if args.logger:
        args.logger.warning('[For this run] Predefined_dict_groups: {}'.format(predefined_dict_groups))
    #
    for ds_path in [args.train_path, args.t2d_path, args.limaye_path, args.wiki_path]:
        ds = read_dataset(args, ds_path)
        # import ipdb; ipdb.set_trace()
        check_segs_general(args.tokenizer, ds, seg_idx=2, tokens_idx=0)
    #
    model = col_classifier(args)
    load_or_initialize_parameters(args, model.encoder)
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    if args.logger:
        args.logger.info('Model sent to device: {}/{}'.format(model.state_dict()['output_layer_2.bias'].device, args.device))
    train_and_eval(args, model)
    if args.logger:
        logging.getLogger(args.logger_name).handlers = []  # https://stackoverflow.com/questions/7484454/removing-handlers-from-pythons-logging-loggers



op_2_1 = {'pooling':'avg-cell-seg'}
op_3_1 = {
    "train_path" : './data/aida/ff_no_dup_train_samples',
    "t2d_path" : './data/aida/ff_no_dup_test_samples_t2d',
    "limaye_path" : './data/aida/ff_no_dup_test_samples_limaye',
    "wiki_path" : './data/aida/ff_no_dup_test_samples_wikipedia',
    "epochs_num" : 30,
}
for ds_options in [op_3_1]:
    for pooling_options in [op_2_1]:
        predefined_dict_groups = {
                                  'debug_options':{
                                      'logger_dir_name':'log_debug_recover_dup',
                                      'logger_file_name':'rec_all_debug_recover_dup'
                                  },
                                  'pooling_set_group':pooling_options,
                                  'ds_set_group':ds_options
                                  }
        print(predefined_dict_groups)
        check(predefined_dict_groups=predefined_dict_groups)

