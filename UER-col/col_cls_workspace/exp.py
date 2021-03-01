from col_cls_workspace.task_col_classifier import *


if __name__ == '__main__':
    op_2_1 = {'pooling': 'avg-token', 'mask_mode': 'cross-wise', 'additional_ban': 2}
    op_3_1 = {
        "train_path": './data/aida/IO/train_samples',
        "t2d_path": './data/aida/IO/test_samples_t2d',
        "limaye_path": './data/aida/IO/test_samples_limaye',
        "wiki_path": './data/aida/IO/test_samples_wikipedia',
        "epochs_num": 25,
    }

    for ds_options in [op_3_1]:
        for key_options in [op_2_1]: # process_2
            predefined_dict_groups = {
                                      'debug_options': {
                                          'logger_dir_name': './col_cls_workspace/log_tem',
                                          'logger_file_name': './col_cls_workspace/rec_all_tem',
                                          'tx_logger_dir_name': './col_cls_workspace/runs/',
                                          'exp_name':  "cleaned-ds"
                                      },
                                      'key_set_group': key_options,
                                      'ds_set_group': ds_options
                                      }
            print(predefined_dict_groups)
            experiment(repeat_time=5, predefined_dict_groups=predefined_dict_groups)
