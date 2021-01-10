# avg-token alerted!
from col_cls_workspace.task_col_classifier import *

if __name__ == '__main__':
    op_2_1 = {'pooling': 'avg-token', 'mask_mode': 'cross-wise', 'mask_skip_level_ban': True}
    op_2_2 = {'pooling': 'avg-token', 'mask_mode': 'row-wise', 'mask_skip_level_ban': True}
    op_2_3 = {'pooling': 'avg-token', 'mask_mode': 'cross-and-hier-wise', 'mask_skip_level_ban': True}
    op_2_4 = {'pooling': 'avg-token', 'mask_mode': 'col-wise', 'mask_skip_level_ban': True}

    op_3_1 = {
        "train_path": './data/aida/ff_no_dup_train_samples',
        "t2d_path": './data/aida/ff_no_dup_test_samples_t2d',
        "limaye_path": './data/aida/ff_no_dup_test_samples_limaye',
        "wiki_path": './data/aida/ff_no_dup_test_samples_wikipedia',
        "epochs_num": 30,
    }
    op_3_2 = {
        "train_path": './data/aida/ff_train_samples',
        "t2d_path": './data/aida/ff_test_samples_t2d',
        "limaye_path": './data/aida/ff_test_samples_limaye',
        "wiki_path": './data/aida/ff_test_samples_wikipedia',
        "epochs_num": 6,
    }

    for ds_options in [op_3_2]:
        for key_options in [op_2_1, op_2_2, op_2_3, op_2_4]:
            predefined_dict_groups = {
                'key_set_group': key_options,
                'ds_set_group': ds_options,
                'debug_options':{
                    'exp_file_name' : 'set6',
                  'logger_dir_name': 'col_cls_workspace/log_recover_avg_tokens',
                  'logger_file_name': 'col_cls_workspace/rec_all_recover_avg_tokens'
                },
            }
            print(predefined_dict_groups)
            experiment(repeat_time=1, predefined_dict_groups=predefined_dict_groups)
