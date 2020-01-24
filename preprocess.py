import numpy as np
import os
import wfdb
from collections import Counter
import pickle
import random
import sys
from tqdm import tqdm

label_group_map = {'N':'N', 'L':'N', 'R':'N', 'V':'V', '/':'Q', 'A':'S', 'F':'F', 'f':'Q', 'j':'S', 'a':'S', 'E':'V', 'J':'S', 'e':'S', 'Q':'Q', 'S':'S'}

def resample_unequal(ts, fs_in, fs_out):
    """
    interploration
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=fs_in, endpoint=True)
        x_new = np.linspace(0, 1, num=fs_out, endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new

if __name__ == "__main__":

    path = 'data/mit-bih-arrhythmia-database-1.0.0'
    save_path = 'data/'
    # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5'] 
    valid_lead = ['MLII'] 
    fs_out = 360
    test_ratio = 0.2

    train_ind = []
    test_ind = []
    all_pid = []
    all_data = []
    all_label = []
    all_group = []

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    test_pid = random.choices(all_record_name, k=int(len(all_record_name)*test_ratio))
    train_pid = list(set(all_record_name) - set(test_pid))

    for record_name in all_record_name:
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print('read data failed')
            continue
        fs = tmp_data_res[1]['fs']
        ## total 1 second for each
        left_offset = int(1.0*fs / 2)
        right_offset = int(fs) - int(1.0*fs / 2)

        lead_in_data = tmp_data_res[1]['sig_name']
        my_lead_all = []
        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)
        if len(my_lead_all) != 0:
            for my_lead in my_lead_all:
                channel = lead_in_data.index(my_lead)
                tmp_data = tmp_data_res[0][:, channel]

                idx_list = list(tmp_ann_res['sample'])
                label_list = tmp_ann_res['symbol']
                for i in range(len(label_list)):
                    s = label_list[i]
                    if s in label_group_map.keys():
                        idx_start = idx_list[i]-left_offset
                        idx_end = idx_list[i]+right_offset
                        if idx_start < 0 or idx_end > len(tmp_data):
                            continue
                        else:
                            all_pid.append(record_name)
                            all_data.append(resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out))
                            all_label.append(s)
                            all_group.append(label_group_map[s])
                            if record_name in train_pid:
                                train_ind.append(True)
                                test_ind.append(False)
                            else:
                                train_ind.append(False)
                                test_ind.append(True)
                    else:
                        continue
                print('record_name:{}, lead:{}, fs:{}, cumcount: {}'.format(record_name, my_lead, fs, len(all_pid)))
        else:
            print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
            continue

    all_pid = np.array(all_pid)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    all_group = np.array(all_group)
    train_ind = np.array(train_ind)
    test_ind = np.array(test_ind)
    print(all_data.shape)
    print(all_label.shape, np.sum(train_ind), np.sum(test_ind))
    print(Counter(all_label))
    print(Counter(all_group))
    print(Counter(all_group[train_ind]), Counter(all_group[test_ind]))
    np.save(os.path.join(save_path, 'mitdb_data.npy'), all_data)
    np.save(os.path.join(save_path, 'mitdb_label.npy'), all_label)
    np.save(os.path.join(save_path, 'mitdb_group.npy'), all_group)
    np.save(os.path.join(save_path, 'mitdb_pid.npy'), all_pid)
    np.save(os.path.join(save_path, 'mitdb_train_ind.npy'), train_ind)
    np.save(os.path.join(save_path, 'mitdb_test_ind.npy'), test_ind)