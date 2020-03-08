from resnet1d.util import read_data_physionet_4
import numpy as np
from dtw import dtw, accelerated_dtw
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
from wfdb import processing
from collections import Counter
from joblib import Parallel, delayed

def plot_longterm_dtw():
    X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4()
    print(X_train.shape, Y_train.shape)
    np.random.seed(0)

    for warp in [1,2,3]:

        for _ in tqdm(range(50)):
            idx1 = np.random.choice(list(range(X_train.shape[0])))
            idx2 = np.random.choice(list(range(X_train.shape[0])))
            seg1 = np.squeeze(X_train[idx1])
            seg2 = np.squeeze(X_train[idx2])
            seg1 = seg1 - np.min(seg1) + 3
            seg2 = seg2 - np.max(seg2) - 3

            start = timeit.default_timer()
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(seg1, seg2, dist='euclidean', warp=warp)
            end = timeit.default_timer()
            print ("time: {} s".format(end-start))
            plt.figure(figsize=(20,3))
            plt.plot(seg1)
            plt.plot(seg2)
            for i in range(0, len(path[0]), 50):
                x1 = path[0][i]
                x2 = path[1][i]
                y1 = seg1[x1]
                y2 = seg2[x2]
                plt.plot([x1, x2], [y1, y2], c='r')
            plt.savefig('img_dtw_rhythm/{}_{}_{}_{:.4f}.png'.format(warp, idx1, idx2, d))

def plot_shortterm_dtw():

    wrap = 1
    # average wave dtw
    X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4()
    print(X_train.shape, Y_train.shape)
    np.random.seed(0)

    for _ in tqdm(range(300)):
        idx1 = np.random.choice(list(range(X_train.shape[0])))
        idx2 = np.random.choice(list(range(X_train.shape[0])))
        seg1 = np.squeeze(X_train[idx1])
        seg2 = np.squeeze(X_train[idx2])
        beat1 = get_avg_beat(seg1)
        beat2 = get_avg_beat(seg2)
        beat1_dist = beat1 - np.mean(beat1)
        beat2_dist = beat2 - np.mean(beat2)
        beat1_plot = beat1 - np.min(beat1) + 3
        beat2_plot = beat2 - np.max(beat2) - 3
        
        start = timeit.default_timer()
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(beat1_dist, beat2_dist, dist='euclidean', warp=wrap)
        end = timeit.default_timer()
        print ("time: {} s".format(end-start))
        plt.figure(figsize=(5,3))
        plt.plot(beat1_plot)
        plt.plot(beat2_plot)
        for i in range(0, len(path[0]), 10):
            x1 = path[0][i]
            x2 = path[1][i]
            y1 = beat1_plot[x1]
            y2 = beat2_plot[x2]
            plt.plot([x1, x2], [y1, y2], c='r')
        plt.savefig('img_dtw_beat/{:.4f}_{}_{}.png'.format(d, idx1, idx2))

def get_avg_beat(seg, window=200, fs=300):
    out = []
    left_offset = window//2
    right_offset = window - left_offset
    len_seg = len(seg)
    qrs_inds = processing.xqrs_detect(sig=seg, fs=fs, verbose=False)
    if len(qrs_inds) == 0:
        return np.zeros(window)
    for ind in qrs_inds:
        if ind >= left_offset and ind <= (len_seg-right_offset):
            out.append(seg[(ind-left_offset):(ind+right_offset)])
    out = np.array(out)
    if len(out.shape) == 1:
        avg_beat = out
    else:
        avg_beat = np.mean(out, axis=0)
    return avg_beat

def get_all_beat():
    """
    this code is a piece of mess, don't look it. 
    """
    window = 200
    np.random.seed(0) # this line is important
    X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_4()
    print(X_train.shape, Y_train.shape)
    print(np.sum(X_train[0])) # 173.1752265279791
    n_sample = X_train.shape[0]
    all_beat = []
    for i in tqdm(range(n_sample)):
        seg = np.squeeze(X_train[i])
        beat = get_avg_beat(seg, window=window)
        if len(beat) == 1:
            beat = np.zeros(window)
        all_beat.append(beat)
    all_beat = np.array(all_beat)
    print(all_beat.shape)
    np.save('all_beat.npy', all_beat)

    out = []
    all_beat = np.load('all_beat.npy', allow_pickle=True)
    for i in range(len(all_beat)):
        if len(all_beat[i]) == 0:
            all_beat[i] = np.zeros(window)
        else:
            out.append(list(all_beat[i]))
    all_beat = np.array(all_beat)
    print(all_beat.shape)
    all_len = [type(_) for _ in all_beat]
    print(Counter(all_len))
    out = np.array(out)
    print(out.shape)
    np.save('all_beat.npy', out)

def compute_batch(i_batch):
    start_idx = i_batch*batch_size
    end_idx = start_idx + batch_size
    if end_idx > n_sample:
        end_idx = n_sample

    batch_beat = all_beat[start_idx:end_idx]
    print(batch_beat.shape)
    local_batch_beat = batch_beat.shape[0]
    mat = np.zeros((local_batch_beat, local_batch_beat))

    for i in tqdm(range(0, local_batch_beat-1)):
        for j in range(i+1, local_batch_beat):
        
            beat1 = batch_beat[i]
            beat2 = batch_beat[j]
            beat1_dist = beat1 - np.mean(beat1)
            beat2_dist = beat2 - np.mean(beat2)
            
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(beat1_dist, beat2_dist, dist='euclidean', warp=wrap)
            mat[i,j] = d

    np.save('sim_mat/{}.npy'.format(i_batch), mat)


if __name__ == "__main__":

    np.random.seed(0) # this line is important

    all_beat = np.load('all_beat.npy')
    n_sample = all_beat.shape[0]
    print(all_beat.shape)
    wrap = 1
    batch_size = 256
    all_batchs = list(range(n_sample//batch_size+1))
    # all_batchs = all_batchs[:10]
    print(all_batchs)

    # Parallel(n_jobs=28)(delayed(compute_batch)(i) for i in all_batchs)

    compute_batch(all_batchs[-1])