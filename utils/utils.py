import os
import random
from utils.kmer import *
import pandas as pd
import numpy as np
from sklearn import metrics

def standardize_dir(dir):
    res_dir = dir
    if not res_dir.endswith('/') and not res_dir.endswith('\\'):
        res_dir += '/'

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return res_dir

def load_seq_dict(file_path):
    df = pd.read_csv(file_path, header=None).values.tolist()
    seq_dict = dict()
    for i, p in enumerate(df):
        seq_dict[i] = p[1]
    return seq_dict


def get_denovo_feature(file_path, vseq_dict, hseq_dict):
    df = pd.read_csv(file_path).values.tolist()
    data = list()
    kmer_obj = kmerFE()
    for p in df:
        cur_feat = list()
        vseq = vseq_dict[p[0]]
        hseq = hseq_dict[p[1]]
        hkmer = kmer_obj.kmer_composition(hseq)
        vkmer = kmer_obj.kmer_composition(vseq)
        minH = min(hkmer)
        maxH = max(hkmer)
        minV = min(vkmer)
        maxV = max(vkmer)
        norm_vfeat = [(item-minV)/(maxV - minV) for item in vkmer]
        norm_hfeat = [(item-minH) / (maxH - minH) for item in hkmer]
        cur_feat.extend(norm_vfeat)
        cur_feat.extend(norm_hfeat)
        data.append(cur_feat)
    return data

def get_threshold(targets, preds):
    if type(targets) != list:
        targets = targets.tolist()
    N = targets.count(1.0)
    tmplist = list(preds)
    tmplist2 = list(preds)
    while len(tmplist2) > N:
        tmplist = list(tmplist2)
        thres = random.randint(0,N)
        thres = tmplist[thres]
        tmplist2 = [item for item in tmplist if item > thres]
        if len(tmplist2) < N:
            break
    sort_list = sorted(tmplist, reverse=True)
    return sort_list[N-1]

def get_top(K, preds, targets):
    columns = ['y_pred', 'y_true','idx']
    indexes = [i for i in range(len(preds))]
    join_list = [[pred, target, idx] for pred, target, idx in zip(preds, targets, indexes)]
    df = pd.DataFrame(np.array(join_list), columns=columns)

    unique_y_pred = df['y_pred'].unique().tolist()
    tmp_df = df[df['y_true'] == 1.0]
    all_pos = len(tmp_df)

    selected_list = list()
    for item in sorted(unique_y_pred, reverse=True):
        pairs = df[df['y_pred'] == item].values.tolist()
        for p in pairs:
            selected_list.append(p)
            if len(selected_list) >= K:
                return selected_list, all_pos


def get_score(targets, preds, K=10):
    auc_score = metrics.roc_auc_score(targets, preds, average='micro')

    aupr_score = metrics.average_precision_score(targets, preds, average='micro')

    thres = get_threshold(targets, preds)
    y_preds = [0 if pred < thres else 1 for pred in preds]
    cm = metrics.confusion_matrix(targets, y_preds)
    tn, fp, fn, tp = cm.ravel()
    sn = round(float(tp) / (tp + fn),4)
    sp = round(float(tn) / (tn + fp),4)

    acc = round(metrics.accuracy_score(targets, y_preds),4)
    f1 = metrics.f1_score(targets, y_preds)
    precision = metrics.precision_score(targets, y_preds)
    recall = metrics.recall_score(targets, y_preds)

    selected_list, n_pos_samples = get_top(K, preds, targets)
    arr = np.array(selected_list)
    lbls = arr[:,1].tolist()
    freq = lbls.count(1.0)
    # freq = 0
    # sn = sp = acc = precision = recall = f1 = 0

    return auc_score, aupr_score, sn, sp, acc, freq, precision, recall, f1

def get_score2(targets, preds, K=10):
    auc_score = metrics.roc_auc_score(targets, preds, average='micro')

    aupr_score = metrics.average_precision_score(targets, preds, average='micro')
    return auc_score, aupr_score

def get_test_pairs(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path).values.tolist()
    neg_df = pd.read_csv(neg_path).values.tolist()
    int_edges = pos_df + neg_df
    return int_edges