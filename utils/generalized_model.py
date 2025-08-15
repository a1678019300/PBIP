import pandas as pd
from kmer import *
import math


def cal_other_feat(encoded_seq):
    freq_count = [encoded_seq.count(idx) for idx in '1234567']
    sum_freq_count = sum(freq_count)
    compositions = [item/sum_freq_count for item in freq_count]

    trans_list = list()
    group_str = '1234567'
    for faa in range(6):
        for saa in range(faa+1, 7):
            trans_list.append([group_str[faa],group_str[saa]])
    N = len(encoded_seq) - 1
    transitions = list()
    for p in trans_list:
        search_str1 = p[0] + p[1]
        search_str2 = p[1] + p[0]
        cur_trans = (encoded_seq.count(search_str1) + encoded_seq.count(search_str2)) / N
        transitions.append(cur_trans)

    # distribution
    distributions = list()
    N = N + 1
    for group in group_str:
        gindexes = [i for i,val in enumerate(encoded_seq) if val==group]
        ng = len(gindexes)
        if ng == 0:
            distributions.extend([0,0,0,0,0])
            continue
        frst_idx = gindexes[0] / N
        frst_quarter = int(ng/4)-1
        frst_quarter = gindexes[frst_quarter] / N
        second_quarter = int(ng/2)-1
        second_quarter = gindexes[second_quarter] / N
        third_quarter = int(3 * ng/4)-1
        third_quarter = gindexes[third_quarter] / N
        fourth_quarter = ng - 1
        fourth_quarter = gindexes[fourth_quarter] / N
        distributions.extend([frst_idx, frst_quarter, second_quarter, third_quarter, fourth_quarter])
    return compositions, transitions, distributions

def get_all_feat(hseq, vseq):
    cur_feat = list()
    kmer_obj = kmerFE()
    # RFAT features
    hrfat = kmer_obj.get_rfat(hseq)
    vrfat = kmer_obj.get_rfat(vseq)
    cur_feat.extend(hrfat)
    cur_feat.extend(vrfat)

    # FDAT features
    abs_diff = [abs(hf-vf) for hf, vf in zip(hrfat, vrfat)]
    avg_F = sum(abs_diff)/len(abs_diff)
    max_F = max(abs_diff)
    fdat = [math.exp((p - avg_F)/(max_F - avg_F)) for p in abs_diff]
    cur_feat.extend(fdat)

    # AC features
    pair_seq = hseq + vseq
    f_list = [pair_seq.count(aa) for aa in kmer_obj.dic.keys()]
    max_aa_f = max(f_list)
    norm_f_list = [item/max_aa_f for item in f_list]
    cur_feat.extend(norm_f_list)

    # compositions, transitions, distributions
    hcomp, htrans, hdis = cal_other_feat(kmer_obj.encode(hseq))
    vcomp, vtrans, vdis = cal_other_feat(kmer_obj.encode(vseq))
    cur_feat.extend(hcomp)
    cur_feat.extend(vcomp)
    cur_feat.extend(htrans)
    cur_feat.extend(vtrans)
    cur_feat.extend(hdis)
    cur_feat.extend(vdis)
    return cur_feat

def get_generalized_feature(file_path, vseq_dict, hseq_dict):
    df = pd.read_csv(file_path).values.tolist()
    data = list()
    for p in df:
        vseq = vseq_dict[p[0]]
        hseq = hseq_dict[p[1]]
        data.append(get_all_feat(hseq, vseq))
    return data




