import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

markers = ['o', 's', 'v', 'D', '*', 'X']
colors = ['C0', 'C1', 'C2', 'C3', 'C6', 'C4']

generalize = [77.06, 85.05, 84.84, 75.08, 95.76, 94.02, 95.88]
generalize_err = [0] * len(generalize)
generalize_err[3] = 0.53
denovo = [77.85, 83.72, 82.02, 74.79, 94.83, 93.15, 95.58]
denovo_err = [0] * len(denovo)
denovo_err[3] = 1.34
names = ['h1n1', 'ebola', 'denovo', 'barman', 'bacillus', 'yersina', 'franci']
mtt = [86.4, 91.23, 84.02, 93.79, 98.76, 97.33, 98.99]
mtt_err = [0.39, 0.43, 1.07, 2.9, 0.36, 0.18, 0.23]

doc2vec = [89.19, 92.33, 88.73, 80.16, 96.31, 94.51, 96.81]
doc2vec_err = [0.47, 0.31, 0.16, 2.12, 0.03, 0.01, 0.04]


mttsub = [85.71, 91.5, 83.75, 95.16, 98.62, 97.53, 98.94]
mttsub_err = [0.26, 1.07, 2.27, 2.56, 0.33, 0.1, 0.22]

stt = [85.98, 90.43, 84.14, 88.99, 97.95, 97.24, 98.88]
stt_err = [0.23, 0.36, 1.89, 5.46, 0.26, 0.33, 0.41]

naive = [76.02, 82.39, 83.90, 73.93, 93.77, 92.63, 94.39]
naive_err = [0] * len(naive)
naive_err[3] = 1.62


x_labels = [ 'Zhou\'s H1N1','Zhou\'s Ebola','Denovo_slim', 'Barman','Bacillus','Yersina','Franci']
x_pos = list(range(len(x_labels)))

def hardcode():
    fig, ax = plt.subplots()

    models = [ 'MTT', 'Doc2vec', 'Denovo', 'Generalized']
    vals = [mtt, doc2vec, denovo, generalize]
    errs = [mtt_err, doc2vec_err, denovo_err, generalize_err]

    for i, val in enumerate(vals):
        j = i if i < 2 else i + 2
        ax.errorbar(x_pos, val,
                # xerr=xerr,
                yerr=errs[i],
                fmt= markers[j], label=models[i],markersize=10, color=colors[j])

    ax.legend()
    ax.set_xlabel('Dataset')
    ax.set_ylabel('F1 score')
    plt.xticks(np.arange(0,len(x_labels), 1), rotation=30)
    ax.set_xticklabels(x_labels)
    plt.tight_layout()

    plt.savefig('small_sota.png')

    plt.close()



def ablation():
    fig, ax = plt.subplots()

    models = ['MTT', 'STT', 'Naive Baseline']
    vals = [mtt, stt,naive]
    errs = [mtt_err, stt_err, naive_err]

    for i, val in enumerate(vals):
        j = i if i == 0 else i + 1
        ax.errorbar(x_pos, val,
                    # xerr=xerr,
                    yerr=errs[i],
                    fmt= markers[j], label=models[i],markersize=10, color=colors[j])#'-' +

    ax.legend()
    ax.set_xlabel('Dataset')
    ax.set_ylabel('F1 score')

    plt.xticks(np.arange(0,len(x_labels), 1), rotation=30)
    ax.set_xticklabels(x_labels)
    plt.tight_layout()

    plt.savefig('small_ablation.png')

    plt.close()



def parse_avg(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    train_rate = df[cols[0]].tolist()
    train_rate = [str(int(item)) for item in train_rate]
    test_rate = df[cols[1]].tolist()
    test_rate = [str(int(item)) for item in test_rate]
    f1 = df[cols[-1]].tolist()
    mean = [round(float(item[:item.find('+-')])*100,2) for item in f1]
    std = [round(float(item[item.find('-') + 1:])*100,2) for item in f1]
    notation = [train + ':' + test for train, test in zip(train_rate, test_rate)]

    df = pd.read_csv(path.replace('_avg', '_allscores'))
    f1s = df[df.columns[-1]].tolist()
    return notation, mean, std, f1s


def draw_novel_plot(data_dir):
    doc2vec = data_dir + 'doc2vec_avg.csv'
    mtt = data_dir + 'mtt_avg.csv'
    stt = data_dir + 'stt_avg.csv'
    naive = data_dir + 'naive_avg.csv'

    notation, doc2vec_avg, doc2vec_err, doc2vec_f1 = parse_avg(doc2vec)
    x_pos = list(range(len(notation)))
    _, mtt_avg, mtt_err, mtt_f1 = parse_avg(mtt)
    _, stt_avg, stt_err, stt_f1 = parse_avg(stt)
    _, naive_avg, naive_err, naive_f1 = parse_avg(naive)
    print(notation)

    models = ['MTT','Doc2vec', 'STT', 'Naive Baseline']
    values = [mtt_avg, doc2vec_avg, stt_avg, naive_avg]
    f1_values = [mtt_f1, doc2vec_f1, stt_f1, naive_f1]
    errs = [mtt_err, doc2vec_err, stt_err, naive_err]
    dataset = 'ebola' if data_dir.find('ebola') > 0 else 'h1n1'
    for model, val, err in zip(models, f1_values, errs):
        if model != 'MTT':
            ttest_val, p = ttest_ind(mtt_f1, val)
            print(model, ttest_val, p)

    print('stt vs naive', ttest_ind(stt_f1, naive_f1))

    # compare with doc2vec
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, val in enumerate(values):
        if i > 1:
            continue
        ax.errorbar(x_pos, val,
                    # xerr=xerr,
                    yerr=errs[i],
                    fmt='-' + markers[i], label=models[i],markersize=10)#'-' +

    ax.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102), #, bbox_to_anchor=(0.5, -0.05), #'upper center'
              fancybox=True, shadow=True, ncol=2)#, mode='expand')

    ax.set_xlabel('Negative training rate: Negative testing rate')
    ax.set_ylabel('F1 score')
    test = np.arange(0,len(notation), 1)
    plt.xticks(test,rotation=30)
    ax.set_xticklabels(notation)
    plt.tight_layout()

    plt.savefig(dataset + '_sota.png')
    # plt.show()

    plt.close()

    # Ablation
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, val in enumerate(values):
        if i == 1:
            continue
        j = i
        ax.errorbar(x_pos, val,
                    yerr=errs[i],
                    fmt= '-' + markers[j], label=models[i],markersize=10, color=colors[j])#'-' +

    ax.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102), #, bbox_to_anchor=(0.5, -0.05), #'upper center'
              fancybox=True, shadow=True, ncol=3)#, mode='expand')
    ax.set_xlabel('Negative training rate: Negative testing rate')
    ax.set_ylabel('F1 score')
    #
    test = np.arange(0,len(notation), 1)
    plt.xticks(test,rotation=30)
    ax.set_xticklabels(notation)
    plt.tight_layout()

    plt.savefig(dataset + '_ablation.png')

    plt.close()


def sar():
    mtt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    mtt_err = [0] * 10
    stt = [0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8]
    stt_err = [0.527, 0.527, 0.5164, 0.5164, 0.5164, 0.483, 0.4216, 0.4216, 0.4216, 0.4216]
    denovo = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    denovo_err = [0] * 10
    doc2vec = [0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    doc2vec_err = [0.527] + [0] * 9
    generalize = [0] * 10
    models = [ 'MTT', 'Doc2vec', 'STT', 'Denovo', 'Generalized']
    x_pos = list(range(1,11))
    values = [mtt, doc2vec, stt, denovo, generalize]
    errs = [mtt_err, doc2vec_err, stt_err, denovo_err,generalize]
    marks = markers[:3] + [markers[4], markers[5]]
    print(marks)
    cols = colors[:3] + [colors[4], colors[5]]
    fig, ax = plt.subplots()#(figsize=(13, 5))

    print(ttest_ind(mtt, doc2vec))
    for val, err, model, col, m in zip(values, errs, models, cols, marks):
        ax.errorbar(x_pos, val,
                    fmt= '-' + m, label=model,markersize=10, color=col, alpha=0.5 if model != 'MTT' else 1)#'-' +

    ax.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102), #, bbox_to_anchor=(0.5, -0.05), #'upper center'
              fancybox=True, shadow=True, ncol=5)#, mode='expand')
    ax.set_xlabel('K')
    ax.set_ylabel('topK')
    #
    test = np.arange(1,11, 1)
    plt.xticks(test,rotation=30)
    ax.set_xticklabels(x_pos)
    plt.tight_layout()

    plt.savefig('case_study.png')

    plt.close()

hardcode()
ablation()
sar()

plt.rcParams.update({'font.size': 22})# comment this when producing plots for small testing datasets
draw_novel_plot('../data/h1n1/')
print('='*30)
draw_novel_plot('../data/ebola/')

