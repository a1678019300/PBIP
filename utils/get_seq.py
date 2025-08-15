import pandas as pd
import argparse
import requests as r
from Bio import SeqIO
from io import StringIO
import numpy as np

def get_seq(id_path, all_seq_path, out_seq_path):
    ids = pd.read_csv(id_path, header=None)
    ids.columns = ['id'] + ['dummy' + str(i) for i in range(len(list(ids.columns))-1)]
    ids = ids['id'].values.tolist()

    if all_seq_path != None:
        seq_vals = pd.read_csv(all_seq_path, header=None).values.tolist()
        seq_dict = {pair[0]:pair[1] for pair in seq_vals}
    else:
        seq_dict = dict()

    writer = open(out_seq_path, 'w')
    for iid, id in enumerate(ids):
        print('processing ', iid, id)
        if id not in seq_dict:
            # get the seq info for id
            baseUrl="http://www.uniprot.org/uniprot/"
            idx = id.find('-')
            if idx > 0:
                cur_id = id[:idx]
            else:
                cur_id = id
            currentUrl=baseUrl+cur_id+".fasta"
            response = r.post(currentUrl)
            cData=''.join(response.text)

            Seq=StringIO(cData)
            # print(Seq)
            pSeq=list(SeqIO.parse(Seq,'fasta'))

            if type(pSeq) == str:
                seq = pSeq
            elif type(pSeq) == list:
                if len(pSeq) == 0:
                    response = r.post(currentUrl + '?version=*')
                    cData=''.join(response.text)

                    Seq=StringIO(cData)
                    pSeq=list(SeqIO.parse(Seq,'fasta'))
                    #
                    if len(pSeq) == 0:
                        writer.write(id + ',\n')
                        print('Error!', id)
                        continue
                seq = pSeq[0].format('fasta').strip()
                lines = seq.split('\n')
                seq = ''.join(lines[1:]).strip()
            else:
                print(type(pSeq), pSeq)
            writer.write(id + ',' + seq + '\n')
        else:
            writer.write(id + ',' + seq_dict[id] + '\n')

    writer.close()


parser = argparse.ArgumentParser(description='Get UniRep representation for dir')
parser.add_argument('--id_path', required=True, help='path to the Uniprot ID file')
parser.add_argument('--corpus', default=None, help='path to the existing sequence corpus file')
parser.add_argument('--save_path', default='', help='path to saved_sequence file. If no value is passed, the default value will be used (the input path with the _seq.csv ending)')
args = parser.parse_args()
get_seq(args.id_path, args.corpus if args.corpus != '' else None, args.save_path if args.save_path != '' else args.id_path.replace('.csv','') + '_seq.csv')
