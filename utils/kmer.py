import numpy as np
from itertools import product
import math

class kmerFE:
    """
    Feature extraction class
    """
    def __init__(self,kk=3):
        self.kk = kk
        self.dic={'A':'1','G':'1','V':'1','I':'2','L':'2','F':'2','P':'2',
                  'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
                  'R':'5','K':'5','D':'6','E':'6','C':'7'}
        keywords = [''.join(i) for i in product("1234567", repeat = kk)]
        self.idict=dict(zip(keywords,range(len(keywords))))

    def encode(self,text): #to convert amino acids to group
        for i, j in self.dic.items():
            text = text.replace(i, j)
        return text

    def kmer_composition(self, read):
        read  = self.encode(str(read))
        num_kmers = len(read) - self.kk + 1
        Z = np.zeros(len(self.idict))
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = read[i:i+self.kk]
            if kmer in self.idict:
                Z[self.idict[kmer]]+=1
        return Z.tolist()

    # feature vector of 343 length represent the frequency of tri-amino acids group in the sequence
    def get_rfat(self,read):    #returns k-mer dictionary with counts in seq read
        read  = self.encode(str(read))
        num_kmers = len(read) - self.kk + 1
        Z = np.zeros(len(self.idict))
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = read[i:i+self.kk]
            if kmer in self.idict:
                Z[self.idict[kmer]]+=1
        maxZ = max(Z)
        avgZ = sum(Z) / len(Z)
        rfat = [math.exp((p - avgZ)/(maxZ - avgZ)) for p in Z]
        return rfat
