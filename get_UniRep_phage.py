from Bio import SeqIO
import numpy as np
from UniRep.unirep import babbler1900 as babbler
import argparse
import os

parser = argparse.ArgumentParser(description='phage')
parser.add_argument('--model-path', default="./UniRep/1900_weights", help='')
parser.add_argument('--batch-size', type=int, default=1, help='')
args = parser.parse_args()

b = babbler(batch_size=args.batch_size, model_path=args.model_path)


def extract_species_id(fasta_id):
    parts = fasta_id.split('_')
    return '_'.join(parts[:-1]) if len(parts) >= 2 else fasta_id


def process_phage_protein(dataset_name):
    output_dir = f"data/{dataset_name}/phage"
    os.makedirs(output_dir, exist_ok=True)
    input_fasta = f"data/{dataset_name}/phage_protein_seq/all_phage_protein.fasta"

    species_sequences = {}
    for record in SeqIO.parse(input_fasta, "fasta"):
        species_id = extract_species_id(record.id)
        if species_id not in species_sequences:
            species_sequences[species_id] = []
        species_sequences[species_id].append(str(record.seq))


    for i, (species_id, seqs) in enumerate(species_sequences.items(), 1):
        features = []
        for seq in seqs:
            try:
                avg_hidden, _, _ = b.get_rep(seq)
                features.append(avg_hidden)
            except Exception as e:
                continue
        avg_features = np.mean(features, axis=0)
        output_file = os.path.join(output_dir, f"{species_id}.txt")
        np.savetxt(output_file, avg_features, fmt="%g")


if __name__ == "__main__":

    dataset_list = ['PBIP', 'PredPHI']
    for dataset_name in dataset_list:
        process_phage_protein(dataset_name)