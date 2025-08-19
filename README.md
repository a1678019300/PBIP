## PBIP

Code and Datasets for "PBIP: A Deep Learning Framework for Predicting Phage–Bacterium Interactions at the Strain Level"

### Datasets

Due to GitHub file size limitations, the phenotype file, raw protein and nucleotide sequences, and UniRep features of the strain-level dataset (PBIP), along with the UniRep features of the species-level dataset (PredPHI), are hosted on Google Drive at [https://drive.google.com/drive/folders/1c1JNePxM5IFlTqt6CglUHWi-J-Z_1uoo]. Users need to download the data folder from the Google Drive and place it within the PBIP directory to run the code properly.

### Models

Due to GitHub file size limitations, the trained models for the strain-level dataset (PBIP) and the species-level dataset (PredPHI) are hosted in the model folder on Google Drive, available at: [https://drive.google.com/drive/folders/1c1JNePxM5IFlTqt6CglUHWi-J-Z_1uoo].

### Environment Requirement

Detailed package information for protein sequence embedding using UniRep is provided in UniRep.yaml, and the package information for the deep learning framework PBIP is provided in PBIP.yaml.

### Protein Sequence Embedding

To extract UniRep features for phages and bacteria:

```
python get_UniRep_phage.py
```

```
python get_UniRep_host.py
```

### Conducting phage-bacteriums Prediction

To predict interactions in the strain-level dataset (PBIP) and species-level dataset (PredPHI):

```
python PBIP_dataset_PBIP.py
```

```
python PBIP_dataset_PredPHI.py
```


