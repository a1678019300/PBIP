## PBIP

Code and Datasets for "PBIP: A Deep Learning Framework for Predicting Phage–Bacterium Interactions at the Strain Level"

### Datasets

The phenotypic file and UniRep features for the strain-level dataset (PBIP) are available in the data folder, while the raw protein and nucleotide sequences are hosted at [https://drive.google.com/drive/folders/1c1JNePxM5IFlTqt6CglUHWi-J-Z_1uoo?usp=drive_link] due to GitHub file size limitations. The UniRep features for the species-level dataset (PredPHI) are also provided in the data folder.


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

