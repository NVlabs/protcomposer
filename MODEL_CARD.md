# ProtComposer Model Card

## 1. Model Overview

### Description:
ProtComposer is a generative model that generates novel protein structures. It leverages joint protein backbone and sequence flow matching and offers controllability through 3D ellipsoids that can be positioned by the user or an auxiliary model to guide the shape of the protein structure during generation. This enables compositional protein structure generation, advancing controllability in protein design tasks.

The model is for research and non-commercial use only.

### License/Terms of Use:
ProtComposer source code and models are licensed under the NVIDIA license, see [LICENSE.txt](LICENSE.txt). By using ProtComposer, you accept the terms and conditions of this license.

### References:
Research paper: [“ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids,”](https://openreview.net/forum?id=0ctvBgKFgc)

### Model Architecture:
- **Architecture Type:** Transformer (Attention, Cross-Attention, Invariant Point Attention)
- **Network Architecture:** ProtComposer

### Input:
- **Input Types:** Number (numbers for total number of protein residues to generate, generation step size, guidance scale and rotational annealing scale). Tensor (tensors for ellipsoid means, flattened ellipsoid covariance matrices and ellipsoid features).
- **Input Formats:** 
  - Number: Integer (residue number), FP32 (step size, guidance scale, annealing scale)
  - Tensor: Pytorch Tensor
- **Input Parameters:** 
  - Number (Integer and FP32): 1D
  - Tensor: 3D (batch elements, ellipsoids, features)
- **Other Properties Related to Input:** Total number of protein residues to generate, generation step size, guidance scale, rotational annealing scale, ellipsoid number of residues and secondary structure type should be positive numbers.

### Output:
- **Output Types:** Tensor (residue coordinates of generated protein). Tensor (amino acid types of residues of generated protein)
- **Output Formats:** 
  - Tensor (coordinates): Pytorch tensor
  - Tensor (amino acid types): Pytorch integer tensor
- **Output Parameters:** 
  - Tensor (coordinates): 3D (batch, length of protein, spatial dimensions)
  - Tensor (amino acid types): 2D (batch, length of protein)

### Software Integration:
- **Runtime Engine:** Pytorch
- **Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere, NVIDIA Hopper
- **Preferred Operating System:** Linux

### Model Versions:
- **ProtComposer v1.0** (trained on Protein Data Bank)
- **ProtComposer v1.1** (trained on AlphaFold Database)

## 2. Training and Evaluation Datasets

### Training Datasets:
- **Protein Data Bank (PDB)**
  - **Link:** [https://www.rcsb.org/](https://www.rcsb.org/)
  - **Data Collection Method by dataset:** Automatic/Sensors/Human (experimental protein structure determination)
  - **Labeling Method by dataset:** N/A
  - **Properties:** The Protein Data Bank (PDB) contains approx. 200K experimentally determined three-dimensional structures of large biological molecules, such as proteins and nucleic acids, along with auxiliary information such as the protein sequences. We train ProtComposer on a filtered subset of the PDB, comprising 20,312 proteins. Specifically, we use monomers between length 60 and 512 with resolution < 5A downloaded from the PDB with a time cutoff on August 8, 2021. This resulted in 23,913 proteins. The data is further filtered by only including proteins with high secondary structure compositions. For each monomer, we ran the Define Secondary Structure of Proteins (DSSP) algorithm and removed monomers with more than 50% loops, resulting in 20,312 proteins.
- **AlphaFold Protein Structure Database (AFDB)**
  - **Link:** [https://alphafold.ebi.ac.uk/](https://alphafold.ebi.ac.uk/)
  - **Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
  - **Labeling Method by dataset:** N/A
  - **Properties:** The AlphaFold Protein Structure Database (AFDB) contains approx. 214M synthetic three-dimensional protein structures predicted by AlphaFold2, along with their corresponding sequences. We train ProtComposer on a subset of the AFDB, comprising 588,570 structures. We use a subsampled version that applies FoldSeek to cluster entries based on structural similarity. We start with all cluster representatives from the FoldSeek-clustered database and then filter them using a pLDDT threshold of >80, to enrich for highly confident predictions, and a maximum sequence length of 256. This results in 588,570 structures.

### Evaluation Dataset:
- **Protein Data Bank (PDB)**
  - **Link:** [https://www.rcsb.org/](https://www.rcsb.org/)
  - **Data Collection Method by dataset:** Automatic/Sensors/Human (experimental protein structure determination)
  - **Labeling Method by dataset:** N/A
  - **Properties:** The Protein Data Bank (PDB) contains approx. 200K experimentally determined three-dimensional structures of large biological molecules, such as proteins and nucleic acids, along with auxiliary information such as the protein sequences. Similar to Multiflow [Campbell et al., “Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design”](https://arxiv.org/abs/2402.04997), we evaluate ProtComposer on a subset of the PDB, comprising 449 protein structures. Our evaluation set is based on a time-based split of the PDB. We downloaded structures and sequences from the PDB that were released between 1st September 2021 and 28th December 2023. We then select all single chain monomeric proteins with length between 50 and 400 inclusive. We further filter out proteins that are more than 50% coil residues and proteins that have a radius of gyration in the 96th percentile of the original dataset or above. We also filter out structures that have missing residues. We cluster proteins using the 30% sequence identity MMSeqs2 clustering provided by RCSB.org. We take a single protein from each cluster that matches our filtering criteria. This gives us an evaluation set of 449 proteins with minimum length 51 and maximum length 398.

### Inference:
- **Engine:** Pytorch
- **Test Hardware:** A100, H100

### Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).