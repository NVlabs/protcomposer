# ProtComposer Model Card

## 1. Model Overview

### Description:
ProtComposer is a generative model that generates novel protein structures. It leverages joint protein backbone and sequence flow matching and offers controllability through 3D ellipsoids that can be positioned by the user or an auxiliary model to guide the shape of the protein structure during generation. This enables compositional protein structure generation, advancing controllability in protein design tasks.

This model is ready for commercial use.

### License/Terms of Use:
ProtComposer source code is licensed under Apache 2.0 and the model is licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). By using ProtComposer, you accept the terms and conditions of this license.

**Deployment Geography:** Global

**Use Case:**  
The ProtComposer model enables researchers and commercial entities in the Drug Discovery and Life Sciences fields to generate novel protein structures and sequences from user-defined three-dimensional layouts. The outputs can be used in the development of protein-based therapeutics, enzymes, and biomaterials.

**Release Date:**  
Github 03/06/2025 via https://github.com/NVlabs/protcomposer

### References:
Research paper: [“ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids,”](https://openreview.net/forum?id=0ctvBgKFgc)

### Model Architecture:
- **Architecture Type:** Transformer (Attention, Cross-Attention, Invariant Point Attention)
- **Network Architecture:** ProtComposer

### Input:
- **Input Types:** Number (numbers for total number of protein residues to generate, generation step size, guidance scale and rotational annealing scale). Tensor (tensors for ellipsoid means, flattened ellipsoid covariance matrices and ellipsoid features).
- **Input Formats:** 
  - Number: Integer (residue number), FP32 (step size, guidance scale, annealing scale)
  - Tensor: PyTorch Tensor
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
- **Other Properties Related to Output:** None

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

### Software Integration:
- **Runtime Engine:** PyTorch
- **Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere, NVIDIA Hopper
- **Preferred Operating System:** Linux

### Model Versions:
- **ProtComposer v1.0** (trained on Protein Data Bank)
- **ProtComposer v1.1** (trained on AlphaFold Database)

## 2. Training and Evaluation Datasets

### Training Datasets:
- **Protein Data Bank (PDB)**
  - **Link:** [https://www.rcsb.org/](https://www.rcsb.org/)
  - **Data Collection Method by dataset:** Hybrid: Automatic/Sensors/Human (experimental protein structure determination)
  - **Labeling Method by dataset:** Not Applicable (N/A)
  - **Properties:** The Protein Data Bank (PDB) contains approx. 200K experimentally determined three-dimensional structures of large biological molecules, such as proteins and nucleic acids, along with auxiliary information such as the protein sequences. We train ProtComposer on a filtered subset of the PDB, comprising 20,312 proteins. Specifically, we use monomers between length 60 and 512 with resolution < 5A downloaded from the PDB with a time cutoff on August 8, 2021. This resulted in 23,913 proteins. The data is further filtered by only including proteins with high secondary structure compositions. For each monomer, we ran the Define Secondary Structure of Proteins (DSSP) algorithm and removed monomers with more than 50% loops, resulting in 20,312 proteins.
- **AlphaFold Protein Structure Database (AFDB)**
  - **Link:** [https://alphafold.ebi.ac.uk/](https://alphafold.ebi.ac.uk/)
  - **Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
  - **Labeling Method by dataset:** N/A
  - **Properties:** The AlphaFold Protein Structure Database (AFDB) contains approx. 214M synthetic three-dimensional protein structures predicted by AlphaFold2, along with their corresponding sequences. We train ProtComposer on a subset of the AFDB, comprising 588,570 structures. We use a subsampled version that applies FoldSeek to cluster entries based on structural similarity. We start with all cluster representatives from the FoldSeek-clustered database and then filter them using a pLDDT threshold of >80, to enrich for highly confident predictions, and a maximum sequence length of 256. This results in 588,570 structures.

### Evaluation Dataset:
- **Protein Data Bank (PDB)**
  - **Link:** [https://www.rcsb.org/](https://www.rcsb.org/)
  - **Data Collection Method by dataset:** Hybrid: Automatic/Sensors/Human (experimental protein structure determination)
  - **Labeling Method by dataset:** N/A
  - **Properties:** The Protein Data Bank (PDB) contains approx. 200K experimentally determined three-dimensional structures of large biological molecules, such as proteins and nucleic acids, along with auxiliary information such as the protein sequences. Similar to Multiflow [Campbell et al., “Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design”](https://arxiv.org/abs/2402.04997), we evaluate ProtComposer on a subset of the PDB, comprising 449 protein structures. Our evaluation set is based on a time-based split of the PDB. We downloaded structures and sequences from the PDB that were released between 1st September 2021 and 28th December 2023. We then select all single chain monomeric proteins with length between 50 and 400 inclusive. We further filter out proteins that are more than 50% coil residues and proteins that have a radius of gyration in the 96th percentile of the original dataset or above. We also filter out structures that have missing residues. We cluster proteins using the 30% sequence identity MMSeqs2 clustering provided by RCSB.org. We take a single protein from each cluster that matches our filtering criteria. This gives us an evaluation set of 449 proteins with minimum length 51 and maximum length 398.

### Inference:
- **Engine:** PyTorch
- **Test Hardware:** A100, H100

### Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Model Subcards

### Bias

| Field | Response |
| :---- | :---- |
| Participation considerations from adversely impacted groups ([protected classes](https://www.senate.ca.gov/content/protected-classes)) in model design and testing: | None of the above |
| Measures taken to mitigate against unwanted bias: | None |

### Explainability

| Field | Response |
| :---- | :---- |
| Intended Applications & Domains: | Molecular drug discovery and protein design |
| Model Type: | Protein Structure Generator |
| Intended Users: | Computational biologists, protein engineers, and researchers designing novel proteins. |
| Output: | 3D Tensor (residue coordinates of generated protein): batch, length of protein, spatial dimensions. 2D Tensor (amino acid types of residues of generated protein): batch, length of protein |
| Describe how the model works: | Generates novel protein backbone structures and sequences using flow matching and spatial layout represented by 3D ellipsoids. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable |
| Technical Limitations: | The model can only generate monomers and is unable to perform complicated protein design tasks, such as motif scaffolding or protein binder design. |
| Verified to have met prescribed quality standards: | Yes |
| Performance Metrics: | Coverage, Misplacement, Accuracy, Likelihood, Soft Accuracy, Resegment JSD, Designability, Novelty, Diversity, Helicity |
| Potential Known Risks: | This model may output protein structures that are unrealistic, of low quality, and are not designable. |
| Licensing: | Apache 2.0 for source code, NVIDIA Open Model License for model |

### Privacy

| Field | Response |
| :---- | :---- |
| Generatable or Reverse engineerable personally-identifiable information? | None |
| Was consent obtained for any personal data used? | None Known |
| Personal data used to create this model? | None Known |
| How often is the dataset reviewed? | Before Release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Applicable NVIDIA Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |

### Safety and Security

| Field | Response |
| :---- | :---- |
| Model Application(s): | Protein Structure Generation, Protein Design |
| Describe life critical application (if present): | Experimental drug discovery and medicine: Additional in silico and in vitro tests are recommended before using the proteins for downstream applications. |
| Use Case Restrictions: | Your use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |
| Model and Dataset Restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |