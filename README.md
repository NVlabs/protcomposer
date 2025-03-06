# <p align="center">ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids <br><br> Oral at ICLR 2025</p>

<div align="center">
  <a href="https://hannes-stark.com/" target="_blank">Hannes&nbsp;Stark*</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://people.csail.mit.edu/bjing/" target="_blank">Bowen&nbsp;Jing*</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://tomasgeffner.github.io/" target="_blank">Tomas&nbsp;Geffner</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://people.csail.mit.edu/jyim/" target="_blank">Jason&nbsp;Yim</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://people.csail.mit.edu/tommi/" target="_blank">Tommi&nbsp;Jaakkola</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a>
  <br> <br>
  *equal contribution <br> <br>
  <a href="https://openreview.net/forum?id=0ctvBgKFgc" target="_blank">Paper</a> &emsp;
</div>
<br><br>
<p align="center">
    <img width="200" alt="Animation" src="assets/image.png"/>
    <img width="750" alt="Animation" src="assets/fig1.png"/>
</p>

## Environment

We recommend installing Miniconda (https://docs.anaconda.com/miniconda/install/) and creating the following conda environment:
```shell script
conda create -n nv python=3.9
pip install jupyterlab
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.1 absl-py einops
pip install pytorch_lightning==2.0.4 fair-esm
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@5484c38'
pip install matplotlib==3.7.2
pip install pydssp biotite omegaconf wandb
pip install numpy==1.21.2
# pip install numpy throws an errror for contourpy but that is fine
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113
pip3 install -U scikit-learn
pip install gpustat
``` 

## Pretrained checkpoints

We provide two pretrained checkpoints in the directory `model_weights`. One is trained on data from the Protein Data Bank, the other on AlphaFold Database.

| Description | Checkpoint path |
|:----------|:----------|
| AlphaFold Database training data | [`model_weights/trained_on_afdb.ckpt`](model_weights/trained_on_afdb.ckpt) |
| Protein Data Bank training data  | [`model_weights/trained_on_pdb.ckpt`](`model_weights/trained_on_pdb.ckpt`) |

## Sampling

To sample ProtComposer conditioned on ellipsoids from our ellipsoid statistical model:

```shell script
python sample.py --guidance 1.0 --num_prots 6 --nu 5 --sigma 6 --helix_frac 0.4  --seed 1 --outdir results --num_blobs 9 --ckpt "model_weights/trained_on_pdb.ckpt"
```

## Evaluation

We use these scripts to compute metrics for the outputs (`evaluate_alignment` computes the ellipsoid adherence metrics):
```shell script
python -m scripts.evaluate_designability --dir results
python -m scripts.evaluate_alignment --dir results
```


## Training


### Data preparations

We use the data from MultiFlow: https://github.com/jasonkyuyim/multiflow

They host the datasets on Zenodo [here](https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A).
Download the following files, and place them into the directory `data`
* `real_train_set.tar.gz` (2.5 GB)
* `synthetic_train_set.tar.gz` (220 MB)
* `test_set.tar.gz` (347 MB)
Next, untar the files
```bash
# Uncompress training data
mkdir train_set
tar -xzvf real_train_set.tar.gz -C train_set/
tar -xzvf synthetic_train_set.tar.gz -C train_set/

# Uncompress test data
mkdir test_set
tar -xzvf test_set.tar.gz -C test_set/
```

Download the files and put them into the directory `data` such that you obtain the following file structure (the `--pkl_dir` argument is `data` by default):
```bash
data
├── train_set
│   ├── processed_pdb
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
│   ├── processed_synthetic
|   |   └── <protein_id>.pkl
├── test_set
|   └── processed
|   |   ├── <subdir>
|   |   |   └── <protein_id>.pkl
...
```

### Launch training run

By default, we train on 8 GPUs.

```shell script
python train.py --batch_size 8 --designability --designability_freq 5 --num_designability_prots 50 --accumulate_grad 8 --inf_batches 5 --val_batches 5 --finetune --dataset multiflow --self_condition --num_workers 10 --save_val --epochs 1000 --run_name my_run_name --wandb
```

## License

Code and model weights are released under an NVIDIA license for non-commercial or research purposes only. Please see the [LICENSE.txt](https://github.com/NVlabs/protcomposer/blob/main/LICENSE.txt) file.


## Citation


```bib
@inproceedings{stark2025protcomposer,
  title={ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids},
  author={Hannes Stark and Bowen Jing and Tomas Geffner and Jason Yim and Tommi Jaakkola and Arash Vahdat and Karsten Kreis},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=0ctvBgKFgc}
}
```
