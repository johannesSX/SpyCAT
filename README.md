# Spatially Context-Aware Transformers Facilitate Modeling-Based Anomaly Detection of Subtle Lesions in Brain MRI Images

PyTorch implementation of A [Spatially Context-Aware Transformers Facilitate Modeling-Based
Anomaly Detection of Subtle Lesions in Brain MRI Images]() (MELBA 2025).


## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Dataset
The data can be accessed at the following link.
- [IXI](https://brain-development.org/ixi-dataset/)
- [BRATS](http://www.braintumorsegmentation.org/)

The path to the data is `./data/SKULL/{BRATS, IXI}`.


## Train the VQ-VAE
Start training the vq-vae network with the following line:
```
cd ./vae
python run_vae.py
```

## Train the transformer
Start training the transformer network with the following line:
```
cd ./perf
python run_perf.py --mode FIT 
```

## Evaluation
The evaluation of the network can be started with the following line:
```
cd ./perf
python run_perf.py --mode VALANDTEST 
```
Be sure to adjust other parameters as well. See run_perf.py for details.

## Code References
- [Performer](https://github.com/lucidrains/performer-pytorch)


## Citation
- ...