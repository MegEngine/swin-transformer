# Swin-Transformer

Swin-Transformer is basically a hierarchical Transformer whose representation is computed with shifted windows. For more details, please refer to ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/pdf/2103.14030.pdf)

This repo is an implementation of MegEngine version Swin-Transformer. This is also a showcase for training on GPU with less memory by leveraging MegEngine [DTR](https://arxiv.org/abs/2006.09616) technique.

There is also an [official PyTorch implementation](https://github.com/microsoft/Swin-Transformer). 


## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/MegEngine/swin-transformer.git
cd swin-transformer
```

- Install `megengine==1.6.0`

```bash
pip3 install megengine==1.6.0 -f https://megengine.org.cn/whl/mge.html
```

### Training

To train a `Swin Transformer` using random data, run:

```bash
python3 -n <num-of-gpus-to-use> -b <batch-size-per-gpu> -s <num-of-train-steps> train_random.py
```

To train a `Swin Transformer` using [AMP (Auto Mix Precision)](https://megengine.org.cn/doc/stable/en/user-guide/model-development/amp/index.html), run:

```bash
python3 -n <num-of-gpus-to-use> -b <batch-size-per-gpu> -s <num-of-train-steps> --mode mp train_random.py
```

To train a `Swin Transformer` using [DTR in dynamic graph mode](https://megengine.org.cn/doc/stable/en/user-guide/model-development/dtr/index.html#dtr-guide), run:

```bash
python3 -n <num-of-gpus-to-use> -b <batch-size-per-gpu> -s <num-of-train-steps> --dtr [--dtr-thd <eviction-threshold-of-dtr>] train_random.py
```

To train a `Swin Transformer` using [DTR in static graph mode](https://megengine.org.cn/doc/stable/en/user-guide/model-development/jit/trace.html#sublinear-memory), run:

```bash
python3 -n <num-of-gpus-to-use> -b <batch-size-per-gpu> -s <num-of-train-steps> --trace --symbolic --dtr --dtr-thd <eviction-threshold-of-dtr> train_random.py
```

For example, to train a `Swin Transformer` with a single GPU using DTR in static graph mode with `threshold=8GB` and AMP, run:

```bash
python3 -n 1 -b 340 -s 10 --trace --symbolic --dtr --dtr-thd 8 --mode mp train_random.py
```

For more usage, run:

```bash
python3 train_random.py -h
```

## Benchmark

- Testing Devices
  - 2080Ti @ cuda-10.1-cudnn-v7.6.3-TensorRT-5.1.5.0 @ Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
  - Reserve all CUDA memory by setting `MGB_CUDA_RESERVE_MEMORY=1`, in order to alleviate memory fragmentation problem

| Settings                       | Maximum Batch Size | Speed(s/step) | Throughput(images/s) |
| ------------------------------ | ------------------ | ------------- | -------------------- |
| None                           | 68                 | 0.490         | 139                  |
| AMP                            | 100                | 0.494         | 202                  |
| DTR in static graph mode       | 300                | 2.592         | 116                  |
| DTR in static graph mode + AMP | 340                | 1.944         | 175                  |

## Acknowledgement

We are inspired by the [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) repository, many thanks to [microsoft](https://github.com/microsoft)!
