# -*- coding: utf-8 -*-
# Copyright (c) 2021 Megvii Inc.
# Licensed under The MIT License [see LICENSE for details]
import argparse
import os
import time
import numpy as np
# pylint: disable=import-error
import swin_transformer as swin_vit

import megengine
import megengine.device as device
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim
import megengine.jit as jit
import megengine.amp as amp


from dataset import get_dataloader
logging = megengine.logger.get_logger()

megengine.device.set_prealloc_config(1024, 1024, 32 * 1024 * 1024, 2.0)

def main():
    parser = argparse.ArgumentParser(description="shufflenet benchmark")
    parser.add_argument(
        "-a",
        "--arch",
        default="swin_tiny_patch4_window7_224",
        help="model architecture (default: swin_tiny_patch4_window7_224)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=1,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=200,
        type=int,
        help="number of train steps (default: 200)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=16,
        type=int,
        help="batch size for single GPU (default: 128)",
    )
    parser.add_argument(
        "--trace",
        action='store_true',
        default=False,
        help="whether use trace or not (default: False)",
    )
    parser.add_argument(
        "--symbolic",
        action='store_true',
        default=False,
        help="whether use symbolic trace or not (default: False)",
    )
    parser.add_argument(
        "--dtr",
        action='store_true',
        default=False,
        help="whether enable DTR optimization or not (default: False)",
    )
    parser.add_argument(
        "--dtr-thd",
        default=0,
        type=float,
        help="eviction threshold of DTR in gigabytes (default: 0)",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        default=0.01,
        help="learning rate for single GPU (default: 0.01)",
    )
    parser.add_argument("--momentum", default=0.9, help="momentum (default: 0.9)")
    parser.add_argument(
        "--weight-decay", default=4e-5, help="weight decay (default: 4e-5)"
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="normal",
        type=str,
        choices=["normal", "mp"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "mp: input type is fp16",
    )

    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--rank", default=0)
    parser.add_argument("--loader", default=False, action="store_true", help="whether use loader")
    parser.add_argument("--preload", default=False, action="store_true", help="whether use preload")


    args = parser.parse_args()

    if args.dtr and args.trace:
        os.environ["MEGENGINE_INPUT_NODE_USE_STATIC_SHAPE"] = "True"
        os.environ["MEGENGINE_INPLACE_UPDATE"] = "1"
        os.environ["MGB_CUDA_RESERVE_MEMORY"] = "1"
    if args.world_size is None:
        args.world_size = args.ngpus
    if args.world_size > 1:
        # launch processes
        train_func = dist.launcher(worker, master_ip=args.dist_addr, port=args.dist_port,
                                world_size=args.world_size, n_gpus=args.ngpus, rank_start=args.rank * args.ngpus)
        train_func(args)
    else:
        worker(args)


def worker(args):
    steps = args.steps

    if args.dtr and not args.trace:
        megengine.dtr.eviction_threshold = int(args.dtr_thd * 1024 ** 3)
        megengine.dtr.enable()

    # build model
    if args.arch in swin_vit.__dict__.keys():
        model = swin_vit.__dict__[args.arch]()
    else:
        raise NotImplementedError

    # Sync parameters
    if args.world_size > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if args.world_size > 1 else None,
    )

    # Optimizer
    params_wd = []
    params_nwd = []
    for n, p in model.named_parameters():
        if n.find("weight") >= 0 and len(p.shape) > 1:
            params_wd.append(p)
        else:
            params_nwd.append(p)
    opt = optim.SGD(
        [{"params": params_wd}, {"params": params_nwd, "weight_decay": 0}, ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay * args.world_size,  # scale weight decay in "SUM" mode
    )

    # train and valid func
    @amp.autocast(enabled=args.mode == "mp")
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)
            gm.backward(loss)
        opt.step().clear_grad()
        return loss

    if args.trace:
        if args.symbolic:
            if args.dtr:
                train_step = jit.trace(train_step, symbolic=True, dtr_config=jit.DTRConfig(eviction_threshold=int(args.dtr_thd)*1024**3), symbolic_shape=False)
            else:
                train_step = jit.trace(train_step, symbolic=True, sublinear_memory_config=jit.SublinearMemoryConfig(genetic_nr_iter=50), symbolic_shape=False)
        else:
            train_step = jit.trace(train_step, symbolic=False, symbolic_shape=False)
    else:
        assert args.symbolic==False, "invalid arguments: trace=Trace, symbolic=True"

    # start training
    objs = AverageMeter("Loss")
    clck = AverageMeter("Time")

    if args.loader:
        dataloader = iter(get_dataloader(args))
        image,label = next(dataloader)
    else:
        image = np.random.randn(args.batch_size, 3, 224, 224).astype("float32")
        label = np.random.randint(0, 1000, size=(args.batch_size,)).astype("int32")


    # warm up
    for step in range(10):
        if args.loader:
            image,label = next(dataloader)
            if not args.preload:
                image = megengine.tensor(image, dtype="float32")
                label = megengine.tensor(label, dtype="int32")
        else:
            image = megengine.tensor(image, dtype="float32")
            label = megengine.tensor(label, dtype="int32")

        loss = train_step(image, label)
        loss.item()

    for step in range(0, steps):
        t = time.time()

        if args.loader:
            image,label = next(dataloader)
            if not args.preload:
                image = megengine.tensor(image, dtype="float32")
                label = megengine.tensor(label, dtype="int32")
        else:
            image = megengine.tensor(image, dtype="float32")
            label = megengine.tensor(label, dtype="int32")

        loss = train_step(image, label)
        objs.update(loss.item())

        clck.update(time.time() - t)
        if step % args.print_freq == 0 and dist.get_rank() == 0:
            print(
                "Step {}, {}, {}".format(
                step,
                objs,
                clck,
            ))
            objs.reset()

    if dist.get_rank() == 0:
        print("="*20, "summary", "="*20)
        print(" benchmark: vision_transformer")
        if args.trace:
            print("      mode: trace(symbolic={})".format(
                    "True, {}".format("dtr=True, thd={}".format(args.dtr_thd) if args.dtr else "sublinear=True") 
                        if args.symbolic else "False"))
        else:
            print("      mode: imperative{}".format("(dtr=true, thd={})".format(args.dtr_thd) if args.dtr else ""))
        print("    loader: {}".format("" if not args.loader else "--loader"))
        if args.loader:
            print("   preload: {}".format("" if not args.preload else "--preload"))
        print("      arch: {}".format(args.arch))
        print("train_mode: {}".format(args.mode))
        print(" batchsize: {}".format(args.batch_size))
        print("      #GPU: {}".format(args.ngpus))
        print("  avg time: {:.3f} seconds".format(clck.avg))

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()