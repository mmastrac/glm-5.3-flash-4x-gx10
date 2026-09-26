#!/usr/bin/env python3
"""Collective-corruption test for the four GLM boxes, no model loaded.

Repeats the collectives a GLM-5.3-Flash TP=4 decode step issues -- bf16
all-reduce of [M, 4096] (o_proj, KDA o_proj, MoE, embedding: ~90 per step)
and the bf16 logits all-gather of [M, 38720] -> [M, 154880] -- with inputs
every rank can regenerate deterministically, and checks each result:

  all-gather : bit-exact against the locally rebuilt concatenation
  all-reduce : "int" data (bf16 integers in [-32, 32], exact in bf16) must be
               bit-exact; "float" data must be within 2 bf16 ulps of an fp32
               reference; in both cases every rank must hold the identical
               result (checked by a gloo all_gather of a hash) and a repeated
               all-reduce of the same input must be bit-identical.

Launch one process per box with RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set
(see nccl_test_launch.sh). --mode picks the communicator:
  vllm   : vLLM's own TP group (init_distributed_environment +
           initialize_model_parallel), i.e. the same dispatcher and PyNCCL
           communicator the model runner uses
  pynccl : vllm.distributed.device_communicators.pynccl.PyNcclCommunicator
           on a gloo process group
  torch  : torch.distributed NCCL backend
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = 4096
VOCAB = 154880


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="vllm", choices=["vllm", "pynccl", "torch"])
    p.add_argument("--iters", type=int, default=2000, help="decode steps to emulate")
    p.add_argument("--ar-per-step", type=int, default=90)
    p.add_argument("--tokens", default="1,2,4,8,16,64")
    p.add_argument("--prefill-tokens", type=int, default=2304, help="0 to skip")
    p.add_argument("--data", default="both", choices=["int", "float", "both"])
    return p.parse_args()


def main():
    args = parse()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    init = f"tcp://{master}:{port}"

    for k in sorted(os.environ):
        if k.startswith(("NCCL_", "GLOO_", "VLLM_HOST_IP")):
            print(f"[rank {rank}] {k}={os.environ[k]}", flush=True)

    if args.mode == "vllm":
        from vllm.distributed import (
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        from vllm.config import VllmConfig, set_current_vllm_config

        _cfg_ctx = set_current_vllm_config(VllmConfig())
        _cfg_ctx.__enter__()  # the TP group and its communicator read the config
        init_distributed_environment(world, rank, init, local_rank=0, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        tp = get_tp_group()
        cpu_group = tp.cpu_group

        def all_reduce(x):
            return tp.all_reduce(x)

        def all_gather(x):
            return tp.all_gather(x, dim=-1)

        names = getattr(getattr(tp, "device_communicator", None), "__class__", type(None)).__name__
        print(f"[rank {rank}] vllm tp group world={tp.world_size} device_communicator={names}", flush=True)
    else:
        dist.init_process_group("gloo" if args.mode == "pynccl" else "nccl",
                                init_method=init, rank=rank, world_size=world)
        cpu_group = dist.new_group(backend="gloo") if args.mode == "torch" else dist.group.WORLD
        if args.mode == "pynccl":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
            assert not comm.disabled

            def all_reduce(x):
                return comm.all_reduce(x)

            def all_gather(x):
                out = torch.empty(x.shape[0], x.shape[1] * world, dtype=x.dtype, device=dev)
                # nccl all_gather is rank-major over the flat buffer
                tmp = torch.empty(world, x.shape[0], x.shape[1], dtype=x.dtype, device=dev)
                comm.all_gather(tmp, x.contiguous())
                out.copy_(tmp.permute(1, 0, 2).reshape(x.shape[0], -1))
                return out
        else:
            def all_reduce(x):
                y = x.clone()
                dist.all_reduce(y)
                return y

            def all_gather(x):
                parts = [torch.empty_like(x) for _ in range(world)]
                dist.all_gather(parts, x.contiguous())
                return torch.cat(parts, dim=-1)

    def gen(seed, shape, data):
        g = torch.Generator(device=dev).manual_seed(seed)
        if data == "int":
            return torch.randint(-32, 33, shape, device=dev, generator=g).to(torch.bfloat16)
        return (torch.randn(shape, device=dev, generator=g) * 3).to(torch.bfloat16)

    def rank_hash(t):
        v = t.contiguous().view(torch.int16).reshape(-1).to(torch.int64)
        w = torch.arange(1, v.numel() + 1, device=dev, dtype=torch.int64)
        return int((v * w).sum().item() & 0xFFFFFFFFFFFF)

    def check_consistent(y, tag):
        h = rank_hash(y)
        objs = [None] * world
        dist.all_gather_object(objs, h, group=cpu_group)
        if len(set(objs)) != 1:
            return f"{tag}: ranks disagree, hashes={objs}"
        return None

    tokens = [int(t) for t in args.tokens.split(",")]
    datas = ["int", "float"] if args.data == "both" else [args.data]
    failures = 0
    t0 = time.time()
    total = 0
    for it in range(args.iters):
        M = tokens[it % len(tokens)]
        data = datas[it % len(datas)]
        for j in range(args.ar_per_step):
            seed = it * 1_000_003 + j * 7919
            xs = [gen(seed + r * 31, (M, HIDDEN), data) for r in range(world)]
            ref32 = torch.zeros(M, HIDDEN, device=dev)
            for r in range(world):
                ref32 += xs[r].float()
            y = all_reduce(xs[rank])
            y2 = all_reduce(xs[rank])
            total += 2
            torch.cuda.synchronize()
            msg = None
            if not torch.isfinite(y).all():
                msg = "non-finite output"
            elif data == "int":
                if not torch.equal(y.float(), ref32):
                    bad = (y.float() != ref32).nonzero()
                    msg = f"int all-reduce mismatch at {bad[:4].tolist()} got {y.float()[tuple(bad[0])].item()} want {ref32[tuple(bad[0])].item()}"
            else:
                err = (y.float() - ref32).abs()
                # bound relative to the partial sums NCCL may round, not the
                # (possibly cancelled) final value; the int rows are the exact check
                ulp = sum(x.float().abs() for x in xs).clamp(min=1e-2) * 2 ** -7
                if (err > 2 * ulp).any():
                    bad = (err > 2 * ulp).nonzero()
                    msg = f"float all-reduce error at {bad[:4].tolist()} got {y.float()[tuple(bad[0])].item()} want {ref32[tuple(bad[0])].item()}"
            if msg is None and not torch.equal(y, y2):
                msg = "repeat all-reduce differs (nondeterministic)"
            if msg is None and (j % 15 == 0):
                msg = check_consistent(y, "all-reduce")
            if msg:
                failures += 1
                print(f"[rank {rank}] FAIL step={it} ar={j} M={M} data={data}: {msg}", flush=True)
        # logits all-gather
        seed = it * 1_000_003 + 999_331
        xs = [gen(seed + r * 31, (M, VOCAB // world), "float") for r in range(world)]
        y = all_gather(xs[rank])
        total += 1
        torch.cuda.synchronize()
        exp = torch.cat(xs, dim=-1)
        if not torch.equal(y, exp):
            bad = (y != exp).nonzero()
            failures += 1
            print(f"[rank {rank}] FAIL step={it} all-gather M={M}: mismatch at {bad[:4].tolist()} (first bad column {int(bad[0,1])} = rank {int(bad[0,1]) // (VOCAB // world)}'s slice)", flush=True)
        if args.prefill_tokens and it % 200 == 0:
            xs = [gen(seed + 5 + r * 31, (args.prefill_tokens, HIDDEN), "int") for r in range(world)]
            ref32 = sum(x.float() for x in xs)
            y = all_reduce(xs[rank])
            total += 1
            torch.cuda.synchronize()
            if not torch.equal(y.float(), ref32):
                failures += 1
                print(f"[rank {rank}] FAIL step={it} prefill all-reduce [{args.prefill_tokens},4096] mismatch", flush=True)
        if it % 100 == 0 and rank == 0:
            print(f"[rank 0] step {it}/{args.iters} M={M} {data} collectives={total} failures={failures} {time.time() - t0:.0f}s", flush=True)
    # global verdict
    objs = [None] * world
    dist.all_gather_object(objs, failures, group=cpu_group)
    if rank == 0:
        print(f"RESULT: failures per rank {objs}, collectives per rank {total}, {'CLEAN' if sum(objs) == 0 else 'CORRUPT'}", flush=True)
    dist.barrier(group=cpu_group)
    sys.exit(0 if sum(objs) == 0 else 1)


if __name__ == "__main__":
    main()
