"""Collective profiler, emitting `all_reduce.csv` and `send_recv.csv`.

Ranks are torch.multiprocessing processes, so no cluster runtime is needed.
Each row records the `--link-type` given, since a node can reach different GPU
pairs over different fabrics and the same collective differs by close to an
order of magnitude between them.

    python -m vidur.vllm_profiling.collective --collective all_reduce \
        --devices 0,1 --max_collective_size 536870912
"""

import argparse
import datetime
import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .common import (
    ACTIVE_STEPS,
    WARMUP_STEPS,
    CudaTimer,
    TimerStatsStore,
    get_collectives_sizes_to_profile,
    results_to_dataframe,
)


def _worker(rank: int, world_size: int, devices: List[int], sizes: List[int],
            collective: str, port: int, out_path: str):
    """One rank: run the collective at every size, report stats from rank 0."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Bound the wait: a NCCL rendezvous failure should raise, not hang for the
    # whole job.
    from datetime import timedelta
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=10))
    torch.cuda.set_device(devices[rank])

    store = TimerStatsStore()
    results = []

    LARGE = 512 * 1024 * 1024
    for nbytes in sizes:
        # fp16 elements, matching vidur's "size * 2 == bytes" convention
        numel = max(1, nbytes // 2)
        # In the multi-GB tail the allocator cannot always return a contiguous
        # block, so reclaim there only.
        if nbytes >= LARGE:
            torch.cuda.empty_cache()
        try:
            buf = torch.ones(numel, dtype=torch.float16, device=f"cuda:{devices[rank]}")
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"  OOM at size={nbytes/1e9:.2f}GB; stopping sweep here", flush=True)
            break

        def run_once():
            if collective == "all_reduce":
                dist.all_reduce(buf)
            else:  # send_recv, rank0 -> rank1
                if rank == 0:
                    dist.send(buf, 1)
                else:
                    dist.recv(buf, 0)

        for _ in range(WARMUP_STEPS):
            run_once()
        torch.cuda.synchronize()
        dist.barrier()

        store.clear_stats()
        for _ in range(ACTIVE_STEPS):
            with CudaTimer(collective, store):
                run_once()
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            results.append({"time_stats": store.get_stats(), "size": nbytes})
        store.clear_stats()
        del buf
        # No empty_cache() per size point: the allocator reuses the freed block
        # for the next size, and flushing it here dominates runtime.

    # Results go through a file rather than a multiprocessing.Queue: the queue's
    # pipe is too small for a full sweep, and nothing drains it while the parent
    # waits in mp.spawn(join=True).
    if rank == 0:
        import json
        with open(out_path, "w") as f:
            json.dump(results, f)
    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="NCCL collective profiling (vidur-compatible)")
    p.add_argument("--collective", choices=["all_reduce", "send_recv"], default="all_reduce")
    p.add_argument("--devices", default="0,1",
                   help="LOCAL torch device ordinals to use, comma separated")
    p.add_argument("--global-ids", default=None,
                   help="Global device ids, recorded alongside the results")
    p.add_argument("--link-type", default="unknown",
                   help="Interconnect carrying this measurement, recorded in the "
                        "output (e.g. NVLink, PCIe). `nvidia-smi topo -m` reports it.")
    p.add_argument("--max_collective_size", type=int, default=512 * 1024 * 1024)
    p.add_argument("--devices_per_node", type=int, default=8)
    p.add_argument("--output_dir", default="profiling_outputs")
    p.add_argument("--port", type=int, default=29511)
    a = p.parse_args()
    a.output_dir = f"{a.output_dir}/collective/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(a.output_dir, exist_ok=True)
    return a


def main():
    args = parse_args()
    devices = [int(x) for x in args.devices.split(",") if x != ""]
    global_ids = ([int(x) for x in args.global_ids.split(",")]
                  if args.global_ids else devices)
    world_size = len(devices)

    if world_size < 2:
        raise SystemExit("collectives need >= 2 devices")
    if args.collective == "send_recv" and world_size != 2:
        raise SystemExit("send_recv is defined for exactly 2 devices (vidur's convention)")

    link = args.link_type
    sizes = get_collectives_sizes_to_profile(args.max_collective_size)
    print(f"{args.collective}: devices={devices} global={global_ids} "
          f"link={link} world_size={world_size} sizes={len(sizes)}")

    import json, tempfile
    fd, res_path = tempfile.mkstemp(suffix=".json", prefix="coll_")
    os.close(fd)
    mp.spawn(
        _worker,
        args=(world_size, devices, sizes, args.collective, args.port, res_path),
        nprocs=world_size,
        join=True,
    )
    with open(res_path) as f:
        raw = json.load(f)
    os.unlink(res_path)

    rows = []
    for r in raw:
        rows.append({
            "time_stats": r["time_stats"],
            "rank": 0,
            "num_workers": world_size,
            "size": r["size"],
            "collective": args.collective,
            "devices_per_node": world_size,
            "max_devices_per_node": args.devices_per_node,
            "link_type": link,
            "devices": "-".join(map(str, global_ids)),
        })

    df = results_to_dataframe(rows)
    path = f"{args.output_dir}/{args.collective}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
