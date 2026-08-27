# a100_nvlink_pcie

An 8-GPU A100 node whose tensor-parallel group is NVLink-connected but whose
prefill-to-decode KV handoff crosses PCIe.

- `all_reduce.csv` is `a100_dgx/all_reduce.csv` unmodified. All-reduce carries
  activations only.
- `send_recv.csv` takes the intra-node rows from `a100_pcie`.
  An agent-trace KV handoff moves several GB, well past where`a100_dgx` stops.
