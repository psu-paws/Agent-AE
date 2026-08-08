# Log of Profiling Data

```text
for each model:
    for each compute device:
        for each TP in 1, 2, 4, 8:
            what is the chunk_size (max_tokens_in_batch) supported for MLP?
            what is the max context length and batch size supported for decode attention?
            what is the chunk sizes and max context length supported for prefill attention?

for each network device:
    what is maximum bytes of all_reduce data supported for TP 1, 2, 4, 8?
    what is the maximum bytes of send_recv data supported and between what?
```

## A100 80GB Compute Device

| Model / Device | A100 80GB TP1 | A100 80GB TP2 |  A100 80GB TP4 | A100 80GB TP8 |
| --- | --- | --- | --- | --- |
| `meta-llama/Meta-Llama-3-8B` | ✅ | ❌ | ❌ | ✅ |
| `meta-llama/Meta-Llama-3-70B` | ❌ | ❌ | ✅ | ✅ |

- For `8B`, max chunk_size (max_tokens_in_batch) supported for MLP is `32768` for all TPs.
- For `8B`, max chunk_size and max kv_cache_size supported for prefill attention is `4096` and `256k` for only TP1 and TP8.
- For `8B`, max context length and batch size supported for decode attention is `64k` and `512` for only TP 1 and 8.
- For `70B`, max chunk_size (max_tokens_in_batch) supported for MLP is `32768` for all TPs.
- For `70B`, max chunk_size and max kv_cache_size supported for prefill attention is `4096` and `256k` for only TP4 and TP8.
- For `70B`, max context length and batch size supported for decode attention is `64k` and `512` for only TP 4 and 8.

## H100 Compute Device

| Model / Device | H100 TP1 | H100 TP2 |  H100 TP4 | H100 TP8 |
| --- | --- | --- | --- | --- |
| `meta-llama/Meta-Llama-3-8B` | ✅ | ✅ | ✅ | ✅ |
| `meta-llama/Meta-Llama-3-70B` | ❌ | ✅ | ✅ | ✅ |

- For `8B`, max chunk_size (max_tokens_in_batch) supported for MLP is `16384` for all TPs.
- For `8B`, max chunk_size and max kv_cache_size supported for prefill attention is `8192` and `256k` respectively for all TPs.
- For `8B`, max context length and batch size supported for decode attention is `64k` and `512` for all TPs.
- For `70B`, max chunk_size (max_tokens_in_batch) supported for MLP is `16384` for all TPs (2, 4, and 8).
- For `70B`, max chunk_size and max kv_cache_size supported for prefill attention is `8192` and `256k` for all TPs (2, 4, and 8).
- For `70B`, max context length and batch size supported for decode attention is `64k` and `512` for all TPs (2, 4, and 8).

## A100 80GB DGX Network Device

`all_reduce` data is till `4096 * 8192` Bytes for TP 1, 2, 4, and 8.
`send_recv` data is till `4096 * 8192` Bytes between {GPU 0-1, GPU-01-23, GPU 0123-4567} and two nodes.

## H100 DGX Network Device

`all_reduce` data is till `4096 * 8192` Bytes for TP 1, 2, 4, and 8.
`send_recv` data is not verified.
