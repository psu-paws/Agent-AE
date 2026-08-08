from typing import Optional

from vidur.entities import Request
from vidur.kv_cache.base_kv_cache_manager import KVCacheManager
from vidur.kv_cache.kv_cache_block import KVCacheBlock
from vidur.logger import init_logger

logger = init_logger(__name__)


class DiskKVCacheManager(KVCacheManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_computed_blocks(self, request):
        computed_blocks, _ = super().get_computed_blocks(request)
        max_prefill_blocks = request.num_prefill_tokens // self.block_size
        if len(computed_blocks) > max_prefill_blocks:
            computed_blocks = computed_blocks[:max_prefill_blocks]
        return computed_blocks, len(computed_blocks) * self.block_size

    def allocate_slots(
        self,
        request: Request,
        num_required_blocks: int,  # total number of blocks in the end for this request
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        if num_required_blocks == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []

        assert num_required_blocks >= len(
            new_computed_blocks
        ), "Cached blocks is more than required blocks"

        req_blocks = self.req_to_blocks[request.id]
        num_new_blocks = (
            num_required_blocks - len(req_blocks) - len(new_computed_blocks)
        )
        # Touch the computed blocks to make sure they won't be evicted.
        self.block_pool.touch(new_computed_blocks)

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
            )
            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(
            request.id, len(new_computed_blocks)
        )
        # For disk cache, all blocks will be cached
        num_full_blocks_after_append = num_required_blocks

        num_cached_after = self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_block[request.id] = num_cached_after
        return new_blocks
