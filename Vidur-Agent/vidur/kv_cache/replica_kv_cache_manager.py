from typing import Optional

from vidur.entities import Request
from vidur.kv_cache.base_kv_cache_manager import KVCacheManager
from vidur.kv_cache.kv_cache_block import KVCacheBlock
from vidur.logger import init_logger
from vidur.utils import cdiv

logger = init_logger(__name__)


class ReplicaKVCacheManager(KVCacheManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def allocate_slots_with_disk_cache(
        self,
        request: Request,
        num_tokens: int,
        num_disk_computed_blocks: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            num_disk_computed_blocks: The number of blocks that are cached in the
                disk for this request
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < disk computed >    | < new > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []

        assert num_disk_computed_blocks >= len(
            new_computed_blocks
        ), f"Cached blocks in disk ({num_disk_computed_blocks}) can not be less than local ({len(new_computed_blocks)})"

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (
            request.num_processed_tokens + num_disk_computed_blocks * self.block_size
        )

        num_required_blocks = cdiv(num_computed_tokens + num_tokens, self.block_size)

        req_blocks = self.req_to_blocks[request.id]
        num_new_blocks = (
            num_required_blocks - len(req_blocks) - len(new_computed_blocks)
        )

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk in new_computed_blocks if blk.ref_cnt == 0
        )
        if (
            num_new_blocks
            > self.block_pool.get_num_free_blocks() - num_evictable_computed_blocks
        ):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when " "prefix caching is disabled"
            )

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

        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(
            request.id, len(new_computed_blocks)
        )
        # We only cache blocks with generated (accepted) tokens.
        num_full_blocks_after_append = (
            num_computed_tokens + num_tokens
        ) // self.block_size

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
