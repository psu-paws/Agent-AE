from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional

from vidur.entities.request import Request
from vidur.kv_cache.kv_cache_block import KVCacheBlock
from vidur.kv_cache.kv_cache_block_pool import BlockPool
from vidur.kv_cache.utils import BlockHashType, hash_request_tokens
from vidur.logger import init_logger
from vidur.utils import cdiv

logger = init_logger(__name__)


@dataclass
class PrefixCacheStats:
    """Stores prefix cache hit statistics."""

    # Whether reset_prefix_cache was invoked.
    reset: bool = False
    # The number of requests in this update.
    requests: int = 0
    # The number of queries in these requests. Note that "queries" here
    # means the number of blocks that were queried from the cache.
    queries: int = 0
    # The number of hits in these requests.
    hits: int = 0


class KVCacheManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        enable_caching: bool,
        caching_hash_algo: str,
        num_preallocate_tokens: int,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

        self.block_pool = BlockPool(num_gpu_blocks, enable_caching)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[str, list[BlockHashType]] = defaultdict(
            list
        )

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats.
        """
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[list[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.id]
        if not block_hashes:
            block_hashes = hash_request_tokens(
                self.caching_hash_fn, self.block_size, request
            )
            self.req_to_block_hashes[request.id] = block_hashes

        self.prefix_cache_stats.requests += 1
        # Check for cache hits
        computed_blocks = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash
            # is not in the cached_block_hash_to_id, the following
            # block hashes are not computed yet for sure.
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break

        self.prefix_cache_stats.queries += len(block_hashes)
        self.prefix_cache_stats.hits += len(computed_blocks)

        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size
        return computed_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
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

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (
            request.num_processed_tokens + len(new_computed_blocks) * self.block_size
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

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.id, [])
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request.id, None)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.block_pool.reset_prefix_cache():
            self.prefix_cache_stats.reset = True
            return True
        return False

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.id, None)

    def register_brought_back_block(self, request: Request, block_index: int) -> None:
        """Register a KV block brought back from the decode replica for prefix caching.
        No-op when prefix caching is disabled or there are no free blocks.
        """
        if not self.enable_caching:
            return

        block_hashes = self.req_to_block_hashes[request.id]
        if not block_hashes:
            block_hashes = hash_request_tokens(
                self.caching_hash_fn, self.block_size, request
            )
            self.req_to_block_hashes[request.id] = block_hashes

        if block_index >= len(block_hashes):
            return

        block_hash = block_hashes[block_index]

        if self.block_pool.get_cached_block(block_hash) is not None:
            return

        if self.block_pool.get_num_free_blocks() == 0:
            return

        (new_block,) = self.block_pool.get_new_blocks(1)
        new_block.block_hash = block_hash
        self.block_pool.cached_block_hash_to_block[block_hash][new_block.block_id] = new_block
        self.block_pool.free_blocks([new_block])

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def get_allotted_blocks(self, request: Request) -> int:
        return len(self.req_to_blocks.get(request.id, []))
