import os
from hashlib import sha256
from typing import Any, Callable, NamedTuple, Optional, Sequence

from vidur.entities.request import Request


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """

    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]


# The hash seed for the first block of the prefix block sequence.
#
# Even if the hash function is the builtin hash(), we use sha256 to generate
# the initial hash to simplify the code. This is not performance critical
# as it is done once per process.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed.
# This aligns with the behavior of Python's hash() function, which also uses
# a random seed if PYTHONHASHSEED is not set.
# TODO (t-nitinkedia): NONE_HASH is calculated before set_seeds() in vidur/main.py runs.
# and hence "PYTHONHASHSEED" is not set. This is not a problem for now as we supply external
# block_hash_ids in the trace files.
NONE_HASH = (
    int.from_bytes(os.urandom(32), byteorder="big")
    if os.getenv("PYTHONHASHSEED") is None
    else sha256(os.getenv("PYTHONHASHSEED").encode())
)


def hash_block_tokens(
    hash_function: Callable,
    parent_block_hash: Optional[int],
    curr_block_token_ids: Sequence[int],
) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash_function((parent_block_hash, curr_block_token_ids_tuple)),
        curr_block_token_ids_tuple,
    )


def hash_request_tokens(
    hash_function: Any, block_size: int, request: Request
) -> list[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Sources are tried in order:

    1. ``token_ids`` — real token IDs from the trace. Preferred: hashes are
       chained over actual token content, so blocks are shared only when the
       token sequences genuinely match, and block boundaries can be re-derived
       at any block_size.
    2. ``block_hash_ids`` — pre-computed per-block hashes from the trace. Coarser:
       the block boundaries are fixed by whoever produced the trace.
    3. Neither — the request cannot participate in prefix caching; return no
       hashes so it is treated as a full cache miss.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.token_ids

    if not token_ids:
        if request.block_hash_ids:
            return [
                BlockHashType(hash_value, tuple())
                for hash_value in request.block_hash_ids
            ]
        # No token-level or block-level identity available for this request.
        return []

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        block_hash = hash_block_tokens(
            hash_function, parent_block_hash_value, block_token_ids
        )
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret
