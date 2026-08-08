import hashlib
import pickle


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def sha256(input) -> int:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        An integer representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")
