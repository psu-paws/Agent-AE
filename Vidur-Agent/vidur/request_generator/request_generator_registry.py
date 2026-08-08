from vidur.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from vidur.request_generator.trace_request_generator import TraceRequestGenerator
from vidur.types import RequestGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
