from dataclasses import dataclass


@dataclass
class ExecutionTimePredictorRequest:
    num_processed_tokens: int
    num_tokens_to_process: int
    is_prefill_complete: bool
