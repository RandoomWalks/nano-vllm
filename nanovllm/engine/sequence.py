from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

# -----------------------------------------------------------------------------
# Sequence and SequenceStatus: Architecture Overview
#
# The Sequence and SequenceStatus classes together represent and manage the state
# of a single text generation request as it flows through the LLM inference engine.
#
# Key Concepts:
#   - Sequence:
#       * Encapsulates all information about a single generation request, including
#         its prompt tokens, sampling parameters, current status, and memory block usage.
#       * Tracks both prompt and generated (completion) tokens, as well as how many
#         tokens are cached in memory blocks for efficient reuse.
#       * Each Sequence is assigned a unique seq_id for tracking and scheduling.
#       * Supports properties for easy access to prompt/completion tokens, status,
#         and block-level memory management.
#   - SequenceStatus (Enum):
#       * Enumerates the possible states of a Sequence:
#           - WAITING: The sequence is queued and not yet running.
#           - RUNNING: The sequence is actively being processed (prefill or decode).
#           - FINISHED: The sequence has completed generation (e.g., hit EOS or max tokens).
#
# High-Level Workflow:
#   1. Creation:
#       - A Sequence is instantiated with a list of token IDs (prompt) and sampling parameters.
#       - It starts in the WAITING state, ready to be scheduled.
#   2. Scheduling:
#       - The Scheduler moves Sequences from WAITING to RUNNING as resources allow.
#       - The Sequence tracks which tokens are cached and which are newly generated.
#   3. Generation:
#       - As tokens are generated, they are appended to the Sequence's token_ids.
#       - The Sequence updates its status and memory block usage accordingly.
#   4. Completion:
#       - When generation is finished (EOS or max tokens), the Sequence status is set to FINISHED.
#       - The Sequence's completion_token_ids property provides the generated output.
#
# Visualization:
#
# +-------------------+         +-------------------+
# |   Sequence        |<------->|   SequenceStatus  |
# |-------------------|         |-------------------|
# | seq_id            |         | WAITING           |
# | status            |         | RUNNING           |
# | token_ids         |         | FINISHED          |
# | num_tokens        |         +-------------------+
# | num_prompt_tokens |
# | num_cached_tokens |
# | block_table       |
# | temperature       |
# | max_tokens        |
# | ignore_eos        |
# +-------------------+
#         |   ^
#         |   | (status transitions)
#         v   |
#   (Tracks prompt, completion, and memory usage)
#
# This design enables efficient, stateful management of each generation request,
# supporting batching, memory sharing, and flexible scheduling in high-throughput
# LLM inference systems.
# -----------------------------------------------------------------------------


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
