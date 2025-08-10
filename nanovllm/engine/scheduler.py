from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# -----------------------------------------------------------------------------
# Scheduler: Architecture Overview
#
# The Scheduler class is responsible for managing the lifecycle and batching of
# inference requests (Sequences) for large language model serving. It orchestrates
# the flow of sequences through different states (waiting, running), determines
# which sequences are ready for prefill (initial prompt processing) or decode
# (incremental token generation), and interacts with the BlockManager to allocate
# and manage memory for KV cache blocks.
#
# Key Components:
#   - Sequence Queues:
#       * waiting: Sequences that are queued for processing but not yet running.
#       * running: Sequences that are actively being processed (prefill or decode).
#   - BlockManager:
#       * Handles allocation and management of KV cache blocks for each sequence.
#   - Scheduling Logic:
#       * Determines which sequences can be scheduled for prefill or decode based
#         on resource constraints (max sequences, max batched tokens, available blocks).
#       * Handles preemption and appending of tokens for running sequences.
#
# High-Level Workflow:
#   1. Adding Requests:
#       - New sequences are added to the waiting queue via add().
#   2. Scheduling (schedule()):
#       - Prefill Phase:
#           * Selects as many waiting sequences as possible (up to max_num_seqs and
#             max_num_batched_tokens), allocates KV cache blocks, and moves them to running.
#       - Decode Phase:
#           * For running sequences, checks if more tokens can be appended (using BlockManager).
#           * Handles preemption if resources are insufficient.
#           * Schedules eligible sequences for decoding.
#   3. Postprocessing:
#       - After model execution, updates sequence status and handles completion.
#   4. Completion Check:
#       - is_finished() returns True when all sequences are processed.
#
# Visualization:
#
# +-------------------+         +-------------------+         +-------------------+
# |   Scheduler       |<------->|   BlockManager    |<------->|   Block           |
# |-------------------|         |-------------------|         |-------------------|
# | waiting: [Seq]    |         | alloc/dealloc     |         | KV cache          |
# | running: [Seq]    |         | can_allocate()    |         | ref_count, hash   |
# | add(seq)          |         | allocate(seq)     |         +-------------------+
# | schedule()        |         | can_append(seq)   |
# | postprocess()     |         | may_append(seq)   |
# +-------------------+         +-------------------+
#         |   ^
#         |   | (add, schedule, postprocess)
#         v   |
#   (Sequences flow through waiting -> running -> finished)
#
# This architecture enables efficient batching, resource-aware scheduling, and
# high-throughput inference for LLM serving, while managing memory via block-based
# KV cache allocation.
# -----------------------------------------------------------------------------

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
