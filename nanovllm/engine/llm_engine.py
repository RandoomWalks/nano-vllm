import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
import torch
import torch.multiprocessing as mp
from typing import Optional

# Lazy import for transformers to reduce startup time
_AutoTokenizer = None

def _get_tokenizer():
    global _AutoTokenizer
    if _AutoTokenizer is None:
        from transformers import AutoTokenizer
        _AutoTokenizer = AutoTokenizer
    return _AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

# -----------------------------------------------------------------------------
# LLMEngine: Architecture Overview
#
# The LLMEngine class orchestrates the end-to-end process of large language model
# inference, managing model runners, scheduling, tokenization, and request handling.

#? Tensor parallel rank refers to the index of a process in tensor parallelism, where a model is split across multiple processes/devices (usually GPUs), each handling a shard of the model.

#
# Key Components:
#   - ModelRunner: Handles the actual model computation, possibly across multiple
#     processes for tensor parallelism.
#   - Scheduler: Manages the queue of sequences (requests), schedules prefill and
#     decode steps, and handles postprocessing.
#   - Tokenizer: Converts between text and token IDs using HuggingFace's transformers.
#   - Config: Stores all configuration parameters for the engine and model.
#
# High-Level Workflow:
#   1. Initialization:
#      - Loads configuration and tokenizer.
#      - Spawns multiple ModelRunner processes (one per tensor parallel rank, except rank 0) to enable tensor parallelism.
#      - Initializes the Scheduler.
#   2. Request Handling:
#      - add_request: Accepts a prompt (string or token IDs) and sampling parameters,
#        encodes if necessary, wraps in a Sequence, and adds to the Scheduler.
#   3. Inference Loop:
#? Prefill is the initial phase where the model processes the full prompt for each new sequence, filling the key/value (KV) cache with prompt tokens before generating new tokens. This is done before the decode phase, which generates tokens one at a time.

#? Decode is the phase where the model generates new tokens one at a time for each sequence, using the KV cache filled during prefill to efficiently continue generation.

#      - step: Scheduler selects sequences for prefill or decode.
#      - ModelRunner(s) process the sequences and return token IDs.
#      - Scheduler postprocesses outputs, updates sequence states.
#      - Outputs are collected for finished sequences.
#   4. Generation:
#      - generate: High-level API for batch generation, iteratively calls step()
#        until all sequences are finished, optionally displaying progress.
#   5. Cleanup:
#      - exit: Ensures all ModelRunner processes are properly terminated.
#
# Visualization:
#
# +-------------------+         +-------------------+         +-------------------+
# |   LLMEngine       |<------->|   Scheduler       |<------->|   Sequence        |
# |-------------------|         |-------------------|         |-------------------|
# | - model_runner(s) |         | - queue           |         | - prompt          |
# | - scheduler       |         | - schedule()      |         | - sampling_params |
# | - tokenizer       |         | - postprocess()   |         | - completion      |
# | - config          |         +-------------------+         +-------------------+
# | - add_request()   |
# | - step()          |         +-------------------+
# | - generate()      |-------->|   ModelRunner     |<--------> (Model, GPU, etc.)
# | - exit()          |         +-------------------+
# +-------------------+
#
# This modular architecture enables efficient, parallel, and scalable LLM inference,
# supporting batching, streaming, and advanced scheduling for high-throughput serving.
# -----------------------------------------------------------------------------


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        AutoTokenizer = _get_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        
        # Initialize memory pools for better performance
        self._tensor_pool = {}
        self._max_pool_size = 10
        
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        # Clear tensor pool
        self._tensor_pool.clear()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # Early validation
        if not prompts:
            return []
        
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Batch add requests for better efficiency
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # Pre-allocate timing variables
        t_start = 0
        
        while not self.is_finished():
            t_start = perf_counter()
            output, num_tokens = self.step()
            
            if use_tqdm:
                elapsed = perf_counter() - t_start
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # Process outputs efficiently
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Batch decode tokens for better performance
        sorted_seq_ids = sorted(outputs.keys())
        token_ids_list = [outputs[seq_id] for seq_id in sorted_seq_ids]
        
        # Batch decode if tokenizer supports it
        if hasattr(self.tokenizer, 'batch_decode'):
            decoded_texts = self.tokenizer.batch_decode(token_ids_list, skip_special_tokens=False)
            outputs = [{"text": text, "token_ids": token_ids} 
                      for text, token_ids in zip(decoded_texts, token_ids_list)]
        else:
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} 
                      for token_ids in token_ids_list]
        
        if use_tqdm:
            pbar.close()
        return outputs
