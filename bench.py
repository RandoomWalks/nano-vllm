# -----------------------------------------------------------------------------
# Benchmark Script for nano-vllm: Architecture Overview
#
# This script benchmarks the throughput of the nano-vllm LLM engine by generating
# a large batch of random input sequences and measuring the time taken to generate
# completions for all of them.
#
# Key Components:
#   - Random Prompt Generation:
#       * Generates a list of random token ID sequences to simulate diverse prompts.
#   - Sampling Parameters:
#       * Each sequence is assigned random sampling parameters (temperature, max_tokens, etc.).
#   - LLM Initialization:
#       * Loads the LLM model from a specified path with configurable settings.
#   - Warmup:
#       * Runs a dummy generation to warm up the model and CUDA kernels.
#   - Benchmark Run:
#       * Runs the actual batch generation and measures elapsed time.
#   - Throughput Calculation:
#       * Computes and prints the total number of tokens generated, total time, and throughput.
#
# Visualization:
#
# +-------------------+         +-------------------+         +-------------------+
# |  Random Prompts   |         |  Sampling Params  |         |   LLM Engine      |
# |-------------------|         |-------------------|         |-------------------|
# | - num_seqs        |         | - temperature     |         | - generate()      |
# | - input_len       |         | - max_tokens      |         | - model loading   |
# | - token_ids[][]   |         | - ignore_eos      |         | - batch infer     |
# +-------------------+         +-------------------+         +-------------------+
#         |                             |                              |
#         +-----------------------------+------------------------------+
#                                       |
#                                +-------------------+
#                                |   Benchmark Loop  |
#                                |-------------------|
#                                | - warmup          |
#                                | - timing          |
#                                | - throughput      |
#                                +-------------------+
#
# This script is designed to stress-test the LLM engine's batching and scheduling
# capabilities, providing a clear measure of end-to-end throughput under synthetic load.
# -----------------------------------------------------------------------------

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams  # Uncomment to benchmark vllm instead

def main():
    # Set random seed for reproducibility
    seed(0)

    # Benchmark parameters
    num_seqs = 256            # Number of sequences in the batch
    max_input_len = 1024      # Maximum length of each input prompt
    max_output_len = 1024     # Maximum number of tokens to generate per sequence

    # Model path (update as needed)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Initialize the LLM engine
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # Generate random prompt token IDs for each sequence
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]

    # Generate random sampling parameters for each sequence
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    # For vllm compatibility, uncomment the following line:
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # Warmup: run a dummy generation to initialize model and CUDA kernels
    llm.generate(["Benchmark: "], SamplingParams())

    # Benchmark: measure time to generate completions for all sequences
    t_start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t_elapsed = time.time() - t_start

    # Calculate total tokens generated and throughput
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t_elapsed

    # Print benchmark results
    print(f"Total: {total_tokens}tok, Time: {t_elapsed:.2f}s, Throughput: {throughput:.2f}tok/s")

if __name__ == "__main__":
    main()
