# GRPO Mini End-to-End Test

This directory contains a minimal end-to-end test for the GRPO (Generative Reward Policy Optimization) algorithm implementation.

## Overview

The test suite verifies the core functionality of the GRPO algorithm by:

1. Testing the `gather_episodes` method which:
   - Generates completions via a generator client
   - Computes rewards for the completions
   - Creates tensors for model input/output
   - Calculates advantages
   
2. Testing the `compute_loss` method which:
   - Processes tensors returned by `gather_episodes`
   - Computes policy gradients and KL penalty
   - Returns a loss value for optimization

## Components

- `mock_generator.py`: A mock implementation of the `GeneratorClient` for testing purposes
- `test_rewards.py`: Simple reward functions for testing
- `test_grpo.py`: The actual test cases for GRPO
- `run_grpo_test.py`: Script to run the tests

## Running the Tests

To run the tests:

```bash
python -m tests.run_grpo_test
```

## Notes

- These tests use mocking to avoid the need for an actual language model or complex reward setup
- The mock generator returns predefined responses
- The tests validate the interfaces and expected return types/shapes rather than the actual values
- This ensures the core functionality is working without running a full training loop