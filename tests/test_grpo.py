"""End-to-end test for GRPO algorithm.

This module contains unit tests for the GRPO (Generative Reward Policy Optimization) algorithm.
It tests the core functionality of the algorithm, particularly:
1. gather_episodes - Collects responses, calculates rewards and prepares tensors for training
2. compute_loss - Processes the tensors and calculates loss for optimization

The tests use mocking to isolate the GRPO functionality from external dependencies
like language models and reward functions.
"""

import os
import sys
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Add the repository root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.mock_generator import MockGeneratorClient
from tests.test_rewards import simple_reward_function
from algos.grpo import GRPO
from gen_utils import GeneratorClient


class TestGRPO(unittest.TestCase):
    """Test cases for GRPO algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample tensor shapes for testing
        self.batch_size = 4
        self.seq_length = 10
        self.vocab_size = 100
        
        # Mock the models and tokenizer for testing
        self.mock_model = MagicMock()
        
        # Mock logits output
        self.mock_output = MagicMock()
        self.mock_output.logits = torch.ones((self.batch_size, self.seq_length, self.vocab_size))
        self.mock_model.return_value = self.mock_output
        
        # Set up patches
        self.auto_model_patcher = patch(
            'algos.grpo.AutoModelForCausalLM.from_pretrained', 
            return_value=self.mock_model
        )
        self.auto_tokenizer_patcher = patch(
            'algos.grpo.AutoTokenizer.from_pretrained', 
            return_value=self._get_mock_tokenizer()
        )
        self.get_shifted_logprobs_patcher = patch(
            'utils.get_shifted_logprobs',
            return_value=torch.zeros((self.batch_size, self.seq_length-1))
        )
        
        # Start patches
        self.auto_model = self.auto_model_patcher.start()
        self.auto_tokenizer = self.auto_tokenizer_patcher.start()
        self.get_shifted_logprobs = self.get_shifted_logprobs_patcher.start()
        
        # Replace generator client for testing
        GeneratorClient.init = self._mock_init_generator
        GeneratorClient.get = self._mock_get_generator
        
        # Set up mock responses
        self.mock_responses = [
            "This is a test response with <reasoning>some reasoning</reasoning> and <answer>42</answer>",
            "This is another response with <reasoning>different reasoning</reasoning> and <answer>7</answer>"
        ]
        self.mock_generator = MockGeneratorClient(
            responses=self.mock_responses, 
            finished=[True, True]
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.auto_model_patcher.stop()
        self.auto_tokenizer_patcher.stop()
        self.get_shifted_logprobs_patcher.stop()

    def _mock_init_generator(self, backend, **kwargs):
        """Mock the GeneratorClient.init method."""
        return None

    def _mock_get_generator(self):
        """Mock the GeneratorClient.get method."""
        return self.mock_generator

    def _get_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        
        # Mock tokenizer methods
        def mock_encode(text):
            # Return a simple token sequence
            return [1, 2, 3, 4, 5]
        
        def mock_apply_chat_template(messages, **kwargs):
            # Return simple token IDs
            return [1, 2, 3, 4, 5]
            
        def mock_call(texts):
            # Return a dict with input_ids for each text
            return {"input_ids": [[1, 2, 3, 4, 5] for _ in texts]}
            
        tokenizer.encode = mock_encode
        tokenizer.apply_chat_template = mock_apply_chat_template
        tokenizer.__call__ = mock_call
        return tokenizer

    def test_grpo_gather_episodes(self):
        """Test the gather_episodes method of GRPO."""
        # Create GRPO instance
        grpo = GRPO(
            model_name="mock_model",
            reward_func=simple_reward_function,
            k=2,  # Generate 2 responses per message
            temperature=0.0,
            max_tokens=100,
            device="cpu",
            kl_ctl=0.0001,
            kl_max=10,
            drgrpo=0,
            rpp=0,
            vocab=None
        )
        
        # Create sample messages and labels
        messages = [
            [{"role": "user", "content": "What is the meaning of life?"}],
            [{"role": "user", "content": "What is 2+2?"}]
        ]
        labels = ["42", "4"]
        
        # Patch utils.flatten to work correctly with our mocked values
        with patch('algos.grpo.flatten', side_effect=lambda x: [item for sublist in x for item in sublist]), \
             patch('algos.grpo.create_joint_tensors', return_value=(
                torch.ones((self.batch_size, self.seq_length), dtype=torch.long),  # query_response
                torch.ones((self.batch_size, self.seq_length), dtype=torch.long),  # query_response_mask
                torch.ones((self.batch_size, self.seq_length), dtype=torch.long)   # response_mask
             )), \
             patch('algos.grpo.get_logprobs', return_value=torch.zeros((self.batch_size, self.seq_length-1))):
            # Call gather_episodes
            episode_data = grpo.gather_episodes(messages, labels)
            
            # Check return values
            self.assertEqual(len(episode_data), 6, "Should return 6 tensors")
            
            # Unpack return values for easier inspection
            (query_response, query_response_mask, response_mask, 
             advantages, ref_logprobs, old_logprobs) = episode_data
             
            # Check types and shapes
            self.assertIsInstance(query_response, torch.Tensor)
            self.assertIsInstance(query_response_mask, torch.Tensor)
            self.assertIsInstance(response_mask, torch.Tensor)
            self.assertIsInstance(advantages, torch.Tensor)
            self.assertIsInstance(ref_logprobs, torch.Tensor)
            self.assertIsInstance(old_logprobs, torch.Tensor)
            
            # Verify tensor shapes
            self.assertEqual(query_response.shape, (self.batch_size, self.seq_length))
            self.assertEqual(query_response_mask.shape, (self.batch_size, self.seq_length))
            self.assertEqual(response_mask.shape, (self.batch_size, self.seq_length))
            self.assertEqual(advantages.shape, (self.batch_size,))
            self.assertEqual(ref_logprobs.shape, (self.batch_size, self.seq_length-1))
            self.assertEqual(old_logprobs.shape, (self.batch_size, self.seq_length-1))
            
    def test_grpo_compute_loss(self):
        """Test the compute_loss method of GRPO separately."""
        # Create GRPO instance
        grpo = GRPO(
            model_name="mock_model",
            reward_func=simple_reward_function,
            k=2,
            temperature=0.0,
            max_tokens=100,
            device="cpu",
            kl_ctl=0.0001,
            kl_max=10,
            drgrpo=0,
            rpp=0,
            vocab=None
        )
        
        # Create mock episode data
        episode_data = (
            torch.ones((self.batch_size, self.seq_length), dtype=torch.long),      # query_response
            torch.ones((self.batch_size, self.seq_length), dtype=torch.long),      # query_response_mask
            torch.ones((self.batch_size, self.seq_length), dtype=torch.long),      # response_mask
            torch.ones((self.batch_size,), dtype=torch.float32),                  # advantages
            torch.zeros((self.batch_size, self.seq_length-1), dtype=torch.float32), # ref_logprobs
            torch.zeros((self.batch_size, self.seq_length-1), dtype=torch.float32)  # old_logprobs
        )
        
        # Skip actual computation and verify interface only
        with patch.object(grpo.model, '__call__', return_value=self.mock_output):
            # We're just checking that the function runs without errors
            # We don't test the actual loss value as it depends on many factors
            loss = grpo.compute_loss(episode_data)
            self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()