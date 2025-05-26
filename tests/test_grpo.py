"""End-to-end test for GRPO algorithm."""

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
        # Mock the models and tokenizer for testing
        self.mock_model = MagicMock()
        self.mock_model.return_value = MagicMock()
        
        # Mock logits output
        self.mock_output = MagicMock()
        self.mock_output.logits = torch.ones((1, 10, 100))
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
        
        # Start patches
        self.auto_model = self.auto_model_patcher.start()
        self.auto_tokenizer = self.auto_tokenizer_patcher.start()
        
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
        
        # Mock the create_joint_tensors function to return fixed tensors
        with patch('algos.grpo.create_joint_tensors', return_value=(
            torch.ones((4, 10), dtype=torch.long),  # query_response
            torch.ones((4, 10), dtype=torch.long),  # query_response_mask
            torch.ones((4, 10), dtype=torch.long)   # response_mask
        )), patch('algos.grpo.get_logprobs', return_value=torch.zeros((4, 9))):
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
            
            # Test compute_loss with the gathered episodes
            loss = grpo.compute_loss(episode_data)
            self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()