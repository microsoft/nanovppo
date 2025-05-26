"""
Test the integration of run_bestofn.py with the dataset registry.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import argparse

# Add the parent directory to the path so we can import run_bestofn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to avoid dependency issues
sys.modules['dataset.registry'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['gen_utils'] = MagicMock()
sys.modules['algos.algo'] = MagicMock() 
sys.modules['algos.vppo'] = MagicMock()
sys.modules['algos.rft'] = MagicMock()
sys.modules['algos.grpo'] = MagicMock()
sys.modules['dataset.data_utils'] = MagicMock()
sys.modules['sgl_utils'] = MagicMock()
sys.modules['ddp_utils'] = MagicMock()
sys.modules['utils'] = MagicMock()

# Now that we've mocked all dependencies, import modules
from dataset.registry import get_dataset_info
import numpy as np


class TestRunBestofnIntegration(unittest.TestCase):
    """Test the integration of run_bestofn.py with the dataset registry."""

    def test_run_bestofn_uses_registry(self):
        """Test that run_bestofn uses the dataset registry."""
        # Since we need to mock many dependencies, we'll just check that the registry import exists
        try:
            # Check if the file correctly uses the registry
            with open('/home/runner/work/nanovppo/nanovppo/run_bestofn.py', 'r') as f:
                content = f.read()
                self.assertIn('from dataset.registry import get_dataset_info', content)
                self.assertIn('dataset_info = get_dataset_info', content)
                
            # Check that the direct imports from datasets were removed
            self.assertNotIn('from dataset.math_utils import', content)
            self.assertNotIn('from dataset.gsm8k_utils import', content)
            self.assertNotIn('from dataset.arc_utils import', content)
            self.assertNotIn('from dataset.cd_utils import', content)
            
            # Verify that the registry pattern is used correctly
            self.assertIn('try:', content)
            self.assertIn('dataset_info = get_dataset_info(args.dataset)', content)
            self.assertIn('prepare_dataset = dataset_info.prepare_dataset', content)
            self.assertIn('reward_fn = dataset_info.get_reward_func', content)
            
            # Test passed if we got this far
            self.assertTrue(True)
            
        except ImportError as e:
            # Skip test if dependencies are missing
            self.skipTest(f"Skipping due to missing dependencies: {e}")
            
        # We don't need to mock the actual execution since we're just checking the imports


if __name__ == '__main__':
    unittest.main()