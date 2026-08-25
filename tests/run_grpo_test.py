#!/usr/bin/env python
"""Run end-to-end test for GRPO algorithm.

This script runs the end-to-end tests for the GRPO (Generative Reward Policy Optimization) 
algorithm. It initializes the test environment and executes the TestGRPO test suite.

To run the tests:
    python -m tests.run_grpo_test
    
The tests verify that GRPO's core functionality works as expected, using mock objects
to avoid dependencies on actual language models.
"""

import os
import sys
import unittest

# Add the repository root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tests.test_grpo import TestGRPO

if __name__ == "__main__":
    # Run the GRPO test
    unittest.main(defaultTest="TestGRPO")