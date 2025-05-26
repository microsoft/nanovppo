#!/usr/bin/env python
"""Run end-to-end test for GRPO algorithm."""

import os
import sys
import unittest

# Add the repository root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tests.test_grpo import TestGRPO

if __name__ == "__main__":
    # Run the GRPO test
    unittest.main(defaultTest="TestGRPO")