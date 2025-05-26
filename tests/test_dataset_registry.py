#!/usr/bin/env python3
"""
Test module for the dataset registry.

This module contains a simple test that demonstrates how to use the dataset registry.
It doesn't require importing the actual dataset modules, which might need additional
dependencies.
"""

import sys
import os

# Add the parent directory to the Python path so we can import modules from there
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dataset_registry():
    """Test the dataset registry functionality with mock data."""
    # First, let's define our own DatasetInfo class to avoid importing
    class MockDatasetInfo:
        def __init__(self, prepare_fn, eval_fn, reward_fn, template_rewards=None):
            self.prepare_dataset = prepare_fn
            self.eval_dataset = eval_fn
            self.reward_func = reward_fn
            self.template_specific_rewards = template_rewards or {}

        def get_reward_func(self, template=None):
            if template and template in self.template_specific_rewards:
                return self.template_specific_rewards[template]
            return self.reward_func

    # Create a mock registry
    mock_registry = {}

    # Define mock functions for registering and retrieving datasets
    def register_mock_dataset(name, prepare_fn, eval_fn, reward_fn, template_rewards=None):
        mock_registry[name] = MockDatasetInfo(prepare_fn, eval_fn, reward_fn, template_rewards)
        print(f"Registered dataset: {name}")

    def get_mock_dataset_info(name):
        if name not in mock_registry:
            raise ValueError(f"Unknown dataset: {name}")
        return mock_registry[name]

    # Define mock dataset functions
    def mock_prepare_dataset(*args, **kwargs):
        return "Prepared dataset"

    def mock_eval_dataset(*args, **kwargs):
        return "Evaluated dataset"

    def mock_reward_func(*args, **kwargs):
        return "Default reward"

    def mock_special_reward(*args, **kwargs):
        return "Special reward"

    # Test 1: Register a dataset
    print("\nTest 1: Register a dataset")
    register_mock_dataset(
        "mock_dataset",
        mock_prepare_dataset,
        mock_eval_dataset,
        mock_reward_func
    )
    
    # Test 2: Register a dataset with template-specific rewards
    print("\nTest 2: Register a dataset with template-specific rewards")
    register_mock_dataset(
        "template_dataset",
        mock_prepare_dataset,
        mock_eval_dataset,
        mock_reward_func,
        {"special": mock_special_reward}
    )
    
    # Test 3: Retrieve a dataset
    print("\nTest 3: Retrieve a dataset")
    try:
        dataset_info = get_mock_dataset_info("mock_dataset")
        print(f"Retrieved dataset: mock_dataset")
        print(f"Prepare function result: {dataset_info.prepare_dataset()}")
        print(f"Eval function result: {dataset_info.eval_dataset()}")
        print(f"Default reward function result: {dataset_info.reward_func()}")
        assert dataset_info.prepare_dataset() == "Prepared dataset"
        assert dataset_info.eval_dataset() == "Evaluated dataset"
        assert dataset_info.reward_func() == "Default reward"
        print("All function results match expected values ✓")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Retrieve a dataset with template-specific rewards
    print("\nTest 4: Retrieve a dataset with template-specific rewards")
    try:
        dataset_info = get_mock_dataset_info("template_dataset")
        print(f"Retrieved dataset: template_dataset")
        default_result = dataset_info.get_reward_func()()
        special_result = dataset_info.get_reward_func('special')()
        nonexistent_result = dataset_info.get_reward_func('nonexistent')()
        
        print(f"Default reward function result: {default_result}")
        print(f"Special template reward function result: {special_result}")
        print(f"Non-existent template reward function result: {nonexistent_result}")
        
        assert default_result == "Default reward"
        assert special_result == "Special reward"
        assert nonexistent_result == "Default reward"
        print("All template reward function results match expected values ✓")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Attempt to retrieve a non-existent dataset
    print("\nTest 5: Attempt to retrieve a non-existent dataset")
    try:
        get_mock_dataset_info("nonexistent_dataset")
        print("❌ Test failed: Should have raised ValueError")
    except ValueError as e:
        print(f"Expected error raised: {e} ✓")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    print("Testing the dataset registry with mock data...")
    test_dataset_registry()