#!/usr/bin/env python3
"""
Integration test for the dataset registry.

This test verifies that the actual registry implementation in the codebase works as expected.
"""

import sys
import os

# Add the parent directory to the Python path so we can import modules from there
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.registry import (
    register_dataset,
    get_dataset_info,
    _DATASET_REGISTRY,
)


def test_registry_integration():
    """Test that the actual registry works as expected."""
    print("Testing the actual dataset registry implementation...")
    
    # Test 1: Check if default datasets are registered
    print("\nTest 1: Check if default datasets are registered")
    default_datasets = ["math", "gsm8k", "arc", "cd"]
    for dataset_name in default_datasets:
        if dataset_name in _DATASET_REGISTRY:
            print(f"✓ Dataset '{dataset_name}' is registered")
        else:
            print(f"❌ Dataset '{dataset_name}' is not registered")
    
    # Test 2: Test registering a new dataset
    print("\nTest 2: Test registering a new dataset")
    
    # Define dummy functions for testing
    def dummy_prepare(*args, **kwargs):
        return "Dummy prepared"
    
    def dummy_eval(*args, **kwargs):
        return "Dummy evaluated"
    
    def dummy_reward(*args, **kwargs):
        return "Dummy reward"
    
    def special_reward(*args, **kwargs):
        return "Special reward"
    
    # Register a new dataset
    register_dataset(
        "dummy_test",
        dummy_prepare,
        dummy_eval,
        dummy_reward,
        {"special": special_reward}
    )
    print("✓ Registered new dataset 'dummy_test'")
    
    # Test 3: Retrieve the registered dataset
    print("\nTest 3: Retrieve the registered dataset")
    try:
        dataset_info = get_dataset_info("dummy_test")
        print("✓ Retrieved dataset 'dummy_test'")
        
        # Check if the functions are correctly stored
        assert dataset_info.prepare_dataset == dummy_prepare
        assert dataset_info.eval_dataset == dummy_eval
        assert dataset_info.reward_func == dummy_reward
        assert "special" in dataset_info.template_specific_rewards
        assert dataset_info.template_specific_rewards["special"] == special_reward
        
        print("✓ All functions are correctly stored")
        
        # Test the reward function with different templates
        reward_func = dataset_info.get_reward_func()
        special_reward_func = dataset_info.get_reward_func("special")
        nonexistent_reward_func = dataset_info.get_reward_func("nonexistent")
        
        assert reward_func == dummy_reward
        assert special_reward_func == special_reward
        assert nonexistent_reward_func == dummy_reward
        
        print("✓ Template-specific reward functions work correctly")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Check for a non-existent dataset
    print("\nTest 4: Check for a non-existent dataset")
    try:
        get_dataset_info("nonexistent_dataset")
        print("❌ Failed to raise error for non-existent dataset")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test 5: Check math dataset for template-specific rewards
    print("\nTest 5: Check math dataset for template-specific rewards")
    try:
        math_info = get_dataset_info("math")
        templates = ["lcot", "cot", "qst"]
        
        for template in templates:
            if template in math_info.template_specific_rewards:
                print(f"✓ Math dataset has '{template}' template reward function")
            else:
                print(f"❌ Math dataset missing '{template}' template reward function")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nIntegration tests completed!")


if __name__ == "__main__":
    test_registry_integration()