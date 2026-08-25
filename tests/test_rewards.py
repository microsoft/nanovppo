"""Simple reward functions for testing.

This module provides simplified reward functions for testing GRPO and other
reinforcement learning algorithms in the nanovppo library. These functions
calculate rewards based on simple criteria like word count, avoiding the
need for complex evaluation metrics during testing.
"""

from typing import List

from algos.algo import Request


def simple_reward_function(requests: List[Request]) -> List[float]:
    """A simple reward function that assigns rewards based on word count.
    
    Args:
        requests: List of Request objects
        
    Returns:
        List of reward values (1.0 for non-empty responses, 0.0 for empty ones)
    """
    rewards = []
    for request in requests:
        # Assign reward based on simple criteria (e.g., response length)
        word_count = len(request.response.split())
        
        # Simple reward: 0.0 for empty responses, up to 2.0 for longer ones
        reward = min(2.0, max(0.0, word_count / 10.0))
        rewards.append(reward)
    
    return rewards