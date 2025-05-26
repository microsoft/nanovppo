"""
Dataset registry for nanovppo.

This module provides a registry for datasets used in nanovppo. Each dataset is registered
with functions for preparing the dataset, evaluating the dataset, and computing rewards.

To add a new dataset, simply register it with the registry using register_dataset.
"""

from typing import Callable, Dict, List, Tuple, Any, Optional

from dataset.math_utils import (
    prepare_math_dataset, 
    eval_math,
    compute_math_reward,
    compute_lcot_math_reward,
    compute_qst_math_reward,
)
from dataset.gsm8k_utils import prepare_gsm8k_dataset, eval_gsm8k, compute_gsm8k_reward
from dataset.arc_utils import prepare_arc_dataset, eval_arc, compute_arc_reward
from dataset.cd_utils import prepare_cd_dataset, eval_cd, compute_cd_reward


class DatasetInfo:
    """Stores functions related to a dataset."""
    
    def __init__(
        self,
        prepare_dataset: Callable,
        eval_dataset: Callable,
        reward_func: Callable,
        template_specific_rewards: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize a DatasetInfo object.
        
        Args:
            prepare_dataset: Function to prepare the dataset
            eval_dataset: Function to evaluate the dataset
            reward_func: Default reward function for the dataset
            template_specific_rewards: Optional dict mapping template names to reward functions
        """
        self.prepare_dataset = prepare_dataset
        self.eval_dataset = eval_dataset
        self.reward_func = reward_func
        self.template_specific_rewards = template_specific_rewards or {}
    
    def get_reward_func(self, template: str = None) -> Callable:
        """
        Get the reward function for the specified template.
        
        Args:
            template: Template name to get the reward function for
            
        Returns:
            The reward function for the specified template, or the default reward function
        """
        if template and template in self.template_specific_rewards:
            return self.template_specific_rewards[template]
        return self.reward_func


# Registry mapping dataset names to DatasetInfo objects
_DATASET_REGISTRY: Dict[str, DatasetInfo] = {}


def register_dataset(
    name: str,
    prepare_dataset: Callable,
    eval_dataset: Callable,
    reward_func: Callable,
    template_specific_rewards: Optional[Dict[str, Callable]] = None,
) -> None:
    """
    Register a dataset with the registry.
    
    Args:
        name: Name of the dataset
        prepare_dataset: Function to prepare the dataset
        eval_dataset: Function to evaluate the dataset
        reward_func: Default reward function for the dataset
        template_specific_rewards: Optional dict mapping template names to reward functions
    """
    _DATASET_REGISTRY[name] = DatasetInfo(
        prepare_dataset, 
        eval_dataset, 
        reward_func,
        template_specific_rewards,
    )


def get_dataset_info(name: str) -> DatasetInfo:
    """
    Get the DatasetInfo for a dataset.
    
    Args:
        name: Name of the dataset
        
    Returns:
        DatasetInfo for the specified dataset
        
    Raises:
        ValueError: If the dataset is not registered
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return _DATASET_REGISTRY[name]


# Register the existing datasets
register_dataset(
    "math",
    prepare_math_dataset,
    eval_math,
    compute_math_reward,
    {
        "lcot": compute_lcot_math_reward,
        "cot": compute_math_reward,
        "qst": compute_qst_math_reward,
    },
)

register_dataset(
    "gsm8k",
    prepare_gsm8k_dataset,
    eval_gsm8k,
    compute_gsm8k_reward,
)

register_dataset(
    "arc",
    prepare_arc_dataset,
    eval_arc,
    compute_arc_reward,
)

register_dataset(
    "cd",
    prepare_cd_dataset,
    eval_cd,
    compute_cd_reward,
)