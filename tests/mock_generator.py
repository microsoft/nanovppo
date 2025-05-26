"""Mock implementation of the GeneratorClient for testing."""

from typing import List, Optional, Tuple, Any


class MockGeneratorClient:
    """Mock implementation of GeneratorClient for testing GRPO algorithm."""

    _instance = None

    def __init__(self, responses=None, finished=None):
        """Initialize mock generator with predetermined responses.
        
        Args:
            responses: List of responses to return from chat method
            finished: List of boolean flags indicating if responses are finished
        """
        self.responses = responses or ["This is a mock response."]
        self.finished = finished or [True]
        MockGeneratorClient._instance = self

    @classmethod
    def get(cls):
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = MockGeneratorClient()
        return cls._instance

    def chat(
        self,
        messages: List[List[dict]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 100,
        n: int = 1,
        allowed_token_ids: Optional[List[int]] = None,
        return_finished: bool = False,
    ) -> Tuple[List[List[str]], Optional[List[List[bool]]]]:
        """Mock chat method that returns predetermined responses.
        
        Args:
            messages: List of message sequences
            temperature: Sampling temperature (ignored in mock)
            top_p: Top-p sampling parameter (ignored in mock)
            max_tokens: Maximum number of tokens to generate (ignored in mock)
            n: Number of responses to generate per message
            allowed_token_ids: Allowed token IDs (ignored in mock)
            return_finished: Whether to return finished flags
            
        Returns:
            Tuple of (responses, finished) where each response is duplicated n times
        """
        # Create response structure expected by GRPO
        all_responses = []
        all_finished = []
        
        for _ in range(len(messages)):
            # Duplicate responses n times
            message_responses = [self.responses[0] for _ in range(n)]
            message_finished = [self.finished[0] for _ in range(n)]
            
            all_responses.append(message_responses)
            all_finished.append(message_finished)
            
        if return_finished:
            return all_responses, all_finished
        return all_responses

    def sleep(self):
        """Mock sleep method."""
        pass

    def wake_up(self):
        """Mock wake_up method."""
        pass