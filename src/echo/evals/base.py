"""Base classes for Echo SDK evaluation and datasets."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseEvalProvider(ABC):
    """Abstract base class for dataset and eval providers."""

    @abstractmethod
    async def run_experiment(
        self,
        name: str,
        dataset_name: str,
        run_func: Callable,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Fetch dataset from provider and run evaluation and upload the result

        Args:
            name: Name for the experiment
            dataset_name: Name for the dataset to be fetched
            description: Description of the experiment run
        """
        pass
