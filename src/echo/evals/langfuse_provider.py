"""Langfuse prompt provider for Echo SDK."""

import asyncio
import os
import logging
from typing import Callable, Optional

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig

from .base import BaseEvalProvider

logger = logging.getLogger(__name__)


class LangfuseEvalProvider(BaseEvalProvider):
    """Langfuse eval with lazy client initialization."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Langfuse client using env vars."""
        if self._client is None:
            try:
                from langfuse import Langfuse
            except ImportError:
                raise ImportError(
                    "langfuse required. Install: pip install 'echo[langfuse]'"
                )

            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if not public_key or not secret_key:
                raise ValueError(
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars"
                )

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
            )
        return self._client

    async def run_experiment(
        self,
        name: str,
        dataset_name: str,
        run_func: Callable,
        description: Optional[str] = None,
    ):
        """
        Fetch dataset from provider and run evaluation and upload the result

        Args:
            name: Name for the experiment
            dataset_name: Name for the dataset to be fetched
            description: Description of the experiment run
        """
        try:
            # Langfuse SDK is sync, run in executor
            loop = asyncio.get_event_loop()
            langfuse_dataset = await loop.run_in_executor(
                None, lambda: self.client.get_dataset(dataset_name)
            )
            
            langfuse_dataset.run_experiment(
                name=name,
                description=description,
                task=run_func,
            )
        except Exception as e:
            logger.error(f"Failed to fetch dataset '{dataset_name}': {e}")
