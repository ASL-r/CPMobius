"""
Base class for a proposer
"""
from abc import ABC, abstractmethod

import torch

from verl import DataProto

__all__ = ['BaseProposer']


class BaseProposer(ABC):

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def propose_instruction(self) -> list[str]:
        """Propose instructions"""
        pass

    @abstractmethod
    def update_proposer(self, data: DataProto):
        """Update the proposer"""
        pass
