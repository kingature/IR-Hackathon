from abc import ABC, abstractmethod


class Layout(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def chain_of_thoughts(self, queries):
        return []

    @abstractmethod
    def similar_queries_fs(self, queries):
        return []

    @abstractmethod
    def similar_queries_zs(self, queries):
        return []
