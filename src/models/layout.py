from abc import ABC, abstractmethod


class Layout(ABC):

    @abstractmethod
    def chain_of_thoughts(self, queries):
        return []

    @abstractmethod
    def similar_queries_fs(self, queries):
        return []

    @abstractmethod
    def similar_queries_zs(self, queries):
        return []
