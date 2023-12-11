from abc import ABC, abstractmethod


class Pruning(ABC):
    MODEL, LAYER = "model", "layer"

    def __init__(self, object_to_prune, **kwargs) -> None:
        """Initializes an instance of the Pruning class."""
        self.object_to_prune = object_to_prune
        self.object_type = self.check_object_type(self.object_to_prune)

        self.__dict__.update(kwargs)

    @staticmethod
    def check_object_type(object):
        """
        docstring
        """
        if isinstance(object, None):
            return Pruning.MODEL
        if isinstance(object, None):
            return Pruning.LAYER

        raise Exception('Passed object is not a Layer, nor a Model.')

    @abstractmethod
    def compute_mask(self):
        """
        docstring
        """
        pass

    def prune():
        """
        docstring
        """
        pass
