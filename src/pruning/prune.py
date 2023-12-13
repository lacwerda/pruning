from abc import ABC, abstractmethod
import numpy as np
from typing import List
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer


class Prune(ABC):
    """A class for pruning operations."""

    def __init__(self, object_to_prune: Sequential | Layer, **kwargs) -> None:
        """Initializes an instance of the Pruning class."""
        self.object_type = Prune.check_object_type(object_to_prune)

        # the object is cloned to be changed throughout the methods
        self.object: Sequential | Layer = keras.models.clone_model(
            object_to_prune)

        self.flat_weight_array: List[float] = self.get_flat_weight_array(
            self.object)
        self.weights: np.array= np.array(
            [w.numpy() for w in self.object.weights], dtype=object)

        self.mask: np.array = self.unpruned_mask(self.object)

        self.__dict__.update(kwargs)

    @staticmethod
    def check_object_type(object) -> str:
        """Define object's type."""
        if isinstance(object, Sequential):
            return Sequential
        if isinstance(object, Layer):
            raise Exception('Layers are not supported yet.')

        raise Exception('Passed object is not a Layer, nor a Model.')

    @staticmethod
    def get_flat_weight_array(object) -> List[float]:
        """Get flatten array of weights."""
        weight_array: list = [
            weight.numpy().flatten() for weight in object.weights]

        flatten_weight_list = list()
        _ = [flatten_weight_list.extend(weight) for weight in weight_array]

        return flatten_weight_list

    @staticmethod
    def unpruned_mask(object) -> np.array:
        """Generates initial unpruned mask filled with ones.

        The mask is later used to zero pruned weights. It is applied by
        calculating the Hadamard product between the weights in each layer and
        the mask.
        """
        resolve = [np.ones(
            weight_array.numpy().shape) for weight_array in object.weights]
        return np.array(resolve, dtype=object)

    @abstractmethod
    def generate_personalized_mask(self, **kwargs) -> np.array:
        """Calculate importance regarding what to prune.

        This method is abstract and is left to the user to decide the logic
        behind the mask being applied to the model.

        This method has to return an np.array of arrays in the shape of the mask
        to be applied to weights of the model.
        """
        pass

    def prune(self) -> (np.array, Sequential | Layer):
        """Prunes weights according to mask.

        This method returns the mask used to prune the object and the pruned
        object.
        """
        new_mask: np.array = self.generate_personalized_mask()
        self.mask = np.multiply(new_mask, self.mask)

        self.weights: np.array = np.multiply(self.weights, self.mask)
        self.object: Sequential | Layer = self.object.set_weights(self.weights)

        return self.mask, self.object

    def fine_tune(self):
        """Abstract fine-tuning method.

        The method can be overriten in order to accommodate specific fine-tuning
        or re-training logic. Or not. It is left for the user to decide if they
        want to keep the fine-tining logic inside the class.
        """
        pass
