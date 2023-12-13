import numpy as np
from pruning import Prune


class PruningRate(Prune):
    """Prunes model or layer by pruning rate.

    Sets to zero weights that are in the bottom Nth percentile of the weights
    in the object. The weights are in their absolute value.

    This class inherits methods and attributes from Prune.
    """

    def __init__(self, pruning_rate: float, **kwargs) -> None:
        """Initializes an instance of the class PruningRate."""
        super().__init__(**kwargs)

        if pruning_rate is None:
            raise Exception('Pruning rate in class object '
                            f'{self.__class__.__name__} can not be None.')

        self.pruning_rate: float = pruning_rate

    def generate_personalized_mask(self) -> np.array:
        """Generates a mask for pruning weights."""
        threshold: float = np.percentile(
            np.abs(self.flat_weight_array), self.pruning_rate)

        return np.array([w > threshold for w in self.weights], dtype=object)
