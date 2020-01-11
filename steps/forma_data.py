import numpy as np
from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


class FormatData(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    def __init__(self, n_classes):
        NonFittableMixin.__init__(self)
        InputAndOutputTransformerMixin.__init__(self)
        BaseStep.__init__(self)
        self.n_classes = n_classes

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs

        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        if expected_outputs is not None:
            if not isinstance(expected_outputs, np.ndarray):
                expected_outputs = np.array(expected_outputs)

            if expected_outputs.shape != (len(data_inputs), self.n_classes):
                expected_outputs = np.reshape(expected_outputs, (len(data_inputs), self.n_classes))

        return data_inputs, expected_outputs
