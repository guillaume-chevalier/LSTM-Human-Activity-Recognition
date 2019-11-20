import numpy as np

from neuraxle.api.flask import JSONDataBodyDecoder


class CustomJSONDecoderFor2DArray(JSONDataBodyDecoder):
    """This is a custom JSON decoder class that precedes the pipeline's transformation."""

    def decode(self, data_inputs):
        """
        Transform a JSON list object into an np.array object.

        :param data_inputs: json object
        :return: np array for data inputs
        """
        return np.array(data_inputs)
