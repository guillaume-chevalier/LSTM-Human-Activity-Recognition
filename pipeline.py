from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner
from neuraxle.steps.encoding import OneHotEncoder
from neuraxle.steps.output_handlers import OutputTransformerWrapper

from steps.lstm_rnn_tensorflow_model_wrapper import ClassificationRNNTensorFlowModel, N_CLASSES, BATCH_SIZE


# TODO: wrap by a validation split wrapper as issue #174
# ValidationSplitWrapper(HumanActivityRecognitionPipeline)

class HumanActivityRecognitionPipeline(MiniBatchSequentialPipeline):
    def __init__(self):
        MiniBatchSequentialPipeline.__init__(self, [
            OutputTransformerWrapper(OneHotEncoder(nb_columns=N_CLASSES, name='one_hot_encoded_label')),
            ClassificationRNNTensorFlowModel(),
            Joiner(batch_size=BATCH_SIZE)
        ])
