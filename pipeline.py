from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner
from steps.lstm_rnn_tensorflow_model_wrapper import ClassificationRNNTensorFlowModel, N_CLASSES, BATCH_SIZE
from steps.one_hot_encoder import OneHotEncoder
from steps.transform_expected_output_wrapper import OutputTransformerWrapper


# TODO: wrap by a validation split wrapper as issue #174
# ValidationSplitWrapper(HumanActivityRecognitionPipeline)

class HumanActivityRecognitionPipeline(MiniBatchSequentialPipeline):
    def __init__(self):
        MiniBatchSequentialPipeline.__init__(self, [
            OutputTransformerWrapper(OneHotEncoder(nb_columns=N_CLASSES, name='one_hot_encoded_label')),
            ClassificationRNNTensorFlowModel(),
            Joiner(batch_size=BATCH_SIZE)
        ])
