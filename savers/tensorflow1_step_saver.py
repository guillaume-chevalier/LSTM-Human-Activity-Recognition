import os
import tensorflow as tf

from neuraxle.base import BaseSaver


class TensorflowV1StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Saver().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.

    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/saved_model>`_
    """

    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save a step that is using tf.train.Saver().

        :param step: step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return: saved step
        """
        with step.get_graph().as_default():
            saver = tf.train.Saver()
            saver.save(
                step.get_session(),
                self._get_saved_model_path(context, step)
            )

            step.strip()

        return step

    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load a step that is using tensorflow using tf.train.Saver().

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        step.is_initialized = False
        step.setup()
        with step.get_graph().as_default():
            saver = tf.train.Saver()
            saver.restore(
                step.get_session(),
                self._get_saved_model_path(context, step)
            )

        return step

    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        """
        Returns whether or not we can load.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        meta_exists = os.path.exists(os.path.join(context.get_path(), "{0}.ckpt.meta".format(step.get_name())))
        index_exists = os.path.exists(os.path.join(context.get_path(), "{0}.ckpt.index".format(step.get_name())))
        return meta_exists and index_exists

    def _get_saved_model_path(self, context, step):
        """
        Returns the saved model path using the given execution context, and step name.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return os.path.join(
            context.get_path(),
            "{0}.ckpt".format(step.get_name())
        )
