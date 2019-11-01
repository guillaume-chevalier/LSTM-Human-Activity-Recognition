from neuraxle.base import NonFittableMixin, MetaStepMixin, BaseStep, DataContainer, ExecutionContext


class TransformExpectedOutputWrapper(NonFittableMixin, MetaStepMixin, BaseStep):
    """
    Transform expected output wrapper step that can sends the expected_outputs to the wrapped step
    so that it can transform the expected outputs.
    """

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        new_expected_outputs_data_container = self.wrapped.handle_transform(
            DataContainer(
                current_ids=data_container.current_ids,
                data_inputs=data_container.expected_outputs,
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def transform(self, data_inputs):
        raise NotImplementedError('must be used inside a pipeline')
