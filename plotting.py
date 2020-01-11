from matplotlib import pyplot as plt


def plot_metric(pipeline, metric_name, xlabel, ylabel, title):
    accuracies = pipeline.get_epoch_metric_train(metric_name)
    plt.plot(range(len(accuracies)), accuracies)

    accuracies = pipeline.get_epoch_metric_validation(metric_name)
    plt.plot(range(len(accuracies)), accuracies)

    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()