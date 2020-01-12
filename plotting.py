from matplotlib import pyplot as plt


def plot_metric(metric_train, metric_validation=None, xlabel='x', ylabel='y', title='Metric'):
    plt.plot(range(len(metric_train)), metric_train)

    legend = ['training']
    if metric_validation is not None:
        plt.plot(range(len(metric_validation)), metric_validation)
        legend.append('validation')

    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)

    plt.legend(legend, loc='upper left')
    plt.show()
