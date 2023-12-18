import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output


sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 15})

LOG_PATH = "runs"


def plot_losses(
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
    title: str,
):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')
    axs[0].set_title(title + ' loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title(title + ' accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.savefig(f'{LOG_PATH}/{title}.png')
    plt.show()
