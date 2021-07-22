import matplotlib.pyplot as plt
import numpy as np


def plot_path(model, path, X_test, y_test):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    """
    n_selected = []
    score = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        score.append(model.score(X_test, y_test))
        lambda_.append(save.lambda_)

    plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()

def plot_importance(state_dict: dict):
    width = 0.3
    abs_weights = np.abs(state_dict['skip.weight'][0].cpu().numpy())
    plt.barh(np.arange(len(abs_weights)),np.sort(abs_weights), width)
    plt.title("Feature importance")

    plt.yticks(np.arange(len(abs_weights)), np.argsort(abs_weights))
    plt.xticks(np.linspace(0, max(abs_weights), 5))

    plt.xlabel("Absolute skip layer weight corresponding to feature")
    plt.ylabel("Features")

    plt.grid(linewidth=1.1)

    plt.tight_layout()
    plt.show()
